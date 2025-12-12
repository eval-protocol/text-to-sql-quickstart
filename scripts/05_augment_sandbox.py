import os
import re
import json
import time
import pathlib
from typing import Any, Dict, List, Optional, Set, Type

import duckdb
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, create_model
from fireworks import LLM


def extract_tables(sql: str) -> Set[str]:
    sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\\*.*?\\*/", "", sql, flags=re.DOTALL)
    patterns = [
        r"(?:FROM|JOIN)\\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        r'(?:FROM|JOIN)\\s+"([^"]+)"',
        r"(?:FROM|JOIN)\\s+`([^`]+)`",
    ]
    tables: Set[str] = set()
    for p in patterns:
        tables.update(re.findall(p, sql, flags=re.IGNORECASE))
    keywords = {
        "select",
        "where",
        "group",
        "order",
        "having",
        "limit",
        "as",
        "on",
        "and",
        "or",
        "not",
        "in",
        "exists",
    }
    return {t for t in tables if t.lower() not in keywords}


def map_sql_type_to_python(sql_type: str) -> Type:
    s = str(sql_type).upper()
    if "DECIMAL" in s:
        return float
    if any(k in s for k in ("DOUBLE", "FLOAT", "REAL")):
        return float
    if any(k in s for k in ("BIGINT", "INT")):
        return int
    if any(k in s for k in ("VARCHAR", "TEXT", "STRING")):
        return str
    if "BOOLEAN" in s:
        return bool
    return str


def main() -> None:
    load_dotenv()
    root = pathlib.Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    synth_db = str(data_dir / "synthetic_openflights.db")
    queries_path = data_dir / "generated_queries.json"
    augment_marker = data_dir / ".augment_done"

    # Check if augmentation already done (skip)
    FORCE_REGEN = os.environ.get("FORCE_REGEN", "").lower() in ("1", "true", "yes")
    if augment_marker.exists() and not FORCE_REGEN:
        print(f"✓ Augmentation already done (marker: {augment_marker})")
        print("  Set FORCE_REGEN=1 to re-run")
        return

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is not set")
    llm = LLM(
        model="accounts/fireworks/models/deepseek-v3p1-terminus",
        deployment_type="serverless",
        api_key=api_key,
    )

    with open(queries_path, "r") as f:
        queries = json.load(f).get("queries", [])

    with duckdb.connect(synth_db, read_only=True) as con_ro:
        schema_df = con_ro.sql("DESCRIBE;").df()
    schema_md = schema_df.to_markdown(index=False)

    def table_cols(name: str) -> List[str]:
        row = schema_df[schema_df["name"] == name]
        return [] if row.empty else list(row.iloc[0]["column_names"])

    def table_types(name: str) -> List[str]:
        row = schema_df[schema_df["name"] == name]
        return [] if row.empty else list(row.iloc[0]["column_types"])

    def build_payload_model(tables: List[str]) -> Type[BaseModel]:
        fields: Dict[str, Any] = {}
        for t in tables:
            cols = table_cols(t)
            types = table_types(t)
            if not cols:
                continue
            row_fields = {
                c: (Optional[map_sql_type_to_python(ty)], None)
                for c, ty in zip(cols, types)
            }
            row_model = create_model(f"{t.capitalize()}Row", **row_fields)
            fields[t] = (List[row_model], [])
        if not fields:
            return create_model("RowsPayload", rows=(List[dict], []))
        return create_model("RowsPayload", **fields)

    def count_rows(con: duckdb.DuckDBPyConnection, sql: str) -> int:
        try:
            return con.sql(f"SELECT COUNT(*) AS c FROM ({sql}) AS t").fetchone()[0]
        except Exception:
            return -1

    MAX_ZERO_PCT = int(os.environ.get("TARGET_MAX_ZERO_PERCENT", "10"))
    BATCH = int(os.environ.get("AUGMENT_BATCH_SIZE", "10"))
    MAX_ROWS_PER_TABLE = int(os.environ.get("MAX_ROWS_PER_TABLE_PER_BATCH", "2"))

    with duckdb.connect(synth_db) as con:
        zero_idx = [i for i, q in enumerate(queries) if count_rows(con, q) == 0]
        total = len(queries)
        print(f"Initial zero-result: {len(zero_idx)}/{total}")
        processed: Set[int] = set()
        iteration = 0
        while True:
            iteration += 1
            cur_zero = [i for i, q in enumerate(queries) if count_rows(con, q) == 0]
            pct = (len(cur_zero) / total * 100) if total else 0.0
            print(
                f"[Iter {iteration}] zero-result: {len(cur_zero)}/{total} ({pct:.1f}%)"
            )
            if pct <= MAX_ZERO_PCT or not cur_zero:
                break
            pending = [i for i in cur_zero if i not in processed][:BATCH]
            if not pending:
                break
            processed.update(pending)
            group: Dict[str, List[int]] = {}
            for idx in pending:
                tset = sorted(extract_tables(queries[idx]))
                if not tset:
                    continue
                key = "|".join(tset)
                group.setdefault(key, []).append(idx)
            for key, idxs in group.items():
                tables = key.split("|")
                RowsPayload = build_payload_model(tables)
                rows_schema = RowsPayload.model_json_schema()
                # cap per table
                for t, spec in rows_schema.get("properties", {}).items():
                    if isinstance(spec, dict) and spec.get("type") == "array":
                        spec["maxItems"] = MAX_ROWS_PER_TABLE
                queries_sample = [queries[i] for i in idxs[:3]]
                user_prompt = f"""
Given this DuckDB schema and zero-result SQL queries, generate minimal new rows to make them return results.

Schema:
{schema_md}

Tables to populate: {tables}

Queries to satisfy (sample):
{chr(10).join("- " + q for q in queries_sample)}

Rules:
1) At most {MAX_ROWS_PER_TABLE} rows per table
2) Maintain referential integrity
3) Use realistic values and new unique IDs
4) Return only JSON matching the schema
""".strip()
                try:
                    resp = llm.chat.completions.create(
                        messages=[{"role": "user", "content": user_prompt}],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "RowsPayload",
                                "schema": rows_schema,
                            },
                        },
                        temperature=0.5,
                    )
                    content = resp.choices[0].message.content
                    payload = json.loads(content) if content else {}
                except Exception as e:
                    print(f"LLM error: {e}")
                    continue
                # insert rows
                inserted = 0
                for t in tables:
                    rows = payload.get(t, [])
                    if not rows:
                        continue
                    df = pd.DataFrame(rows[:MAX_ROWS_PER_TABLE])
                    cols = table_cols(t)
                    for c in cols:
                        if c not in df.columns:
                            df[c] = None
                    df = df[cols]
                    try:
                        con.register("new_rows_df", df)
                        con.execute(
                            f'INSERT INTO "{t}" SELECT * FROM new_rows_df EXCEPT SELECT * FROM "{t}"'
                        )
                        con.unregister("new_rows_df")
                        inserted += len(df)
                    except Exception as e:
                        print(f"Insert error for {t}: {e}")
                print(f"Inserted rows: {inserted} for tables {tables}")
                time.sleep(0.5)
        # Final dedupe
        tables = [r[0] for r in con.sql("SHOW TABLES;").fetchall()]
        for t in tables:
            before = con.sql(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            con.execute(
                f'CREATE OR REPLACE TABLE "{t}" AS SELECT DISTINCT * FROM "{t}"'
            )
            after = con.sql(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            if after != before:
                print(f"Dedup: {t} {before}->{after}")

    # Mark augmentation as done
    augment_marker.write_text("done")
    print("Augmentation complete.")


if __name__ == "__main__":
    main()
