import os
import time
import json
import math
import uuid
import decimal
import datetime
import pathlib
from typing import Any, Dict, List, Optional, Type

import duckdb
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, create_model
from fireworks import LLM


def map_sql_type_to_python(sql_type: str) -> Type:
    s = str(sql_type).upper()
    if "DECIMAL" in s:
        return decimal.Decimal
    if any(k in s for k in ("DOUBLE", "FLOAT", "REAL")):
        return float
    if any(k in s for k in ("BIGINT", "INT")):
        return int
    if any(k in s for k in ("VARCHAR", "TEXT", "STRING")):
        return str
    if "TIMESTAMP" in s:
        return datetime.datetime
    if "DATE" in s:
        return datetime.date
    if "TIME" in s:
        return datetime.time
    if "BOOLEAN" in s:
        return bool
    if any(k in s for k in ("BLOB", "BYTEA")):
        return bytes
    if "UUID" in s:
        return uuid.UUID
    return object


def main() -> None:
    load_dotenv()
    root = pathlib.Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    prod_db = str(data_dir / "prod_openflights.db")
    synth_db_path = data_dir / "synthetic_openflights.db"
    
    # Check if synthetic DB already exists (skip regeneration)
    FORCE_REGEN = os.environ.get("FORCE_REGEN", "").lower() in ("1", "true", "yes")
    if synth_db_path.exists() and not FORCE_REGEN:
        print(f"✓ Synthetic DB already exists at {synth_db_path}")
        print("  Set FORCE_REGEN=1 to regenerate")
        return

    with duckdb.connect(prod_db, read_only=True) as con_ro:
        schema_df = con_ro.sql("DESCRIBE;").df()
    schema_for_prompt = schema_df.to_markdown(index=False)

    # Build dynamic Pydantic models per table
    pydantic_models: Dict[str, Type[BaseModel]] = {}
    table_names = list(schema_df["name"].unique())
    for table_name in table_names:
        row = schema_df[schema_df["name"] == table_name].iloc[0]
        col_names = list(row["column_names"])
        col_types = list(row["column_types"])
        fields: Dict[str, Any] = {}
        for c, t in zip(col_names, col_types):
            fields[c] = (Optional[map_sql_type_to_python(t)], None)
        model_name = f"{table_name.capitalize()}Row"
        pydantic_models[table_name] = create_model(model_name, **fields)

    dataset_fields: Dict[str, Any] = {t: (List[m], ...) for t, m in pydantic_models.items()}
    SyntheticDataset = create_model("SyntheticDataset", **dataset_fields)

    TARGET_ROW_COUNT = int(os.environ.get("TARGET_ROW_COUNT", "100"))
    ROWS_PER_API_CALL = int(os.environ.get("ROWS_PER_API_CALL", "2"))
    TOTAL_ROW_COUNTS = {t: TARGET_ROW_COUNT for t in table_names}

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is not set")
    llm = LLM(model="accounts/fireworks/models/deepseek-v3p1-terminus", deployment_type="serverless", api_key=api_key)

    all_synthetic: Dict[str, List[Dict[str, Any]]] = {t: [] for t in table_names}
    chunk_row_counts = {t: ROWS_PER_API_CALL for t in table_names}

    base_prompt = f"""
You are a highly capable data generator. Create realistic, consistent synthetic rows for each table.
Respect referential integrity across tables. Return only JSON conforming to the provided schema.

Database schema (DuckDB DESCRIBE):
{schema_for_prompt}
""".strip()

    call_count = 0
    while not all(len(rows) >= TOTAL_ROW_COUNTS[t] for t, rows in all_synthetic.items()):
        call_count += 1
        existing_summary = ""
        if any(all_synthetic[t] for t in table_names):
            parts = ["Existing recent rows (do not duplicate):"]
            for t in table_names:
                rows = all_synthetic[t][-20:]  # recent sample
                if rows:
                    df = pd.DataFrame(rows)
                    if len(df.columns) > 10:
                        df = df.iloc[:, :10]
                    parts.append(f"\nTABLE {t}\n{df.to_markdown(index=False)}")
            existing_summary = "\n".join(parts)

        final_prompt = (
            base_prompt
            + "\n"
            + existing_summary
            + "\n\nGenerate new rows per table in this shape:\n"
            + json.dumps(chunk_row_counts, indent=2)
        )

        resp = llm.chat.completions.create(
            messages=[{"role": "user", "content": final_prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "SyntheticDataset", "schema": SyntheticDataset.model_json_schema()},
            },
            temperature=0.7,
        )
        choice = resp.choices[0]
        content = choice.message.content
        if not content or choice.finish_reason == "length":
            print(f"Chunk {call_count}: empty or truncated, skipping")
            time.sleep(1)
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Chunk {call_count}: JSON parse error {e}")
            time.sleep(1)
            continue

        for t, rows in payload.items():
            if isinstance(rows, list) and t in all_synthetic:
                all_synthetic[t].extend(rows)

        for t in table_names:
            print(f"  {t}: {len(all_synthetic[t])}/{TOTAL_ROW_COUNTS[t]}")
        time.sleep(1)

    # Deduplicate and trim
    for t in table_names:
        df = pd.DataFrame(all_synthetic[t]).drop_duplicates()
        all_synthetic[t] = df.to_dict("records")[: TOTAL_ROW_COUNTS[t]]

    # Write synthetic DB
    synth_db = str(data_dir / "synthetic_openflights.db")
    with duckdb.connect(synth_db) as con:
        for t in table_names:
            rows = all_synthetic[t]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            cols = list(schema_df[schema_df["name"] == t].iloc[0]["column_names"])
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            df = df[cols]
            con.execute(f'CREATE OR REPLACE TABLE "{t}" AS SELECT * FROM df')
    print(f"Synthetic DB written to: {synth_db}")


if __name__ == "__main__":
    main()
