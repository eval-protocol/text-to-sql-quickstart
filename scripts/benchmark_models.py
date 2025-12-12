import os
import json
import time
import pathlib
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv
from fireworks import LLM


def parse_duckdb_ascii(table_string: str) -> List[Dict[str, Any]]:
    lines = [
        ln
        for ln in table_string.strip().split("\n")
        if ln.strip() and not ln.startswith("+")
    ]
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split("|")[1:-1]]
    data_lines = lines[1:]
    out: List[Dict[str, Any]] = []
    for ln in data_lines:
        vals = [v.strip() for v in ln.split("|")[1:-1]]
        if len(vals) != len(headers):
            continue
        row: Dict[str, Any] = {}
        for k, v in zip(headers, vals):
            if v.upper() == "NULL" or v == "":
                row[k] = None
                continue
            try:
                if "." in v:
                    row[k] = float(v)
                else:
                    row[k] = int(v)
            except Exception:
                row[k] = v
        out.append(row)
    return out


def are_equal(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
    def norm(v: Any) -> str:
        return "None" if v is None else str(v)

    av = sorted([sorted(map(norm, r.values())) for r in a])
    bv = sorted([sorted(map(norm, r.values())) for r in b])
    return av == bv


def run_eval(
    llm: LLM,
    mcp_url: str,
    system_prompt: str,
    user_prompt: str,
    ground_truth: List[Dict[str, Any]],
) -> int:
    resp = llm.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    sql = (resp.choices[0].message.content or "").strip()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    payload = {
        "id": "eval",
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "session": {"id": "bench"},
            "name": "query",
            "arguments": {"query": sql},
        },
    }
    with requests.post(
        f"{mcp_url}/mcp/", headers=headers, json=payload, timeout=30, stream=True
    ) as r:
        r.raise_for_status()
        ev = None
        for line in r.iter_lines():
            if line and line.decode("utf-8").startswith("data:"):
                js = line.decode("utf-8")[5:].strip()
                if js:
                    ev = json.loads(js)
                    break
    if not ev or "error" in ev:
        return 0
    ascii_table = ev["result"]["content"][0]["text"]
    pred = parse_duckdb_ascii(ascii_table)
    return 1 if are_equal(pred, ground_truth) else 0


def main() -> None:
    load_dotenv()
    root = pathlib.Path(__file__).resolve().parents[1]
    ds_path = root / "datasets" / "final_rft_sql_test_data.jsonl"
    if not ds_path.exists():
        print("Test dataset not found. Run generation scripts first.")
        return
    mcp_url = os.getenv("MCP_SERVER_URL")
    if not mcp_url:
        print("MCP_SERVER_URL not set")
        return
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("FIREWORKS_API_KEY not set")
        return
    BASE = os.getenv("BASE_MODEL_ID", "accounts/fireworks/models/qwen2p5-7b")
    LARGE = os.getenv(
        "LARGE_BASE_MODEL_ID",
        "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
    )
    TUNED = os.getenv(
        "FINE_TUNED_MODEL_ID", "accounts/<your-account-id>/models/<your-model-id>"
    )
    llm_base = LLM(model=BASE, deployment_type="serverless", api_key=api_key)
    llm_large = LLM(model=LARGE, deployment_type="serverless", api_key=api_key)
    llm_tuned = LLM(model=TUNED, deployment_type="serverless", api_key=api_key)

    # Load dataset
    rows: List[Dict[str, Any]] = []
    with open(ds_path, "r") as f:
        for ln in f:
            rows.append(json.loads(ln))
    total = len(rows)
    print(f"Loaded {total} examples.")

    scores = {"base": 0, "large": 0, "tuned": 0}
    for i, item in enumerate(rows):
        system_prompt = item["messages"][0]["content"]
        user_prompt = item["messages"][1]["content"]
        gt = item["ground_truth"]
        scores["base"] += run_eval(llm_base, mcp_url, system_prompt, user_prompt, gt)
        time.sleep(0.5)
        scores["large"] += run_eval(llm_large, mcp_url, system_prompt, user_prompt, gt)
        time.sleep(0.5)
        scores["tuned"] += run_eval(llm_tuned, mcp_url, system_prompt, user_prompt, gt)
        time.sleep(0.5)
        if (i + 1) % 10 == 0:
            print(f"Progress {i + 1}/{total}")

    def pct(v: int) -> float:
        return (v / total) * 100 if total else 0.0

    print("Results:")
    print(f"Base  : {scores['base']}/{total} ({pct(scores['base']):.2f}%)")
    print(f"Large : {scores['large']}/{total} ({pct(scores['large']):.2f}%)")
    print(f"Tuned : {scores['tuned']}/{total} ({pct(scores['tuned']):.2f}%)")


if __name__ == "__main__":
    main()
