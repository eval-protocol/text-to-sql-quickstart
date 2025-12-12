#!/usr/bin/env python
"""
Utility script to strip assistant messages from a JSONL dataset.

Usage (from repo root):
    python scripts/strip_assistant_messages.py \
        datasets/final_gepa_rft_sql_train_data.jsonl \
        datasets/final_gepa_rft_sql_train_no_assistant.jsonl
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def strip_assistant_messages(in_path: str, out_path: str) -> None:
    """Read JSONL from in_path, write JSONL to out_path with assistant messages removed."""
    in_file = Path(in_path)
    out_file = Path(out_path)

    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    with in_file.open("r", encoding="utf-8") as fin, out_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj: Dict[str, Any] = json.loads(line)
            msgs: List[Dict[str, Any]] = obj.get("messages", [])

            # Keep only non-assistant messages (e.g. system + user)
            obj["messages"] = [m for m in msgs if m.get("role") != "assistant"]

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main(argv: List[str]) -> None:
    if len(argv) != 3:
        print("Usage: python scripts/strip_assistant_messages.py <input.jsonl> <output.jsonl>")
        raise SystemExit(1)

    _, in_path, out_path = argv
    strip_assistant_messages(in_path, out_path)


if __name__ == "__main__":
    main(sys.argv)



