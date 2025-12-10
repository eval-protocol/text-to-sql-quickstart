"""
GEPA Prompt Optimization for Text-to-SQL

This script adapts the SQL RFT evaluator for GEPA prompt optimization.
It takes the existing @evaluation_test and optimizes the system prompt
to improve SQL generation quality.

Usage:
    cd text-to-sql-quickstart
    export MCP_SERVER_URL=http://127.0.0.1:8080  # or your MCP server URL
    python evaluator/sql_gepa_training.py
"""

import os
import json
import math
from typing import Any, Dict, List
from pathlib import Path

import requests
from eval_protocol.models import EvaluateResult, EvaluationRow, Message, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.training import GEPATrainer
from eval_protocol.training.gepa_utils import build_reflection_lm


# ============================================================================
# System prompt - this is what GEPA will optimize
# ============================================================================
SYSTEM_PROMPT = """You are an expert SQL data analyst.
Write a single DuckDB SQL query to answer the user's question based on the schema.
Return only the SQL text, no explanations, and avoid duplicates via GROUP BY when needed.

Schema:
|    | column_name    | column_type   | null   | key   | default   | extra   |
|---:|:---------------|:--------------|:-------|:------|:----------|:--------|
|  0 | airline_id     | INTEGER       | YES    | NULL  | NULL      | NULL    |
|  1 | name           | VARCHAR       | YES    | NULL  | NULL      | NULL    |
|  2 | alias          | VARCHAR       | YES    | NULL  | NULL      | NULL    |
|  3 | iata           | VARCHAR       | YES    | NULL  | NULL      | NULL    |
|  4 | icao           | VARCHAR       | YES    | NULL  | NULL      | NULL    |
|  5 | callsign       | VARCHAR       | YES    | NULL  | NULL      | NULL    |
|  6 | country        | VARCHAR       | YES    | NULL  | NULL      | NULL    |
|  7 | active         | VARCHAR       | YES    | NULL  | NULL      | NULL    |"""


# ============================================================================
# MCP Server interaction (from sql_rft_evaluator.py)
# ============================================================================
def _parse_duckdb_ascii(table: str) -> List[Dict[str, Any]]:
    lines = [ln for ln in table.strip().split("\n") if ln.strip() and not ln.startswith("+")]
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split("|")[1:-1]]
    data_lines = lines[1:]
    if data_lines:
        try:
            first_vals = [v.strip() for v in data_lines[0].split("|")[1:-1]]
            if len(first_vals) == len(headers) and all(v.isupper() for v in first_vals):
                data_lines = data_lines[1:]
        except Exception:
            pass
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


def execute_sql_via_mcp(sql_query: str) -> Dict[str, Any]:
    """Execute SQL via MCP server and return result or error."""
    mcp_url = os.getenv("MCP_SERVER_URL")
    if not mcp_url:
        return {"error": "MCP_SERVER_URL not set", "result": None}
    
    if not sql_query.strip():
        return {"error": "Empty SQL query", "result": None}
    
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    payload = {
        "id": "eval-1",
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"session": {"id": "stateless-eval"}, "name": "query", "arguments": {"query": sql_query}},
    }
    
    try:
        with requests.post(f"{mcp_url}/mcp/", headers=headers, json=payload, timeout=20, stream=True) as r:
            r.raise_for_status()
            resp = None
            for line in r.iter_lines():
                if line:
                    txt = line.decode("utf-8")
                    if txt.startswith("data:"):
                        js = txt[5:].strip()
                        if js:
                            resp = json.loads(js)
                            break
            if not resp:
                return {"error": "No event-stream JSON found", "result": None}
            if "error" in resp:
                return {"error": f"MCP error: {resp['error']}", "result": None}
            ascii_table = resp["result"]["content"][0]["text"]
            return {"result": _parse_duckdb_ascii(ascii_table), "error": None}
    except Exception as e:
        return {"error": f"MCP request failed: {e}", "result": None}


def compare_results(pred: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare predicted and ground truth SQL results."""
    def norm(v: Any) -> str:
        if v is None:
            return "None"
        if isinstance(v, float) and not (math.isinf(v) or math.isnan(v)) and v == int(v):
            v = int(v)
        return str(v)

    try:
        gt_vals = sorted([sorted(map(norm, r.values())) for r in ground_truth])
        pr_vals = sorted([sorted(map(norm, r.values())) for r in pred])
        ok = gt_vals == pr_vals
        return {
            "match": ok,
            "reason": "Results match" if ok else f"Mismatch: expected {ground_truth}, got {pred}"
        }
    except Exception as e:
        return {"match": False, "reason": f"Comparison error: {e}"}


# ============================================================================
# Dataset Loading
# ============================================================================
def _load_eval_rows(max_rows: int | None = None, include_test: bool = False) -> List[EvaluationRow]:
    """Load evaluation rows for GEPA training.
    
    Args:
        max_rows: Maximum number of rows to load (None for all)
        include_test: If False (default), only load train data. 
                     Test data (60 rows) is held out for final evaluation.
    """
    root = Path(__file__).resolve().parents[1]
    rows: List[EvaluationRow] = []
    
    # By default, only load TRAIN data for GEPA
    # Test data is held out and used by eval_baseline.py for fair comparison
    filenames = ["final_rft_sql_train_data.jsonl"]
    if include_test:
        filenames.append("final_rft_sql_test_data.jsonl")
    
    for filename in filenames:
        ds_path = root / "datasets" / filename
        if ds_path.exists():
            with open(ds_path, "r") as f:
                for line in f:
                    if max_rows is not None and len(rows) >= max_rows:
                        break
                    obj = json.loads(line)
                    messages = []
                    for m in obj.get("messages", []):
                        messages.append(Message(role=m.get("role", ""), content=m.get("content", "")))
                    rows.append(EvaluationRow(messages=messages, ground_truth=obj.get("ground_truth")))
            print(f"Loaded {filename}: {len(rows)} rows total")
    
    if not rows:
        print("Warning: No dataset files found!")
        print("Please run 'make all-data' to generate the dataset first.")
    
    return rows




def sql_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """Adapter for converting raw dataset rows to EvaluationRows."""
    converted: List[EvaluationRow] = []
    for r in rows:
        messages = []
        for m in r.get("messages", []):
            messages.append(Message(role=m.get("role", ""), content=m.get("content", "")))
        converted.append(EvaluationRow(messages=messages, ground_truth=r.get("ground_truth")))
    return converted


# ============================================================================
# Helper functions
# ============================================================================
def _coerce_messages_for_eval(row_messages: List[Any]) -> List[Dict[str, str]]:
    """Convert EvaluationRow.messages to simple {role, content} dicts."""
    out: List[Dict[str, str]] = []
    for m in row_messages:
        try:
            role = getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")
            content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                text = content[0].get("text", "")
            else:
                text = content if isinstance(content, str) else ""
            # Remove thinking content if present
            if "</think>" in text:
                text = text.split("</think>")[1]
            out.append({"role": role or "", "content": text})
        except Exception:
            continue
    return out


def _extract_sql_from_response(content: str) -> str:
    """Extract SQL from model response, handling markdown code blocks."""
    if not content:
        return ""
    
    content = content.strip()
    
    # Handle ```sql ... ``` blocks
    if "```sql" in content:
        parts = content.split("```sql")
        if len(parts) > 1:
            sql_part = parts[1].split("```")[0]
            return sql_part.strip()
    
    # Handle ``` ... ``` blocks (generic)
    if "```" in content:
        parts = content.split("```")
        if len(parts) > 1:
            return parts[1].strip()
    
    # Return as-is if no code blocks
    return content.strip()


def _analyze_result_differences(
    expected: List[Dict[str, Any]], 
    actual: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze differences between expected and actual SQL results."""
    analysis = {
        "column_name_issues": [],
        "row_count_match": len(expected) == len(actual),
        "expected_row_count": len(expected),
        "actual_row_count": len(actual),
        "value_match": False,
        "order_match": False,
        "suggestions": [],
    }
    
    if not expected and not actual:
        analysis["value_match"] = True
        analysis["order_match"] = True
        return analysis
    
    if not expected or not actual:
        if not expected and actual:
            analysis["suggestions"].append("Expected empty result but query returned data. Check WHERE/HAVING conditions.")
        elif expected and not actual:
            analysis["suggestions"].append("Query returned no results. Check table names, JOIN conditions, and WHERE clauses.")
        return analysis
    
    # Get column names
    expected_cols = set(expected[0].keys()) if expected else set()
    actual_cols = set(actual[0].keys()) if actual else set()
    
    # Check column name differences
    if expected_cols != actual_cols:
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        
        if missing_cols and extra_cols:
            # Try to match columns by position/value similarity
            for exp_col in missing_cols:
                for act_col in extra_cols:
                    # Check if values are similar (likely just different alias)
                    exp_vals = [str(r.get(exp_col)) for r in expected[:3]]
                    act_vals = [str(r.get(act_col)) for r in actual[:3]]
                    if exp_vals == act_vals:
                        analysis["column_name_issues"].append({
                            "expected": exp_col,
                            "got": act_col,
                            "suggestion": f"Use '{exp_col}' AS alias instead of '{act_col}'"
                        })
        
        if missing_cols:
            analysis["suggestions"].append(f"Missing columns in output: {missing_cols}. Add these to your SELECT.")
        if extra_cols and not missing_cols:
            analysis["suggestions"].append(f"Unexpected columns: {extra_cols}. Remove or rename these.")
    
    # Normalize and compare values (ignoring column names and order)
    def normalize_row(row):
        return tuple(sorted(str(v) for v in row.values()))
    
    def normalize_value(v):
        if v is None:
            return "None"
        if isinstance(v, float):
            # Round to handle precision differences
            return str(round(v, 2))
        return str(v)
    
    def normalize_row_rounded(row):
        return tuple(sorted(normalize_value(v) for v in row.values()))
    
    expected_values = sorted([normalize_row_rounded(r) for r in expected])
    actual_values = sorted([normalize_row_rounded(r) for r in actual])
    
    analysis["value_match"] = expected_values == actual_values
    
    # Check if order is the issue
    if analysis["value_match"]:
        # Values match when sorted, check if original order matches
        expected_ordered = [normalize_row_rounded(r) for r in expected]
        actual_ordered = [normalize_row_rounded(r) for r in actual]
        analysis["order_match"] = expected_ordered == actual_ordered
        
        if not analysis["order_match"]:
            analysis["suggestions"].append(
                "Results contain correct values but in wrong order. "
                "Check your ORDER BY clause - ensure it matches the expected sorting."
            )
    else:
        # Find specific value differences
        expected_set = set(expected_values)
        actual_set = set(actual_values)
        
        missing_rows = expected_set - actual_set
        extra_rows = actual_set - expected_set
        
        if missing_rows:
            analysis["suggestions"].append(
                f"Missing {len(missing_rows)} expected row(s). "
                "Check your WHERE conditions and JOINs."
            )
        if extra_rows:
            analysis["suggestions"].append(
                f"Query returned {len(extra_rows)} unexpected row(s). "
                "Your filters may be too permissive."
            )
    
    return analysis


def _build_feedback_text(
    *,
    is_match: bool,
    sql_query: str,
    ground_truth: Any,
    pred_result: List[Dict[str, Any]] | None,
    error: str | None,
) -> str:
    """Build detailed feedback text for GEPA optimization."""
    
    # Handle execution errors
    if error:
        feedback_parts = [f"❌ SQL EXECUTION FAILED: {error}"]
        
        # Add specific suggestions based on error type
        error_lower = error.lower()
        if "no such table" in error_lower or "does not exist" in error_lower:
            feedback_parts.append("SUGGESTION: Check table names. Available tables: airlines, airports, countries, planes, routes")
        elif "no such column" in error_lower or "column" in error_lower:
            feedback_parts.append("SUGGESTION: Check column names match the schema exactly.")
        elif "syntax" in error_lower:
            feedback_parts.append("SUGGESTION: Check SQL syntax - ensure proper DuckDB SQL format.")
        
        feedback_parts.append(f"YOUR QUERY: {sql_query[:300]}")
        return "\n".join(feedback_parts)
    
    # Handle success
    if is_match:
        return "✅ CORRECT: Your SQL query returns the expected results."
    
    # Analyze the differences in detail
    gt_list = ground_truth if isinstance(ground_truth, list) else []
    pred_list = pred_result if pred_result else []
    
    analysis = _analyze_result_differences(gt_list, pred_list)
    
    feedback_parts = ["❌ INCORRECT RESULTS"]
    
    # Row count info
    feedback_parts.append(
        f"ROW COUNT: Expected {analysis['expected_row_count']}, Got {analysis['actual_row_count']}"
    )
    
    # Column name issues (most actionable!)
    if analysis["column_name_issues"]:
        feedback_parts.append("\n⚠️ COLUMN ALIAS MISMATCH (critical):")
        for issue in analysis["column_name_issues"]:
            feedback_parts.append(f"  - Expected column '{issue['expected']}' but got '{issue['got']}'")
            feedback_parts.append(f"    FIX: {issue['suggestion']}")
    
    # Value/Order analysis
    if analysis["value_match"]:
        if not analysis["order_match"]:
            feedback_parts.append("\n✓ VALUES CORRECT but WRONG ORDER")
            feedback_parts.append("  FIX: Add or fix your ORDER BY clause")
    else:
        feedback_parts.append("\n✗ VALUE MISMATCH")
    
    # Specific suggestions
    if analysis["suggestions"]:
        feedback_parts.append("\n📝 SUGGESTIONS:")
        for suggestion in analysis["suggestions"]:
            feedback_parts.append(f"  • {suggestion}")
    
    # Show expected vs actual (truncated)
    if gt_list:
        expected_preview = str(gt_list[:2])[:200]
        feedback_parts.append(f"\nEXPECTED (first 2): {expected_preview}")
    if pred_list:
        actual_preview = str(pred_list[:2])[:200]
        feedback_parts.append(f"GOT (first 2): {actual_preview}")
    
    # Show the query for context
    feedback_parts.append(f"\nYOUR QUERY: {sql_query[:250]}...")
    
    return "\n".join(feedback_parts)


# ============================================================================
# Evaluation Test
# ============================================================================
@evaluation_test(
    input_rows=_load_eval_rows(max_rows=None),  # Only loads TRAIN data (183 rows), test is held out
    completion_params=[
        {
            "temperature": 0.0,
            "max_tokens": 32000,  # Large buffer for DeepSeek's verbose reasoning
            "model": "fireworks_ai/accounts/fireworks/models/deepseek-v3p1-terminus"
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    passed_threshold=0.0,
    num_runs=1,
    mode="pointwise",
    max_dataset_rows=None,  # Use full dataset for GEPA
)
def test_sql_gepa(row: EvaluationRow) -> EvaluationRow:
    """
    SQL evaluation test for GEPA optimization.
    
    This evaluates the model's SQL generation against ground truth results
    by executing the SQL via MCP server.
    """
    if not row.messages or row.ground_truth is None:
        row.evaluation_result = EvaluateResult(
            score=0.0, 
            reason="Missing messages or ground_truth", 
            is_score_valid=False
        )
        return row

    # Ensure MCP server URL is set
    if not os.getenv("MCP_SERVER_URL"):
        os.environ["MCP_SERVER_URL"] = "http://127.0.0.1:8080"

    # Get the assistant's response (last message should be assistant)
    assistant_msgs = [m for m in row.messages if m.role == "assistant"]
    if not assistant_msgs:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason="No assistant response found",
            is_score_valid=False
        )
        return row
    
    raw_content = assistant_msgs[-1].content or ""
    if isinstance(raw_content, list):
        raw_content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content)
    
    sql_query = _extract_sql_from_response(raw_content)
    
    if not sql_query:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason="Empty SQL query from model",
            is_score_valid=True
        )
        return row

    # Execute SQL via MCP
    mcp_result = execute_sql_via_mcp(sql_query)
    
    if mcp_result.get("error"):
        feedback = _build_feedback_text(
            is_match=False,
            sql_query=sql_query,
            ground_truth=row.ground_truth,
            pred_result=None,
            error=mcp_result["error"],
        )
        row.evaluation_result = EvaluateResult(
            score=0.0,
            reason=feedback,
            is_score_valid=True,
            metrics={
                "sql_execution": MetricResult(
                    score=0.0,
                    is_score_valid=True,
                    reason=mcp_result["error"],
                    data={"sql_query": sql_query},
                )
            },
        )
        return row

    # Compare results
    pred_result = mcp_result["result"]
    ground_truth = row.ground_truth if isinstance(row.ground_truth, list) else []
    comparison = compare_results(pred_result, ground_truth)
    
    score = 1.0 if comparison["match"] else 0.0
    
    feedback = _build_feedback_text(
        is_match=comparison["match"],
        sql_query=sql_query,
        ground_truth=ground_truth,
        pred_result=pred_result,
        error=None,
    )
    
    row.evaluation_result = EvaluateResult(
        score=score,
        reason=feedback,
        is_score_valid=True,
        metrics={
            "result_match": MetricResult(
                score=score,
                is_score_valid=True,
                reason=comparison["reason"],
                data={
                    "sql_query": sql_query,
                    "predicted_result": pred_result,
                    "ground_truth": ground_truth,
                },
            )
        },
    )
    return row


# ============================================================================
# GEPA Training Entry Point
# ============================================================================
if __name__ == "__main__":
    import asyncio
    
    print("=" * 80)
    print("TEXT-TO-SQL GEPA PROMPT OPTIMIZATION")
    print("=" * 80)
    
    # Check MCP server
    mcp_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8080")
    print(f"\nMCP Server URL: {mcp_url}")
    print("Make sure MCP server is running before starting training!")
    
    # Initialize trainer
    # Train data (183 rows) is split into train/val
    # Test data (60 rows) is held out separately in eval_baseline.py
    trainer = GEPATrainer(
        test_sql_gepa,
        train_ratio=0.7,   # 70% for training (~128 examples)
        val_ratio=0.3,     # 30% for validation (~55 examples)
        # test_ratio = 0% (test is held out in separate file)
        input_field="problem",   # Maps to user question
        output_field="answer",   # Maps to SQL query
        module_type="chain_of_thought",  # Use CoT for step-by-step SQL reasoning
    )
    
    # Use Fireworks model for reflection
    reflection_lm = build_reflection_lm("fireworks_ai/accounts/fireworks/models/deepseek-v3p1-terminus")

    print("\nStarting GEPA training...")
    optimized_program = trainer.train(
        num_threads=4,     # Reduced to avoid API overload
        track_stats=True,
        reflection_minibatch_size=50,  # Sample 50 examples per iteration for better feedback
        reflection_lm=reflection_lm,
        # Use explicit budget for more iterations
        auto=None,
        max_metric_calls=3000,  # Higher budget for more exploration
    )

    # Evaluate with DSPy
    print("\n=== DSPy Evaluation ===")
    print(trainer.evaluate(optimized_program))
    
    # Get optimized prompt
    print("\n=== Optimized System Prompt ===")
    optimized_prompt = trainer.get_optimized_system_prompt(optimized_program)
    print(optimized_prompt)
    
    # Optional: Run full EP evaluation
    # print("\n=== EP Evaluation (with tracing) ===")
    # results = trainer.run_ep_evaluation(optimized_program)
    # print(f"Final EP Score: {results['score']:.3f}")

