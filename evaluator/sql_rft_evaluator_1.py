import os
import json
import math
from typing import Any, Dict, List
from pathlib import Path

import requests
from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor


MCP_SERVER_URL = "https://mcp-sql-rft-server-644257448872.us-central1.run.app"


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
    mcp_url = MCP_SERVER_URL

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


@evaluation_test(
    input_dataset=[
        str(
            Path(__file__).resolve().parents[1]
            / "datasets"
            / "final_rft_sql_train_no_assistant.jsonl"
        )
    ],
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
    max_dataset_rows=25,
)
def test_sql_rft_local_1(row: EvaluationRow) -> EvaluationRow:
    """
    Local evaluation test: uses SingleTurnRolloutProcessor to have the model produce SQL,
    then evaluates via MCP server against ground_truth.
    Run with: pytest evaluator/sql_rft_evaluator.py -vs
    Environment: export MCP_SERVER_URL=http://127.0.0.1:8080
    """
    if not row.messages or row.ground_truth is None:
        row.evaluation_result = EvaluateResult(
            score=0.0, 
            reason="Missing messages or ground_truth", 
            is_score_valid=False
        )
        return row

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
