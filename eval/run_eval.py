"""Evaluation harness for VADAR API against seeded spatial benchmark queries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from backend.database import init_db, list_eval_queries, save_eval_query, save_eval_result

EVAL_DATA: List[Tuple[str, Dict[str, Any], str]] = [
    ("How many objects are in the scene?", {"type": "int", "min": 1}, "counting"),
    ("What is the name of the largest object?", {"type": "str"}, "comparison"),
    ("Find all objects within 1 metre of the origin", {"type": "list"}, "proximity"),
    ("Is there any object above height 2.0?", {"type": "bool"}, "threshold"),
    ("What is the closest object to position [1,0,1]?", {"type": "str"}, "proximity"),
    ("Find all objects that are red", {"type": "list"}, "attribute"),
    ("What is the average height of all objects?", {"type": "float"}, "aggregation"),
    ("Are any two objects overlapping?", {"type": "bool"}, "overlap"),
    ("Find the object furthest from the origin", {"type": "str"}, "proximity"),
    ("List objects sorted by distance to [0,0,0]", {"type": "list"}, "sorting"),
    ("How many objects are taller than 1.5 units?", {"type": "int"}, "comparison"),
    ("What colour is the object at position [2,0,2]?", {"type": "str"}, "attribute"),
    ("Find all objects directly above the floor (y < 0.1)", {"type": "list"}, "containment"),
    ("What is the total volume of all objects combined?", {"type": "float"}, "aggregation"),
    ("Is the blue object to the left of the red object?", {"type": "bool"}, "relative"),
    ("Find objects that are not touching any other object", {"type": "list"}, "negation"),
    ("What is the bounding box of the entire scene?", {"type": "dict"}, "geometry"),
    ("Find all pairs of objects within 0.5 metres of each other", {"type": "list"}, "proximity"),
    ("Which object is most central in the scene?", {"type": "str"}, "comparison"),
    ("Describe the spatial layout of the scene in one sentence", {"type": "str"}, "description"),
]


def seed_eval_queries() -> None:
    """Insert the canonical 20 eval queries when not present."""
    init_db()
    existing_queries = {row.query for row in list_eval_queries()}
    for query, expected, category in EVAL_DATA:
        if query not in existing_queries:
            save_eval_query(query=query, expected_result_json=json.dumps(expected), category=category)


def _matches_expected(result: Any, expected: Dict[str, Any]) -> bool:
    """Return whether runtime result matches expected type/constraints."""
    expected_type = expected.get("type")
    type_map = {
        "int": int,
        "str": str,
        "list": list,
        "bool": bool,
        "float": float,
        "dict": dict,
    }
    py_type = type_map.get(expected_type)
    if py_type is None:
        return False

    if expected_type == "float" and isinstance(result, int):
        result = float(result)

    if not isinstance(result, py_type):
        return False

    if expected_type == "int" and "min" in expected:
        return int(result) >= int(expected["min"])

    return True


def run_evaluation(api_url: str) -> Dict[str, Any]:
    """Execute all evaluation queries against API and persist/print results."""
    seed_eval_queries()
    eval_queries = list_eval_queries()

    rows: List[Dict[str, Any]] = []
    failures = Counter()

    for idx, row in enumerate(eval_queries, start=1):
        expected = json.loads(row.expected_result_json)
        response = requests.post(
            f"{api_url.rstrip('/')}/query",
            json={"query": row.query, "scene_id": "default"},
            timeout=30,
        )
        payload = response.json()

        result = payload.get("result")
        passed = bool(response.status_code == 200 and payload.get("success") and _matches_expected(result, expected))
        failure_reason = payload.get("failure_reason")
        latency_ms = float(payload.get("latency_ms", 0.0))

        run_id = str(payload.get("run_id", ""))
        if run_id:
            save_eval_result(eval_query_id=row.id, run_id=run_id, passed=passed)

        if not passed:
            failures[failure_reason or "unknown"] += 1

        rows.append(
            {
                "index": idx,
                "query": row.query,
                "category": row.category,
                "pass": passed,
                "latency_ms": latency_ms,
                "failure_reason": failure_reason or "",
            }
        )

    passed_count = sum(1 for item in rows if item["pass"])
    total = len(rows)
    avg_latency = sum(item["latency_ms"] for item in rows) / total if total else 0.0
    accuracy = (passed_count / total) * 100 if total else 0.0

    results_path = Path(__file__).resolve().parent / "results.md"
    lines = [
        "| # | Query | Category | Pass | Latency(ms) | Failure Reason |",
        "|---|-------|----------|------|-------------|----------------|",
    ]
    for item in rows:
        lines.append(
            f"| {item['index']} | {item['query']} | {item['category']} | {item['pass']} | {item['latency_ms']:.2f} | {item['failure_reason']} |"
        )

    lines.append("")
    lines.append(
        f"**Overall accuracy: {accuracy:.2f}%  |  Avg latency: {avg_latency:.2f}ms  |  Failure rate: {100 - accuracy:.2f}%**"
    )

    results_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Failure breakdown: {dict(failures)}")

    return {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "failure_breakdown": dict(failures),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line args for evaluation script."""
    parser = argparse.ArgumentParser(description="Run VADAR eval harness")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL for running VADAR API")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.api_url)
