"""Tests for evaluation harness helpers and seed initialization."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["DATABASE_URL"] = f"sqlite:///{Path('test_eval.db').resolve()}"

from backend.database import init_db, list_eval_queries
from eval.run_eval import _matches_expected, seed_eval_queries


def test_seed_eval_queries_inserts_twenty() -> None:
    """Evaluation seed operation should store 20 canonical eval queries."""
    init_db()
    seed_eval_queries()
    rows = list_eval_queries()
    assert len(rows) >= 20


def test_matches_expected_types() -> None:
    """Expected-type matching helper should validate supported value shapes."""
    assert _matches_expected(2, {"type": "int", "min": 1})
    assert _matches_expected("hello", {"type": "str"})
    assert _matches_expected([1, 2], {"type": "list"})
    assert _matches_expected(True, {"type": "bool"})
    assert _matches_expected(1.23, {"type": "float"})
    assert _matches_expected({"a": 1}, {"type": "dict"})
