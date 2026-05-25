"""API tests for VADAR FastAPI endpoints."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["DATABASE_URL"] = f"sqlite:///{Path('test_api.db').resolve()}"
os.environ["MLFLOW_TRACKING_URI"] = f"{Path('mlruns_test').resolve()}"

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Health endpoint should return connected status and model metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_query_and_runs_endpoints() -> None:
    """Query should execute and runs endpoints should return persisted data."""
    query_response = client.post("/query", json={"query": "How many objects are in the scene?", "scene_id": "default"})
    assert query_response.status_code == 200
    query_payload = query_response.json()
    run_id = query_payload["run_id"]

    list_response = client.get("/runs")
    assert list_response.status_code == 200
    assert isinstance(list_response.json(), list)

    detail_response = client.get(f"/runs/{run_id}")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["run_id"] == run_id


def test_eval_summary_endpoint() -> None:
    """Eval summary endpoint should return aggregate metrics payload."""
    response = client.get("/eval/summary")
    assert response.status_code == 200
    payload = response.json()
    assert "total_runs" in payload
    assert "success_rate" in payload
    assert "avg_latency_ms" in payload
    assert "failure_breakdown" in payload
