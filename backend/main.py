"""FastAPI backend for VADAR query execution, logging, and run analytics."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend.database import get_run, get_session, init_db, list_runs, save_run
from backend.schemas import EvalSummaryResponse, HealthResponse, QueryRequest, QueryResponse, RunDetailResponse, RunListItem
from backend.tracking import get_experiment_summary, init_tracking, log_run
from vadar.agent import MAX_ITERATIONS, run_agent_loop

app = FastAPI(title="VADAR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    """Initialize database and MLflow tracking infrastructure."""
    init_db()
    init_tracking()


@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest) -> QueryResponse:
    """Run agent loop for a user query and persist all run artifacts."""
    started_at = time.perf_counter()
    result = run_agent_loop(
        query=payload.query,
        scene_id=payload.scene_id,
        max_iterations=int(os.getenv("MAX_ITERATIONS", str(MAX_ITERATIONS))),
    )
    latency_ms = (time.perf_counter() - started_at) * 1000.0

    failure_reason = result.failure_reason
    valid_failures = {"max_iterations_exceeded", "synthesis_failed", "execution_error", "unsolvable_query", None}
    if failure_reason not in valid_failures:
        failure_reason = "execution_error"

    row = save_run(
        query=payload.query,
        synthesised_program=result.synthesised_program,
        result=result.result,
        success=result.success,
        failure_reason=failure_reason,
        iterations=result.iterations,
        latency_ms=latency_ms,
        scene_id=payload.scene_id,
    )

    log_run(
        query=payload.query,
        synthesised_program=result.synthesised_program,
        result=result.result,
        success=result.success,
        failure_reason=failure_reason,
        latency_ms=latency_ms,
        iterations=result.iterations,
        scene_id=payload.scene_id,
    )

    return QueryResponse(
        query=payload.query,
        synthesised_program=result.synthesised_program,
        result=result.result,
        success=result.success,
        failure_reason=failure_reason,
        iterations=result.iterations,
        latency_ms=latency_ms,
        run_id=row.id,
    )


@app.get("/runs", response_model=List[RunListItem])
def runs_endpoint() -> List[RunListItem]:
    """Return latest 50 runs sorted by newest first."""
    rows = list_runs(limit=50)
    return [
        RunListItem(
            run_id=row.id,
            query=row.query,
            success=row.success,
            latency_ms=row.latency_ms,
            failure_reason=row.failure_reason,
            timestamp=row.created_at,
        )
        for row in rows
    ]


@app.get("/runs/{run_id}", response_model=RunDetailResponse)
def run_detail_endpoint(run_id: str) -> RunDetailResponse:
    """Return full details for one run ID."""
    row = get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    result: Any = None
    if row.result_json:
        try:
            result = json.loads(row.result_json)
        except json.JSONDecodeError:
            result = row.result_json

    return RunDetailResponse(
        run_id=row.id,
        query=row.query,
        synthesised_program=row.synthesised_program,
        result=result,
        success=row.success,
        failure_reason=row.failure_reason,
        iterations=row.iterations,
        latency_ms=row.latency_ms,
        scene_id=row.scene_id,
        timestamp=row.created_at,
    )


@app.get("/health", response_model=HealthResponse)
def health_endpoint() -> HealthResponse:
    """Return liveness and backing dependency status."""
    db_status = "connected"
    session = get_session()
    try:
        session.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"
    finally:
        session.close()

    return HealthResponse(
        status="ok",
        model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022"),
        db=db_status,
    )


@app.get("/eval/summary", response_model=EvalSummaryResponse)
def eval_summary_endpoint() -> EvalSummaryResponse:
    """Return aggregate stats over stored run history."""
    rows = list_runs(limit=1_000_000)
    if not rows:
        return EvalSummaryResponse(total_runs=0, success_rate=0.0, avg_latency_ms=0.0, failure_breakdown={})

    total_runs = len(rows)
    success_count = sum(1 for row in rows if row.success)
    avg_latency = sum(float(row.latency_ms or 0.0) for row in rows) / total_runs
    breakdown = Counter(row.failure_reason for row in rows if row.failure_reason)

    _ = get_experiment_summary()

    return EvalSummaryResponse(
        total_runs=total_runs,
        success_rate=success_count / total_runs,
        avg_latency_ms=avg_latency,
        failure_breakdown=dict(breakdown),
    )
