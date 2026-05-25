"""MLflow tracking helpers for VADAR agent runs."""

from __future__ import annotations

import hashlib
import os
import uuid
from collections import Counter
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "vadar-agent-runs"


def _program_hash(program: Optional[str]) -> str:
    """Return SHA256 hash for synthesized program content."""
    content = (program or "").encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def init_tracking() -> None:
    """Initialize MLflow tracking URI and experiment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_run(
    query: str,
    synthesised_program: Optional[str],
    result: Any,
    success: bool,
    failure_reason: Optional[str],
    latency_ms: float,
    iterations: int,
    scene_id: str,
) -> str:
    """Log one run to MLflow and return MLflow run ID."""
    init_tracking()
    try:
        with mlflow.start_run():
            mlflow.set_tag("scene_id", scene_id)
            mlflow.log_param("query", query[:250])
            mlflow.log_param("synthesised_program_hash", _program_hash(synthesised_program))
            mlflow.log_param("result_preview", str(result)[:250])
            mlflow.log_metric("latency_ms", float(latency_ms))
            mlflow.log_metric("iterations", int(iterations))
            mlflow.log_metric("success", int(success))
            if failure_reason:
                mlflow.set_tag("failure_reason", failure_reason)
            active = mlflow.active_run()
            return active.info.run_id if active else str(uuid.uuid4())
    except Exception:
        return str(uuid.uuid4())


def get_experiment_summary() -> Dict[str, Any]:
    """Return aggregate summary across all MLflow runs in experiment."""
    init_tracking()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return {
            "mean_latency_ms": 0.0,
            "success_rate": 0.0,
            "most_common_failure_reason": None,
            "total_runs": 0,
        }

    runs = client.search_runs([experiment.experiment_id])
    if not runs:
        return {
            "mean_latency_ms": 0.0,
            "success_rate": 0.0,
            "most_common_failure_reason": None,
            "total_runs": 0,
        }

    latencies = [run.data.metrics.get("latency_ms", 0.0) for run in runs]
    successes = [run.data.metrics.get("success", 0.0) for run in runs]
    failures = Counter(run.data.tags.get("failure_reason") for run in runs if run.data.tags.get("failure_reason"))

    return {
        "mean_latency_ms": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "success_rate": float(sum(successes) / len(successes)) if successes else 0.0,
        "most_common_failure_reason": failures.most_common(1)[0][0] if failures else None,
        "total_runs": len(runs),
    }
