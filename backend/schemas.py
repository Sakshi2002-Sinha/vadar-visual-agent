"""Pydantic schemas for VADAR backend API models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""

    query: str
    scene_id: str = Field(default="default")


class QueryResponse(BaseModel):
    """Response payload for /query endpoint."""

    query: str
    synthesised_program: str
    result: Any
    success: bool
    failure_reason: Optional[str]
    iterations: int
    latency_ms: float
    run_id: str


class RunListItem(BaseModel):
    """Compact run item for listing runs."""

    run_id: str
    query: str
    success: bool
    latency_ms: Optional[float]
    failure_reason: Optional[str]
    timestamp: datetime


class RunDetailResponse(BaseModel):
    """Detailed run response for one run ID."""

    run_id: str
    query: str
    synthesised_program: Optional[str]
    result: Any
    success: bool
    failure_reason: Optional[str]
    iterations: int
    latency_ms: Optional[float]
    scene_id: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health status response."""

    status: str
    model: str
    db: str


class EvalSummaryResponse(BaseModel):
    """Aggregate summary over stored runs."""

    total_runs: int
    success_rate: float
    avg_latency_ms: float
    failure_breakdown: Dict[str, int]
