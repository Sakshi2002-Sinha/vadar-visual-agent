"""SQLAlchemy models and CRUD helpers for VADAR backend persistence."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vadar.db")

connect_args: Dict[str, Any] = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Run(Base):
    """Represents one end-to-end agent execution run."""

    __tablename__ = "runs"

    id = String(36),
    id = Base.metadata.tables.get("runs") if False else None

    id = __import__("sqlalchemy").Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query = __import__("sqlalchemy").Column(Text, nullable=False)
    synthesised_program = __import__("sqlalchemy").Column(Text, nullable=True)
    result_json = __import__("sqlalchemy").Column(Text, nullable=True)
    success = __import__("sqlalchemy").Column(Boolean, nullable=False)
    failure_reason = __import__("sqlalchemy").Column(String(64), nullable=True)
    iterations = __import__("sqlalchemy").Column(Integer, default=0, nullable=False)
    latency_ms = __import__("sqlalchemy").Column(Float, nullable=True)
    scene_id = __import__("sqlalchemy").Column(String(128), default="default", nullable=False)
    created_at = __import__("sqlalchemy").Column(DateTime, default=datetime.utcnow, nullable=False)

    eval_results = relationship("EvalResult", back_populates="run")


class EvalQuery(Base):
    """Stores canonical evaluation queries and expected outputs."""

    __tablename__ = "eval_queries"

    id = __import__("sqlalchemy").Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query = __import__("sqlalchemy").Column(Text, nullable=False)
    expected_result_json = __import__("sqlalchemy").Column(Text, nullable=False)
    category = __import__("sqlalchemy").Column(String(64), nullable=False)
    created_at = __import__("sqlalchemy").Column(DateTime, default=datetime.utcnow, nullable=False)

    eval_results = relationship("EvalResult", back_populates="eval_query")


class EvalResult(Base):
    """Stores pass/fail outcomes for each evaluation query run."""

    __tablename__ = "eval_results"

    id = __import__("sqlalchemy").Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    eval_query_id = __import__("sqlalchemy").Column(String(36), ForeignKey("eval_queries.id"), nullable=False)
    run_id = __import__("sqlalchemy").Column(String(36), ForeignKey("runs.id"), nullable=False)
    passed = __import__("sqlalchemy").Column(Boolean, nullable=False)
    created_at = __import__("sqlalchemy").Column(DateTime, default=datetime.utcnow, nullable=False)

    eval_query = relationship("EvalQuery", back_populates="eval_results")
    run = relationship("Run", back_populates="eval_results")


def init_db() -> None:
    """Create all configured tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Create and return a new SQLAlchemy session."""
    return SessionLocal()


def save_run(
    query: str,
    synthesised_program: Optional[str],
    result: Any,
    success: bool,
    failure_reason: Optional[str],
    iterations: int,
    latency_ms: Optional[float],
    scene_id: str,
) -> Run:
    """Persist one run and return the saved row."""
    session = get_session()
    try:
        row = Run(
            query=query,
            synthesised_program=synthesised_program,
            result_json=json.dumps(result, default=str) if result is not None else None,
            success=success,
            failure_reason=failure_reason,
            iterations=iterations,
            latency_ms=latency_ms,
            scene_id=scene_id,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row
    finally:
        session.close()


def get_run(run_id: str) -> Optional[Run]:
    """Return one run row by ID."""
    session = get_session()
    try:
        return session.query(Run).filter(Run.id == run_id).first()
    finally:
        session.close()


def list_runs(limit: int = 50) -> List[Run]:
    """Return latest run rows ordered by descending creation timestamp."""
    session = get_session()
    try:
        return session.query(Run).order_by(Run.created_at.desc()).limit(limit).all()
    finally:
        session.close()


def save_eval_query(query: str, expected_result_json: str, category: str) -> EvalQuery:
    """Insert one evaluation query row."""
    session = get_session()
    try:
        row = EvalQuery(query=query, expected_result_json=expected_result_json, category=category)
        session.add(row)
        session.commit()
        session.refresh(row)
        return row
    finally:
        session.close()


def list_eval_queries() -> List[EvalQuery]:
    """Return all evaluation queries ordered by creation time."""
    session = get_session()
    try:
        return session.query(EvalQuery).order_by(EvalQuery.created_at.asc()).all()
    finally:
        session.close()


def save_eval_result(eval_query_id: str, run_id: str, passed: bool) -> EvalResult:
    """Insert one evaluation result row."""
    session = get_session()
    try:
        row = EvalResult(eval_query_id=eval_query_id, run_id=run_id, passed=passed)
        session.add(row)
        session.commit()
        session.refresh(row)
        return row
    finally:
        session.close()
