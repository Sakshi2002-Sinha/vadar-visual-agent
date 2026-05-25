"""Initial schema for runs, eval queries, and eval results tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial VADAR database schema."""
    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("synthesised_program", sa.Text(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("failure_reason", sa.String(length=64), nullable=True),
        sa.Column("iterations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("scene_id", sa.String(length=128), nullable=False, server_default="default"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "eval_queries",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("expected_result_json", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "eval_results",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("eval_query_id", sa.String(length=36), sa.ForeignKey("eval_queries.id"), nullable=False),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    """Drop all initial tables in reverse dependency order."""
    op.drop_table("eval_results")
    op.drop_table("eval_queries")
    op.drop_table("runs")
