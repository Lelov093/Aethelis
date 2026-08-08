"""pgvector embedding evidence tables

Revision ID: 20260704_0002
Revises: 20260704_0001
Create Date: 2026-07-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260704_0002"
down_revision: str | None = "20260704_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "embedding_records",
        sa.Column("embedding_id", sa.String(length=180), primary_key=True),
        sa.Column("run_id", sa.String(length=120), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column(
            "step_id", sa.String(length=120), sa.ForeignKey("run_steps.id", ondelete="CASCADE")
        ),
        sa.Column(
            "provider_call_id",
            sa.String(length=120),
            sa.ForeignKey("provider_call_records.id"),
        ),
        sa.Column("source_type", sa.String(length=120), nullable=False),
        sa.Column("source_object_id", sa.String(length=180), nullable=False),
        sa.Column("embedding_model", sa.String(length=240), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("vector_norm", sa.Float(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("redaction_status", sa.String(length=120), nullable=False),
        sa.Column("provider_called", sa.Boolean(), nullable=False),
        sa.Column("fallback_used", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_json", _jsonb(), nullable=False),
    )
    op.create_table(
        "embedding_chunks",
        sa.Column("chunk_id", sa.String(length=220), primary_key=True),
        sa.Column(
            "embedding_id",
            sa.String(length=180),
            sa.ForeignKey("embedding_records.embedding_id", ondelete="CASCADE"),
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(length=120), nullable=False),
        sa.Column("source_object_id", sa.String(length=180), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding_vector", sa.Text(), nullable=False),
        sa.Column("vector_norm", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_json", _jsonb(), nullable=False),
    )
    op.execute(
        "ALTER TABLE embedding_chunks "
        "ALTER COLUMN embedding_vector TYPE vector(1024) "
        "USING embedding_vector::vector(1024)"
    )


def downgrade() -> None:
    op.drop_table("embedding_chunks")
    op.drop_table("embedding_records")
