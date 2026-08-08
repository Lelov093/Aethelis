"""product content packages

Revision ID: 20260801_0006
Revises: 20260801_0005
Create Date: 2026-08-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260801_0006"
down_revision: str | None = "20260801_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "product_world_content_packages",
        sa.Column(
            "content_version_id",
            sa.String(120),
            sa.ForeignKey("product_world_content_versions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("package_id", sa.String(120), nullable=False, unique=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column(
            "package_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("product_world_content_packages")
