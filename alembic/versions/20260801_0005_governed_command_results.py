"""governed command results and projections

Revision ID: 20260801_0005
Revises: 20260801_0004
Create Date: 2026-08-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260801_0005"
down_revision: str | None = "20260801_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.drop_constraint("ck_product_command_status", "product_player_commands", type_="check")
    op.create_check_constraint(
        "ck_product_command_status",
        "product_player_commands",
        "status in ('submitted', 'interpreting', 'needs_clarification', "
        "'ready_for_governance', 'verifying', 'committed', 'rejected', "
        "'projecting', 'completed', 'cancelled', 'failed')",
    )
    op.create_table(
        "product_command_governance_records",
        sa.Column(
            "command_id",
            sa.String(120),
            sa.ForeignKey("product_player_commands.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("action_proposal_json", _jsonb(), nullable=False),
        sa.Column("event_candidate_json", _jsonb(), nullable=False),
        sa.Column("verification_result_json", _jsonb(), nullable=False),
        sa.Column("committed_event_json", _jsonb()),
        sa.Column("state_apply_report_json", _jsonb()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "product_command_results",
        sa.Column(
            "command_id",
            sa.String(120),
            sa.ForeignKey("product_player_commands.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("message", sa.String(500), nullable=False),
        sa.Column("source_world_version", sa.Integer(), nullable=False),
        sa.Column("resulting_world_version", sa.Integer()),
        sa.Column(
            "snapshot_id",
            sa.String(120),
            sa.ForeignKey("product_world_snapshots.id", ondelete="RESTRICT"),
        ),
        sa.Column("consequences_json", _jsonb(), nullable=False),
        sa.Column("available_actions_json", _jsonb(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("source_world_version >= 0", name="ck_result_source_version"),
        sa.CheckConstraint(
            "resulting_world_version is null or resulting_world_version >= source_world_version",
            name="ck_result_version_order",
        ),
    )


def downgrade() -> None:
    op.drop_table("product_command_results")
    op.drop_table("product_command_governance_records")
    op.drop_constraint("ck_product_command_status", "product_player_commands", type_="check")
    op.create_check_constraint(
        "ck_product_command_status",
        "product_player_commands",
        "status in ('submitted', 'interpreting', 'needs_clarification', "
        "'ready_for_governance', 'cancelled', 'failed')",
    )
