"""durable player commands

Revision ID: 20260801_0004
Revises: 20260801_0003
Create Date: 2026-08-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260801_0004"
down_revision: str | None = "20260801_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "product_command_rate_windows",
        sa.Column(
            "principal_id",
            sa.String(120),
            sa.ForeignKey("product_principals.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("window_started_at", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("command_count", sa.Integer(), nullable=False),
        sa.CheckConstraint("command_count > 0", name="ck_command_rate_positive"),
    )
    op.create_table(
        "product_player_commands",
        sa.Column("id", sa.String(120), primary_key=True),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column(
            "principal_id",
            sa.String(120),
            sa.ForeignKey("product_principals.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "player_profile_id",
            sa.String(120),
            sa.ForeignKey("product_player_profiles.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "world_instance_id",
            sa.String(120),
            sa.ForeignKey("product_world_instances.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "play_session_id",
            sa.String(120),
            sa.ForeignKey("product_play_sessions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("input_mode", sa.String(40), nullable=False),
        sa.Column("action_id", sa.String(120)),
        sa.Column("text", sa.Text()),
        sa.Column("actor_id", sa.String(120), nullable=False),
        sa.Column("target_ids_json", _jsonb(), nullable=False),
        sa.Column("location_id", sa.String(120)),
        sa.Column("client_scene_id", sa.String(120)),
        sa.Column("expected_world_version", sa.Integer(), nullable=False),
        sa.Column("locale", sa.String(35), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("cancellation_requested", sa.Boolean(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "principal_id",
            "world_instance_id",
            "idempotency_key",
            name="uq_product_command_idempotency",
        ),
        sa.CheckConstraint(
            "status in ('submitted', 'interpreting', 'needs_clarification', "
            "'ready_for_governance', 'cancelled', 'failed')",
            name="ck_product_command_status",
        ),
        sa.CheckConstraint("expected_world_version >= 0", name="ck_command_world_version"),
    )
    op.create_index(
        "ix_product_commands_world_status",
        "product_player_commands",
        ["world_instance_id", "status", "submitted_at"],
    )
    op.create_table(
        "product_command_executions",
        sa.Column(
            "command_id",
            sa.String(120),
            sa.ForeignKey("product_player_commands.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("lease_owner", sa.String(160)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("parsed_intent_json", _jsonb()),
        sa.Column("error_code", sa.String(120)),
        sa.Column("error_message", sa.String(500)),
        sa.Column("retryable", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("attempt_count >= 0", name="ck_execution_attempt_count"),
        sa.CheckConstraint("max_attempts between 1 and 10", name="ck_execution_max_attempts"),
    )
    op.create_index(
        "ix_product_executions_lease",
        "product_command_executions",
        ["lease_expires_at", "attempt_count"],
    )


def downgrade() -> None:
    op.drop_table("product_command_executions")
    op.drop_table("product_player_commands")
    op.drop_table("product_command_rate_windows")
