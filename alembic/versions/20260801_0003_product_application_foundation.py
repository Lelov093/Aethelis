"""product application foundation

Revision ID: 20260801_0003
Revises: 20260704_0002
Create Date: 2026-08-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260801_0003"
down_revision: str | None = "20260704_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "product_principals",
        sa.Column("id", sa.String(120), primary_key=True),
        sa.Column("identity_provider", sa.String(80), nullable=False),
        sa.Column("external_subject", sa.String(255), nullable=False),
        sa.Column("roles_json", _jsonb(), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("identity_provider", "external_subject", name="uq_product_identity"),
        sa.CheckConstraint(
            "status in ('active', 'suspended', 'disabled')", name="ck_principal_status"
        ),
    )
    op.create_table(
        "product_player_profiles",
        sa.Column("id", sa.String(120), primary_key=True),
        sa.Column(
            "principal_id",
            sa.String(120),
            sa.ForeignKey("product_principals.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("display_name", sa.String(80), nullable=False),
        sa.Column("locale", sa.String(35), nullable=False),
        sa.Column("accessibility_preferences", _jsonb(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "product_world_definitions",
        sa.Column("id", sa.String(120), primary_key=True),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status in ('active', 'retired')", name="ck_world_definition_status"),
    )
    op.create_table(
        "product_world_content_versions",
        sa.Column("id", sa.String(120), primary_key=True),
        sa.Column(
            "world_definition_id",
            sa.String(120),
            sa.ForeignKey("product_world_definitions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("schema_version", sa.String(80), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("world_definition_id", "content_hash", name="uq_world_content_hash"),
        sa.CheckConstraint("status in ('draft', 'published', 'retired')", name="ck_content_status"),
        sa.CheckConstraint(
            "status <> 'published' or published_at is not null",
            name="ck_published_content_timestamp",
        ),
    )
    op.create_table(
        "product_world_instances",
        sa.Column("id", sa.String(120), primary_key=True),
        sa.Column(
            "owner_principal_id",
            sa.String(120),
            sa.ForeignKey("product_principals.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "world_definition_id",
            sa.String(120),
            sa.ForeignKey("product_world_definitions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "content_version_id",
            sa.String(120),
            sa.ForeignKey("product_world_content_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("current_world_version", sa.Integer(), nullable=False),
        sa.Column("current_snapshot_id", sa.String(120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status in ('active', 'suspended', 'archived')", name="ck_instance_status"
        ),
        sa.CheckConstraint("current_world_version >= 0", name="ck_instance_world_version"),
    )
    op.create_index(
        "ix_product_world_instances_owner",
        "product_world_instances",
        ["owner_principal_id", "updated_at"],
    )
    op.create_table(
        "product_world_access_grants",
        sa.Column("id", sa.String(120), primary_key=True),
        sa.Column(
            "principal_id",
            sa.String(120),
            sa.ForeignKey("product_principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "world_instance_id",
            sa.String(120),
            sa.ForeignKey("product_world_instances.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("access_level", sa.String(40), nullable=False),
        sa.Column(
            "granted_by_principal_id",
            sa.String(120),
            sa.ForeignKey("product_principals.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("principal_id", "world_instance_id", name="uq_world_access_principal"),
        sa.CheckConstraint(
            "access_level in ('view', 'play', 'manage')", name="ck_world_access_level"
        ),
    )
    op.create_table(
        "product_world_snapshots",
        sa.Column("id", sa.String(120), primary_key=True),
        sa.Column(
            "world_instance_id",
            sa.String(120),
            sa.ForeignKey("product_world_instances.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("world_version", sa.Integer(), nullable=False),
        sa.Column("previous_snapshot_id", sa.String(120)),
        sa.Column("source_command_id", sa.String(120)),
        sa.Column("source_committed_event_id", sa.String(120)),
        sa.Column(
            "content_version_id",
            sa.String(120),
            sa.ForeignKey("product_world_content_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("engine_schema_version", sa.String(80), nullable=False),
        sa.Column("state_sha256", sa.String(64), nullable=False),
        sa.Column("world_state_json", _jsonb(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("world_instance_id", "world_version", name="uq_world_snapshot_version"),
        sa.CheckConstraint("world_version >= 0", name="ck_snapshot_world_version"),
    )
    op.create_index(
        "ix_product_world_snapshots_instance",
        "product_world_snapshots",
        ["world_instance_id", "created_at"],
    )
    op.create_table(
        "product_play_sessions",
        sa.Column("id", sa.String(120), primary_key=True),
        sa.Column(
            "world_instance_id",
            sa.String(120),
            sa.ForeignKey("product_world_instances.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "player_profile_id",
            sa.String(120),
            sa.ForeignKey("product_player_profiles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("entry_world_version", sa.Integer(), nullable=False),
        sa.Column("last_observed_world_version", sa.Integer(), nullable=False),
        sa.Column("last_acknowledged_command_id", sa.String(120)),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("suspended_at", sa.DateTime(timezone=True)),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status in ('active', 'suspended', 'closed')", name="ck_session_status"),
        sa.CheckConstraint(
            "entry_world_version >= 0 and last_observed_world_version >= 0",
            name="ck_session_world_versions",
        ),
    )
    op.create_index(
        "ix_product_sessions_resume",
        "product_play_sessions",
        ["world_instance_id", "player_profile_id", "status", "last_active_at"],
    )
    op.create_table(
        "product_save_points",
        sa.Column("id", sa.String(120), primary_key=True),
        sa.Column(
            "world_instance_id",
            sa.String(120),
            sa.ForeignKey("product_world_instances.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("world_version", sa.Integer(), nullable=False),
        sa.Column(
            "snapshot_id",
            sa.String(120),
            sa.ForeignKey("product_world_snapshots.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "content_version_id",
            sa.String(120),
            sa.ForeignKey("product_world_content_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "play_session_id",
            sa.String(120),
            sa.ForeignKey("product_play_sessions.id", ondelete="SET NULL"),
        ),
        sa.Column("command_id", sa.String(120)),
        sa.Column("reason", sa.String(40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("world_version >= 0", name="ck_save_world_version"),
        sa.CheckConstraint(
            "reason in ('instance_created', 'auto', 'manual', "
            "'session_suspended', 'session_closed')",
            name="ck_save_reason",
        ),
    )
    op.create_index(
        "ix_product_save_points_latest", "product_save_points", ["world_instance_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_table("product_save_points")
    op.drop_table("product_play_sessions")
    op.drop_table("product_world_snapshots")
    op.drop_table("product_world_access_grants")
    op.drop_table("product_world_instances")
    op.drop_table("product_world_content_versions")
    op.drop_table("product_world_definitions")
    op.drop_table("product_player_profiles")
    op.drop_table("product_principals")
