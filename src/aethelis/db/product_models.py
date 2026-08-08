from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from aethelis.db.models import Base, jsonb, utcnow

PRODUCT_TABLE_NAMES = (
    "product_principals",
    "product_player_profiles",
    "product_world_definitions",
    "product_world_content_versions",
    "product_world_content_packages",
    "product_world_instances",
    "product_world_access_grants",
    "product_world_snapshots",
    "product_play_sessions",
    "product_save_points",
)


class ProductPrincipalRecord(Base):
    __tablename__ = "product_principals"
    __table_args__ = (
        UniqueConstraint("identity_provider", "external_subject", name="uq_product_identity"),
        CheckConstraint(
            "status in ('active', 'suspended', 'disabled')", name="ck_principal_status"
        ),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    identity_provider: Mapped[str] = mapped_column(String(80), nullable=False)
    external_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    roles_json: Mapped[dict[str, Any]] = jsonb()
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProductPlayerProfileRecord(Base):
    __tablename__ = "product_player_profiles"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    principal_id: Mapped[str] = mapped_column(
        ForeignKey("product_principals.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    locale: Mapped[str] = mapped_column(String(35), nullable=False)
    accessibility_preferences: Mapped[dict[str, Any]] = jsonb()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProductWorldDefinitionRecord(Base):
    __tablename__ = "product_world_definitions"
    __table_args__ = (
        CheckConstraint("status in ('active', 'retired')", name="ck_world_definition_status"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProductWorldContentVersionRecord(Base):
    __tablename__ = "product_world_content_versions"
    __table_args__ = (
        UniqueConstraint("world_definition_id", "content_hash", name="uq_world_content_hash"),
        CheckConstraint("status in ('draft', 'published', 'retired')", name="ck_content_status"),
        CheckConstraint(
            "status <> 'published' or published_at is not null",
            name="ck_published_content_timestamp",
        ),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    world_definition_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_definitions.id", ondelete="RESTRICT"), nullable=False
    )
    schema_version: Mapped[str] = mapped_column(String(80), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ProductWorldContentPackageRecord(Base):
    __tablename__ = "product_world_content_packages"

    content_version_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_content_versions.id", ondelete="CASCADE"), primary_key=True
    )
    package_id: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    package_json: Mapped[dict[str, Any]] = jsonb()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProductWorldInstanceRecord(Base):
    __tablename__ = "product_world_instances"
    __table_args__ = (
        CheckConstraint("status in ('active', 'suspended', 'archived')", name="ck_instance_status"),
        CheckConstraint("current_world_version >= 0", name="ck_instance_world_version"),
        CheckConstraint(
            "(forked_from_world_instance_id is null and forked_from_save_point_id is null "
            "and forked_from_snapshot_id is null) or "
            "(forked_from_world_instance_id is not null and forked_from_save_point_id is not null "
            "and forked_from_snapshot_id is not null)",
            name="ck_world_instance_fork_provenance",
        ),
        Index("ix_product_world_instances_owner", "owner_principal_id", "updated_at"),
        Index("ix_product_world_instances_fork_parent", "forked_from_world_instance_id"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    owner_principal_id: Mapped[str] = mapped_column(
        ForeignKey("product_principals.id", ondelete="RESTRICT"), nullable=False
    )
    world_definition_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_definitions.id", ondelete="RESTRICT"), nullable=False
    )
    content_version_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_content_versions.id", ondelete="RESTRICT"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    forked_from_world_instance_id: Mapped[str | None] = mapped_column(String(120))
    forked_from_save_point_id: Mapped[str | None] = mapped_column(String(120))
    forked_from_snapshot_id: Mapped[str | None] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    current_world_version: Mapped[int] = mapped_column(Integer, nullable=False)
    current_snapshot_id: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProductWorldAccessGrantRecord(Base):
    __tablename__ = "product_world_access_grants"
    __table_args__ = (
        UniqueConstraint("principal_id", "world_instance_id", name="uq_world_access_principal"),
        CheckConstraint("access_level in ('view', 'play', 'manage')", name="ck_world_access_level"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    principal_id: Mapped[str] = mapped_column(
        ForeignKey("product_principals.id", ondelete="CASCADE"), nullable=False
    )
    world_instance_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_instances.id", ondelete="CASCADE"), nullable=False
    )
    access_level: Mapped[str] = mapped_column(String(40), nullable=False)
    granted_by_principal_id: Mapped[str] = mapped_column(
        ForeignKey("product_principals.id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProductWorldSnapshotRecord(Base):
    __tablename__ = "product_world_snapshots"
    __table_args__ = (
        UniqueConstraint("world_instance_id", "world_version", name="uq_world_snapshot_version"),
        CheckConstraint("world_version >= 0", name="ck_snapshot_world_version"),
        Index("ix_product_world_snapshots_instance", "world_instance_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    world_instance_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_instances.id", ondelete="CASCADE"), nullable=False
    )
    world_version: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_snapshot_id: Mapped[str | None] = mapped_column(String(120))
    source_command_id: Mapped[str | None] = mapped_column(String(120))
    source_committed_event_id: Mapped[str | None] = mapped_column(String(120))
    content_version_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_content_versions.id", ondelete="RESTRICT"), nullable=False
    )
    engine_schema_version: Mapped[str] = mapped_column(String(80), nullable=False)
    state_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    world_state_json: Mapped[dict[str, Any]] = jsonb()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProductPlaySessionRecord(Base):
    __tablename__ = "product_play_sessions"
    __table_args__ = (
        CheckConstraint("status in ('active', 'suspended', 'closed')", name="ck_session_status"),
        CheckConstraint(
            "entry_world_version >= 0 and last_observed_world_version >= 0",
            name="ck_session_world_versions",
        ),
        Index(
            "ix_product_sessions_resume",
            "world_instance_id",
            "player_profile_id",
            "status",
            "last_active_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    world_instance_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_instances.id", ondelete="CASCADE"), nullable=False
    )
    player_profile_id: Mapped[str] = mapped_column(
        ForeignKey("product_player_profiles.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    entry_world_version: Mapped[int] = mapped_column(Integer, nullable=False)
    last_observed_world_version: Mapped[int] = mapped_column(Integer, nullable=False)
    last_acknowledged_command_id: Mapped[str | None] = mapped_column(String(120))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ProductSavePointRecord(Base):
    __tablename__ = "product_save_points"
    __table_args__ = (
        CheckConstraint("world_version >= 0", name="ck_save_world_version"),
        CheckConstraint(
            "reason in ('instance_created', 'auto', 'manual', "
            "'session_suspended', 'session_closed')",
            name="ck_save_reason",
        ),
        Index("ix_product_save_points_latest", "world_instance_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    world_instance_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_instances.id", ondelete="CASCADE"), nullable=False
    )
    world_version: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    content_version_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_content_versions.id", ondelete="RESTRICT"), nullable=False
    )
    play_session_id: Mapped[str | None] = mapped_column(
        ForeignKey("product_play_sessions.id", ondelete="SET NULL")
    )
    command_id: Mapped[str | None] = mapped_column(String(120))
    name: Mapped[str | None] = mapped_column(String(120))
    reason: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
