from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from aethelis.db.models import Base, jsonb

COMMAND_TABLE_NAMES = (
    "product_command_rate_windows",
    "product_player_commands",
    "product_command_executions",
    "product_command_governance_records",
    "product_command_results",
)


class ProductCommandRateWindowRecord(Base):
    __tablename__ = "product_command_rate_windows"
    __table_args__ = (CheckConstraint("command_count > 0", name="ck_command_rate_positive"),)

    principal_id: Mapped[str] = mapped_column(
        ForeignKey("product_principals.id", ondelete="CASCADE"), primary_key=True
    )
    window_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    command_count: Mapped[int] = mapped_column(Integer, nullable=False)


class ProductPlayerCommandRecord(Base):
    __tablename__ = "product_player_commands"
    __table_args__ = (
        UniqueConstraint(
            "principal_id",
            "world_instance_id",
            "idempotency_key",
            name="uq_product_command_idempotency",
        ),
        CheckConstraint(
            "status in ('submitted', 'interpreting', 'needs_clarification', "
            "'ready_for_governance', 'verifying', 'committed', 'rejected', "
            "'projecting', 'completed', 'cancelled', 'failed')",
            name="ck_product_command_status",
        ),
        CheckConstraint("expected_world_version >= 0", name="ck_command_world_version"),
        Index("ix_product_commands_world_status", "world_instance_id", "status", "submitted_at"),
    )

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    principal_id: Mapped[str] = mapped_column(
        ForeignKey("product_principals.id", ondelete="RESTRICT"), nullable=False
    )
    player_profile_id: Mapped[str] = mapped_column(
        ForeignKey("product_player_profiles.id", ondelete="RESTRICT"), nullable=False
    )
    world_instance_id: Mapped[str] = mapped_column(
        ForeignKey("product_world_instances.id", ondelete="CASCADE"), nullable=False
    )
    play_session_id: Mapped[str] = mapped_column(
        ForeignKey("product_play_sessions.id", ondelete="RESTRICT"), nullable=False
    )
    input_mode: Mapped[str] = mapped_column(String(40), nullable=False)
    action_id: Mapped[str | None] = mapped_column(String(120))
    text: Mapped[str | None] = mapped_column(Text)
    actor_id: Mapped[str] = mapped_column(String(120), nullable=False)
    target_ids_json: Mapped[dict[str, Any]] = jsonb()
    location_id: Mapped[str | None] = mapped_column(String(120))
    client_scene_id: Mapped[str | None] = mapped_column(String(120))
    expected_world_version: Mapped[int] = mapped_column(Integer, nullable=False)
    locale: Mapped[str] = mapped_column(String(35), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    cancellation_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProductCommandExecutionRecord(Base):
    __tablename__ = "product_command_executions"
    __table_args__ = (
        CheckConstraint("attempt_count >= 0", name="ck_execution_attempt_count"),
        CheckConstraint("max_attempts between 1 and 10", name="ck_execution_max_attempts"),
        Index("ix_product_executions_lease", "lease_expires_at", "attempt_count"),
    )

    command_id: Mapped[str] = mapped_column(
        ForeignKey("product_player_commands.id", ondelete="CASCADE"), primary_key=True
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(160))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parsed_intent_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(120))
    error_message: Mapped[str | None] = mapped_column(String(500))
    retryable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProductCommandGovernanceRecord(Base):
    __tablename__ = "product_command_governance_records"

    command_id: Mapped[str] = mapped_column(
        ForeignKey("product_player_commands.id", ondelete="CASCADE"), primary_key=True
    )
    action_proposal_json: Mapped[dict[str, Any]] = jsonb()
    event_candidate_json: Mapped[dict[str, Any]] = jsonb()
    verification_result_json: Mapped[dict[str, Any]] = jsonb()
    committed_event_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    state_apply_report_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProductCommandResultRecord(Base):
    __tablename__ = "product_command_results"
    __table_args__ = (
        CheckConstraint("source_world_version >= 0", name="ck_result_source_version"),
        CheckConstraint(
            "resulting_world_version is null or resulting_world_version >= source_world_version",
            name="ck_result_version_order",
        ),
    )

    command_id: Mapped[str] = mapped_column(
        ForeignKey("product_player_commands.id", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    message: Mapped[str] = mapped_column(String(500), nullable=False)
    source_world_version: Mapped[int] = mapped_column(Integer, nullable=False)
    resulting_world_version: Mapped[int | None] = mapped_column(Integer)
    snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("product_world_snapshots.id", ondelete="RESTRICT")
    )
    consequences_json: Mapped[dict[str, Any]] = jsonb()
    available_actions_json: Mapped[dict[str, Any]] = jsonb()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
