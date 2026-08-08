from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

import aethelis.db.command_models as records
import aethelis.db.product_models as product_records
from aethelis.product.command_contracts import (
    CommandExecution,
    CommandInputMode,
    CommandResultView,
    GovernanceWorkItem,
    ParsedPlayerIntent,
    PlayerCommand,
    PlayerCommandStatus,
)
from aethelis.product.content_contracts import ProductContentPackage
from aethelis.product.world_engine import GovernedWorldOutcome
from aethelis.schemas.world import WorldState


class SQLAlchemyCommandRepository:
    def __init__(self, engine: Engine) -> None:
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    def add(self, command: PlayerCommand, execution: CommandExecution) -> PlayerCommand:
        try:
            with self._sessions.begin() as session:
                session.add(_command_record(command))
                session.flush()
                session.add(_execution_record(execution))
            return command
        except IntegrityError:
            existing = self.get_by_idempotency(
                command.principal_id,
                command.world_instance_id,
                command.idempotency_key,
            )
            if existing is None:
                raise
            return existing

    def consume_rate_limit(
        self,
        *,
        principal_id: str,
        now: datetime,
        limit: int,
        window_seconds: int,
    ) -> bool:
        epoch = int(now.timestamp())
        window_epoch = epoch - (epoch % window_seconds)
        window_started_at = datetime.fromtimestamp(window_epoch, tz=now.tzinfo)
        table = records.ProductCommandRateWindowRecord
        statement = (
            insert(table)
            .values(
                principal_id=principal_id,
                window_started_at=window_started_at,
                command_count=1,
            )
            .on_conflict_do_update(
                index_elements=[table.principal_id, table.window_started_at],
                set_={"command_count": table.command_count + 1},
                where=table.command_count < limit,
            )
            .returning(table.command_count)
        )
        with self._sessions.begin() as session:
            return session.scalar(statement) is not None

    def get(self, command_id: str) -> PlayerCommand | None:
        with self._sessions() as session:
            row = session.get(records.ProductPlayerCommandRecord, command_id)
            return _command(row) if row else None

    def get_execution(self, command_id: str) -> CommandExecution | None:
        with self._sessions() as session:
            row = session.get(records.ProductCommandExecutionRecord, command_id)
            return _execution(row) if row else None

    def get_by_idempotency(
        self, principal_id: str, world_instance_id: str, idempotency_key: str
    ) -> PlayerCommand | None:
        with self._sessions() as session:
            row = session.scalar(
                select(records.ProductPlayerCommandRecord).where(
                    records.ProductPlayerCommandRecord.principal_id == principal_id,
                    records.ProductPlayerCommandRecord.world_instance_id == world_instance_id,
                    records.ProductPlayerCommandRecord.idempotency_key == idempotency_key,
                )
            )
            return _command(row) if row else None

    def request_cancellation(
        self, command_id: str, principal_id: str, now: datetime
    ) -> PlayerCommand | None:
        with self._sessions.begin() as session:
            row = session.scalar(
                select(records.ProductPlayerCommandRecord)
                .where(
                    records.ProductPlayerCommandRecord.id == command_id,
                    records.ProductPlayerCommandRecord.principal_id == principal_id,
                )
                .with_for_update()
            )
            if row is None:
                return None
            if row.status in {
                "committed",
                "projecting",
                "completed",
                "rejected",
                "failed",
                "cancelled",
            }:
                return _command(row)
            row.cancellation_requested = True
            row.updated_at = now
            if row.status not in {"interpreting", "verifying"}:
                row.status = PlayerCommandStatus.CANCELLED.value
                self._add_cancelled_result(session, row, now)
            session.flush()
            return _command(row)

    def claim_next(
        self, *, worker_id: str, now: datetime, lease_duration: timedelta
    ) -> tuple[PlayerCommand, CommandExecution] | None:
        with self._sessions.begin() as session:
            statement = (
                select(records.ProductPlayerCommandRecord, records.ProductCommandExecutionRecord)
                .join(
                    records.ProductCommandExecutionRecord,
                    records.ProductCommandExecutionRecord.command_id
                    == records.ProductPlayerCommandRecord.id,
                )
                .where(
                    records.ProductPlayerCommandRecord.status.in_(("submitted", "interpreting")),
                    records.ProductPlayerCommandRecord.cancellation_requested.is_(False),
                    records.ProductCommandExecutionRecord.attempt_count
                    < records.ProductCommandExecutionRecord.max_attempts,
                    or_(
                        records.ProductCommandExecutionRecord.lease_expires_at.is_(None),
                        records.ProductCommandExecutionRecord.lease_expires_at <= now,
                    ),
                )
                .order_by(
                    records.ProductPlayerCommandRecord.submitted_at,
                    records.ProductPlayerCommandRecord.id,
                )
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            row = session.execute(statement).one_or_none()
            if row is None:
                return None
            command_row, execution_row = row
            command_row.status = PlayerCommandStatus.INTERPRETING.value
            command_row.updated_at = now
            execution_row.attempt_count += 1
            execution_row.lease_owner = worker_id
            execution_row.lease_expires_at = now + lease_duration
            execution_row.heartbeat_at = now
            execution_row.retryable = False
            execution_row.error_code = None
            execution_row.error_message = None
            session.flush()
            return _command(command_row), _execution(execution_row)

    def finish_attempt(
        self,
        *,
        command_id: str,
        worker_id: str,
        now: datetime,
        status: PlayerCommandStatus,
        parsed_intent: ParsedPlayerIntent | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        retryable: bool = False,
    ) -> PlayerCommand:
        with self._sessions.begin() as session:
            command_row, execution_row = self._locked_attempt(session, command_id, worker_id)
            if command_row.cancellation_requested:
                status = PlayerCommandStatus.CANCELLED
                retryable = False
            command_row.status = status.value
            command_row.updated_at = now
            execution_row.parsed_intent_json = (
                parsed_intent.model_dump(mode="json") if parsed_intent else None
            )
            execution_row.error_code = error_code
            execution_row.error_message = error_message
            execution_row.retryable = retryable
            execution_row.lease_owner = None
            execution_row.lease_expires_at = None
            execution_row.heartbeat_at = now
            execution_row.updated_at = now
            if command_row.cancellation_requested:
                self._add_cancelled_result(session, command_row, now)
            elif retryable and execution_row.attempt_count < execution_row.max_attempts:
                command_row.status = PlayerCommandStatus.SUBMITTED.value
            session.flush()
            return _command(command_row)

    def heartbeat(
        self,
        *,
        command_id: str,
        worker_id: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> None:
        with self._sessions.begin() as session:
            execution = session.scalar(
                select(records.ProductCommandExecutionRecord)
                .where(
                    records.ProductCommandExecutionRecord.command_id == command_id,
                    records.ProductCommandExecutionRecord.lease_owner == worker_id,
                    records.ProductCommandExecutionRecord.lease_expires_at > now,
                )
                .with_for_update()
            )
            if execution is None:
                raise LookupError("command attempt lease is no longer owned by this worker")
            execution.heartbeat_at = now
            execution.lease_expires_at = now + lease_duration
            execution.updated_at = now

    def claim_next_governance(
        self, *, worker_id: str, now: datetime, lease_duration: timedelta
    ) -> GovernanceWorkItem | None:
        with self._sessions.begin() as session:
            row = session.execute(
                select(
                    records.ProductPlayerCommandRecord,
                    records.ProductCommandExecutionRecord,
                    product_records.ProductWorldInstanceRecord,
                    product_records.ProductWorldSnapshotRecord,
                )
                .join(records.ProductCommandExecutionRecord)
                .join(
                    product_records.ProductWorldInstanceRecord,
                    product_records.ProductWorldInstanceRecord.id
                    == records.ProductPlayerCommandRecord.world_instance_id,
                )
                .join(
                    product_records.ProductWorldSnapshotRecord,
                    product_records.ProductWorldSnapshotRecord.id
                    == product_records.ProductWorldInstanceRecord.current_snapshot_id,
                )
                .where(
                    records.ProductPlayerCommandRecord.status.in_(
                        ("ready_for_governance", "verifying")
                    ),
                    records.ProductPlayerCommandRecord.cancellation_requested.is_(False),
                    records.ProductCommandExecutionRecord.parsed_intent_json.is_not(None),
                    records.ProductCommandExecutionRecord.attempt_count
                    < records.ProductCommandExecutionRecord.max_attempts,
                    or_(
                        records.ProductCommandExecutionRecord.lease_expires_at.is_(None),
                        records.ProductCommandExecutionRecord.lease_expires_at <= now,
                    ),
                )
                .order_by(
                    records.ProductPlayerCommandRecord.submitted_at,
                    records.ProductPlayerCommandRecord.id,
                )
                .with_for_update(skip_locked=True)
                .limit(1)
            ).one_or_none()
            if row is None:
                return None
            command_row, execution_row, instance_row, snapshot_row = row
            package_row = session.get(
                product_records.ProductWorldContentPackageRecord,
                instance_row.content_version_id,
            )
            command_row.status = PlayerCommandStatus.VERIFYING.value
            command_row.updated_at = now
            execution_row.attempt_count += 1
            execution_row.lease_owner = worker_id
            execution_row.lease_expires_at = now + lease_duration
            execution_row.heartbeat_at = now
            execution_row.updated_at = now
            session.flush()
            return GovernanceWorkItem(
                command=_command(command_row),
                execution=_execution(execution_row),
                world_state=WorldState.model_validate(snapshot_row.world_state_json),
                source_snapshot_id=snapshot_row.id,
                current_world_version=instance_row.current_world_version,
                content_version_id=instance_row.content_version_id,
                content_package=(
                    ProductContentPackage.model_validate(package_row.package_json)
                    if package_row is not None
                    else None
                ),
            )

    def commit_governed_outcome(
        self,
        *,
        item: GovernanceWorkItem,
        outcome: GovernedWorldOutcome,
        worker_id: str,
        now: datetime,
    ) -> CommandResultView:
        with self._sessions.begin() as session:
            command_row, execution_row = self._locked_attempt(session, item.command.id, worker_id)
            instance_row = session.get(
                product_records.ProductWorldInstanceRecord,
                command_row.world_instance_id,
                with_for_update=True,
            )
            if instance_row is None:
                raise LookupError("world instance disappeared during governance")

            cancelled = command_row.cancellation_requested
            archived = instance_row.status == "archived"
            stale = (
                instance_row.current_world_version != command_row.expected_world_version
                or instance_row.current_snapshot_id != item.source_snapshot_id
            )
            committed = outcome.committed and not cancelled and not archived and not stale
            snapshot_id: str | None = None
            resulting_version: int | None = None
            status = PlayerCommandStatus.REJECTED
            message = outcome.player_message
            consequences = outcome.consequences

            if cancelled:
                status = PlayerCommandStatus.CANCELLED
                message = "The command was cancelled before world commit."
                consequences = ()
            elif archived:
                status = PlayerCommandStatus.REJECTED
                message = "The timeline was archived before this action could commit."
                consequences = ()
            elif stale:
                status = PlayerCommandStatus.REJECTED
                message = "The world changed before this action could commit. Try again."
                consequences = ()
            elif committed:
                assert outcome.resulting_world_state is not None
                resulting_version = instance_row.current_world_version + 1
                snapshot_id = f"world_snapshot_{uuid4().hex}"
                snapshot = product_records.ProductWorldSnapshotRecord(
                    id=snapshot_id,
                    world_instance_id=instance_row.id,
                    world_version=resulting_version,
                    previous_snapshot_id=instance_row.current_snapshot_id,
                    source_command_id=command_row.id,
                    source_committed_event_id=outcome.committed_event.id,
                    content_version_id=instance_row.content_version_id,
                    engine_schema_version=outcome.resulting_world_state.schema_version,
                    state_sha256=_world_state_hash(outcome.resulting_world_state),
                    world_state_json=outcome.resulting_world_state.model_dump(mode="json"),
                    created_at=now,
                )
                session.add(snapshot)
                session.flush()
                instance_row.current_world_version = resulting_version
                instance_row.current_snapshot_id = snapshot_id
                instance_row.updated_at = now
                session.add(
                    product_records.ProductSavePointRecord(
                        id=f"save_point_{uuid4().hex}",
                        world_instance_id=instance_row.id,
                        world_version=resulting_version,
                        snapshot_id=snapshot_id,
                        content_version_id=instance_row.content_version_id,
                        play_session_id=command_row.play_session_id,
                        command_id=command_row.id,
                        reason="auto",
                        created_at=now,
                    )
                )
                status = PlayerCommandStatus.COMPLETED

            session.add(
                records.ProductCommandGovernanceRecord(
                    command_id=command_row.id,
                    action_proposal_json=outcome.proposal.model_dump(mode="json"),
                    event_candidate_json=outcome.candidate.model_dump(mode="json"),
                    verification_result_json=outcome.verification.model_dump(mode="json"),
                    committed_event_json=(
                        outcome.committed_event.model_dump(mode="json") if committed else None
                    ),
                    state_apply_report_json=(
                        outcome.apply_report.model_dump(mode="json")
                        if committed and outcome.apply_report
                        else None
                    ),
                    created_at=now,
                )
            )
            result = records.ProductCommandResultRecord(
                command_id=command_row.id,
                status=status.value,
                message=message,
                source_world_version=command_row.expected_world_version,
                resulting_world_version=resulting_version,
                snapshot_id=snapshot_id,
                consequences_json={"items": list(consequences)},
                available_actions_json={"items": ["return_to_scene"]},
                created_at=now,
            )
            session.add(result)
            command_row.status = status.value
            command_row.updated_at = now
            execution_row.lease_owner = None
            execution_row.lease_expires_at = None
            execution_row.heartbeat_at = now
            execution_row.updated_at = now
            session.flush()
            return _result(result)

    def get_result(self, command_id: str) -> CommandResultView | None:
        with self._sessions() as session:
            row = session.get(records.ProductCommandResultRecord, command_id)
            return _result(row) if row else None

    @staticmethod
    def _add_cancelled_result(
        session: Session,
        command_row: records.ProductPlayerCommandRecord,
        now: datetime,
    ) -> None:
        if session.get(records.ProductCommandResultRecord, command_row.id) is not None:
            return
        session.add(
            records.ProductCommandResultRecord(
                command_id=command_row.id,
                status=PlayerCommandStatus.CANCELLED.value,
                message="The command was cancelled before world commit.",
                source_world_version=command_row.expected_world_version,
                resulting_world_version=None,
                snapshot_id=None,
                consequences_json={"items": []},
                available_actions_json={"items": ["return_to_scene"]},
                created_at=now,
            )
        )

    @staticmethod
    def _locked_attempt(
        session: Session, command_id: str, worker_id: str
    ) -> tuple[records.ProductPlayerCommandRecord, records.ProductCommandExecutionRecord]:
        row = session.execute(
            select(records.ProductPlayerCommandRecord, records.ProductCommandExecutionRecord)
            .join(records.ProductCommandExecutionRecord)
            .where(
                records.ProductPlayerCommandRecord.id == command_id,
                records.ProductCommandExecutionRecord.lease_owner == worker_id,
            )
            .with_for_update()
        ).one_or_none()
        if row is None:
            raise LookupError("command attempt lease is missing or owned by another worker")
        return row


CommandRepositoryFactory = Callable[[], SQLAlchemyCommandRepository]


def _command_record(command: PlayerCommand) -> records.ProductPlayerCommandRecord:
    data = command.model_dump(
        mode="python",
        exclude={"target_ids", "target_hints", "dialogue_interaction_id"},
    )
    data["target_ids_json"] = {
        "ids": list(command.target_ids),
        "hints": command.target_hints,
        "dialogue_interaction_id": command.dialogue_interaction_id,
    }
    return records.ProductPlayerCommandRecord(**data)


def _execution_record(execution: CommandExecution) -> records.ProductCommandExecutionRecord:
    data = execution.model_dump(mode="python", exclude={"parsed_intent"})
    data["parsed_intent_json"] = (
        execution.parsed_intent.model_dump(mode="json") if execution.parsed_intent else None
    )
    return records.ProductCommandExecutionRecord(**data)


def _command(row: records.ProductPlayerCommandRecord) -> PlayerCommand:
    return PlayerCommand(
        id=row.id,
        idempotency_key=row.idempotency_key,
        principal_id=row.principal_id,
        player_profile_id=row.player_profile_id,
        world_instance_id=row.world_instance_id,
        play_session_id=row.play_session_id,
        input_mode=CommandInputMode(row.input_mode),
        action_id=row.action_id,
        text=row.text,
        actor_id=row.actor_id,
        target_ids=tuple(row.target_ids_json["ids"]),
        target_hints=dict(row.target_ids_json.get("hints", {})),
        dialogue_interaction_id=row.target_ids_json.get("dialogue_interaction_id"),
        location_id=row.location_id,
        client_scene_id=row.client_scene_id,
        expected_world_version=row.expected_world_version,
        locale=row.locale,
        status=PlayerCommandStatus(row.status),
        cancellation_requested=row.cancellation_requested,
        submitted_at=row.submitted_at,
        updated_at=row.updated_at,
    )


def _execution(row: records.ProductCommandExecutionRecord) -> CommandExecution:
    return CommandExecution(
        command_id=row.command_id,
        attempt_count=row.attempt_count,
        max_attempts=row.max_attempts,
        lease_owner=row.lease_owner,
        lease_expires_at=row.lease_expires_at,
        heartbeat_at=row.heartbeat_at,
        parsed_intent=(
            ParsedPlayerIntent.model_validate(row.parsed_intent_json)
            if row.parsed_intent_json
            else None
        ),
        error_code=row.error_code,
        error_message=row.error_message,
        retryable=row.retryable,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _result(row: records.ProductCommandResultRecord) -> CommandResultView:
    return CommandResultView(
        command_id=row.command_id,
        status=PlayerCommandStatus(row.status),
        message=row.message,
        source_world_version=row.source_world_version,
        resulting_world_version=row.resulting_world_version,
        snapshot_id=row.snapshot_id,
        consequences=tuple(row.consequences_json["items"]),
        available_actions=tuple(row.available_actions_json["items"]),
        created_at=row.created_at,
    )


def _world_state_hash(world_state: WorldState) -> str:
    encoded = json.dumps(
        world_state.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()
