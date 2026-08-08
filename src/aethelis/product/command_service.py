from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from aethelis.product.command_contracts import (
    CommandExecution,
    CommandReceipt,
    CommandResultView,
    PlayerCommand,
    SubmitPlayerCommand,
)
from aethelis.product.contracts import (
    PlaySessionStatus,
    PrincipalContext,
    PrincipalStatus,
    WorldAccessLevel,
    WorldInstanceStatus,
    utc_timestamp,
)
from aethelis.product.errors import (
    ProductAccessDeniedError,
    ProductConflictError,
    ProductNotFoundError,
)
from aethelis.product.ports import ProductUnitOfWork


class CommandRepository(Protocol):
    def consume_rate_limit(
        self,
        *,
        principal_id: str,
        now: datetime,
        limit: int,
        window_seconds: int,
    ) -> bool: ...

    def add(self, command: PlayerCommand, execution: CommandExecution) -> PlayerCommand: ...

    def get(self, command_id: str) -> PlayerCommand | None: ...

    def get_execution(self, command_id: str) -> CommandExecution | None: ...

    def get_result(self, command_id: str) -> CommandResultView | None: ...

    def get_by_idempotency(
        self, principal_id: str, world_instance_id: str, idempotency_key: str
    ) -> PlayerCommand | None: ...

    def request_cancellation(
        self, command_id: str, principal_id: str, now: datetime
    ) -> PlayerCommand | None: ...


class CommandApplicationService:
    def __init__(
        self,
        uow_factory: Callable[[], ProductUnitOfWork],
        command_repository: CommandRepository,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        id_factory: Callable[[], str] = lambda: f"player_command_{uuid4().hex}",
        max_attempts: int = 3,
        commands_per_minute: int = 60,
    ) -> None:
        self._uow_factory = uow_factory
        self._commands = command_repository
        self._clock = clock
        self._id_factory = id_factory
        self._max_attempts = max_attempts
        self._commands_per_minute = commands_per_minute

    def submit(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
        request: SubmitPlayerCommand,
    ) -> CommandReceipt:
        now = utc_timestamp(self._clock())
        existing = self._commands.get_by_idempotency(
            principal.principal_id, world_instance_id, request.idempotency_key
        )
        if existing is not None:
            self._ensure_same_request(existing, request)
            return self._receipt(existing)
        with self._uow_factory() as uow:
            stored_principal = uow.identities.get_principal(principal.principal_id)
            if stored_principal is None or stored_principal.status != PrincipalStatus.ACTIVE:
                raise ProductAccessDeniedError(
                    "principal_not_active", "Authenticated principal is not active."
                )
            if not set(principal.roles).issubset(stored_principal.roles):
                raise ProductAccessDeniedError(
                    "principal_role_escalation", "Authenticated role context is not permitted."
                )
            profile = uow.identities.get_profile(request.player_profile_id)
            if profile is None or profile.principal_id != principal.principal_id:
                raise ProductAccessDeniedError(
                    "player_profile_forbidden", "Player profile is not authorized."
                )
            if request.actor_id != request.player_profile_id:
                raise ProductAccessDeniedError(
                    "player_actor_forbidden", "Command actor must be the authorized player profile."
                )
            instance = uow.worlds.get_instance(world_instance_id)
            if instance is None:
                raise ProductNotFoundError(
                    "world_instance_not_found", "World instance was not found."
                )
            if instance.status == WorldInstanceStatus.ARCHIVED:
                raise ProductConflictError(
                    "world_instance_archived",
                    "Archived world instance cannot accept commands.",
                )
            grant = uow.access.get_grant(principal.principal_id, world_instance_id)
            if grant is None or grant.access_level not in {
                WorldAccessLevel.PLAY,
                WorldAccessLevel.MANAGE,
            }:
                raise ProductAccessDeniedError(
                    "world_access_forbidden", "Principal cannot play this world."
                )
            session = uow.sessions.get_session(request.play_session_id)
            if (
                session is None
                or session.world_instance_id != world_instance_id
                or session.player_profile_id != request.player_profile_id
                or session.status != PlaySessionStatus.ACTIVE
            ):
                raise ProductConflictError(
                    "play_session_not_active", "An active matching play session is required."
                )
            if request.expected_world_version != instance.current_world_version:
                raise ProductConflictError(
                    "stale_world_version", "Observed world version is no longer current."
                )

        if not self._commands.consume_rate_limit(
            principal_id=principal.principal_id,
            now=now,
            limit=self._commands_per_minute,
            window_seconds=60,
        ):
            raise ProductConflictError(
                "command_rate_limited", "Command submission rate limit was exceeded."
            )

        command = PlayerCommand(
            id=self._id_factory(),
            idempotency_key=request.idempotency_key,
            principal_id=principal.principal_id,
            player_profile_id=request.player_profile_id,
            world_instance_id=world_instance_id,
            play_session_id=request.play_session_id,
            input_mode=request.input_mode,
            action_id=request.action_id,
            text=request.text,
            actor_id=request.actor_id,
            target_ids=request.target_ids,
            target_hints=request.target_hints,
            dialogue_interaction_id=request.dialogue_interaction_id,
            location_id=request.location_id,
            client_scene_id=request.client_scene_id,
            expected_world_version=request.expected_world_version,
            locale=request.locale,
            submitted_at=now,
            updated_at=now,
        )
        execution = CommandExecution(
            command_id=command.id,
            attempt_count=0,
            max_attempts=self._max_attempts,
            created_at=now,
            updated_at=now,
        )
        persisted = self._commands.add(command, execution)
        self._ensure_same_request(persisted, request)
        return self._receipt(persisted)

    def get(
        self, *, principal: PrincipalContext, world_instance_id: str, command_id: str
    ) -> CommandReceipt:
        command = self._commands.get(command_id)
        if (
            command is None
            or command.world_instance_id != world_instance_id
            or command.principal_id != principal.principal_id
        ):
            raise ProductNotFoundError("player_command_not_found", "Command was not found.")
        return self._receipt(command)

    def cancel(
        self, *, principal: PrincipalContext, world_instance_id: str, command_id: str
    ) -> CommandReceipt:
        command = self._commands.get(command_id)
        if (
            command is None
            or command.world_instance_id != world_instance_id
            or command.principal_id != principal.principal_id
        ):
            raise ProductNotFoundError("player_command_not_found", "Command was not found.")
        cancelled = self._commands.request_cancellation(
            command_id, principal.principal_id, utc_timestamp(self._clock())
        )
        if cancelled is None:
            raise ProductNotFoundError("player_command_not_found", "Command was not found.")
        return self._receipt(cancelled)

    def _receipt(self, command: PlayerCommand) -> CommandReceipt:
        execution = self._commands.get_execution(command.id)
        if execution is None:
            raise ProductConflictError(
                "command_execution_missing", "Command execution state is missing."
            )
        return CommandReceipt(
            command=command,
            execution=execution,
            status_url=(
                f"/api/v1/world-instances/{command.world_instance_id}/commands/{command.id}"
            ),
            result=self._commands.get_result(command.id),
        )

    @staticmethod
    def _ensure_same_request(command: PlayerCommand, request: SubmitPlayerCommand) -> None:
        compared = (
            "player_profile_id",
            "play_session_id",
            "input_mode",
            "action_id",
            "text",
            "actor_id",
            "target_ids",
            "target_hints",
            "dialogue_interaction_id",
            "location_id",
            "client_scene_id",
            "expected_world_version",
            "locale",
        )
        if any(getattr(command, field) != getattr(request, field) for field in compared):
            raise ProductConflictError(
                "idempotency_key_reused", "Idempotency key was already used for another request."
            )
