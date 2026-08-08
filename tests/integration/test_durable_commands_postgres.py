from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

import aethelis.db.command_models  # noqa: F401
import aethelis.db.product_models  # noqa: F401
from aethelis.api.app import create_app
from aethelis.api.auth import PrincipalResolver
from aethelis.db.command_models import COMMAND_TABLE_NAMES
from aethelis.db.command_repository import SQLAlchemyCommandRepository
from aethelis.db.connection import create_db_engine, load_database_settings
from aethelis.db.models import Base
from aethelis.db.product_models import PRODUCT_TABLE_NAMES
from aethelis.db.product_repository import sqlalchemy_product_uow_factory
from aethelis.product.command_contracts import CommandInputMode, SubmitPlayerCommand
from aethelis.product.command_service import CommandApplicationService
from aethelis.product.command_worker import CommandWorker, StructuredIntentParser
from aethelis.product.contracts import (
    ContentVersionStatus,
    CreateWorldInstanceRequest,
    PlayerProfile,
    PrincipalContext,
    PrincipalRole,
    ProductPrincipal,
    WorldContentVersion,
    WorldDefinition,
)
from aethelis.product.errors import ProductConflictError
from aethelis.product.projections import ProjectionService
from aethelis.product.services import ProductApplicationService
from aethelis.schemas.world import PlayerContext, WorldState

pytestmark = pytest.mark.skipif(
    os.environ.get("AETHELIS_RUN_DB_TESTS") != "1",
    reason="set AETHELIS_RUN_DB_TESTS=1 for isolated PostgreSQL integration tests",
)


def test_idempotency_leases_recovery_cancellation_and_restart_readback() -> None:
    base_engine = create_db_engine(load_database_settings())
    schema = f"aethelis_command_test_{uuid4().hex}"
    quoted_schema = f'"{schema}"'
    with base_engine.begin() as connection:
        connection.execute(text(f"create schema {quoted_schema}"))
    engine = base_engine.execution_options(schema_translate_map={None: schema})
    tables = [Base.metadata.tables[name] for name in (*PRODUCT_TABLE_NAMES, *COMMAND_TABLE_NAMES)]
    try:
        Base.metadata.create_all(engine, tables=tables)
        now = datetime(2026, 8, 1, 10, 0, tzinfo=UTC)
        uow_factory = sqlalchemy_product_uow_factory(engine)
        principal, context, instance_id, session_id = _seed_product(uow_factory, now)
        repository = SQLAlchemyCommandRepository(engine)
        commands = CommandApplicationService(
            uow_factory,
            repository,
            clock=lambda: now,
            id_factory=lambda: f"command_{uuid4().hex}",
        )
        request = _request("request-0001", session_id)

        first = commands.submit(principal=context, world_instance_id=instance_id, request=request)
        duplicate = commands.submit(
            principal=context, world_instance_id=instance_id, request=request
        )
        assert duplicate.command.id == first.command.id

        worker = CommandWorker(
            repository,
            StructuredIntentParser(),
            worker_id="worker-a",
            clock=lambda: now,
        )
        completed = worker.run_once()
        assert completed is not None
        assert completed.status == "ready_for_governance"
        assert repository.get_execution(completed.id).parsed_intent.normalized_action == "inspect"

        second = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_request("request-0002", session_id),
        )
        claimed = repository.claim_next(
            worker_id="worker-a", now=now, lease_duration=timedelta(seconds=30)
        )
        assert claimed is not None and claimed[0].id == second.command.id
        assert (
            repository.claim_next(
                worker_id="worker-b", now=now, lease_duration=timedelta(seconds=30)
            )
            is None
        )
        reclaimed = repository.claim_next(
            worker_id="worker-b",
            now=now + timedelta(seconds=31),
            lease_duration=timedelta(seconds=30),
        )
        assert reclaimed is not None and reclaimed[1].attempt_count == 2

        third = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_request("request-0003", session_id),
        )
        cancelled = commands.cancel(
            principal=context,
            world_instance_id=instance_id,
            command_id=third.command.id,
        )
        assert cancelled.command.status == "cancelled"

        restarted = CommandApplicationService(
            uow_factory, SQLAlchemyCommandRepository(engine), clock=lambda: now
        )
        readback = restarted.get(
            principal=PrincipalContext(principal_id=principal.id, roles=(PrincipalRole.PLAYER,)),
            world_instance_id=instance_id,
            command_id=first.command.id,
        )
        assert readback.command.status == "ready_for_governance"
        assert readback.execution.lease_owner is None

        with pytest.raises(ProductConflictError) as reused:
            restarted.submit(
                principal=context,
                world_instance_id=instance_id,
                request=_request("request-0001", session_id, action_id="different"),
            )
        assert reused.value.code == "idempotency_key_reused"

        api = TestClient(
            create_app(
                command_service=restarted,
                product_service=ProductApplicationService(uow_factory, clock=lambda: now),
                principal_resolver=PrincipalResolver(
                    _StaticVerifier("subject_command"),
                    uow_factory,
                    provider_id="test_oidc",
                ),
                projection_service=ProjectionService(uow_factory),
            )
        )
        response = api.post(
            f"/api/v1/world-instances/{instance_id}/commands",
            headers={
                "Authorization": "Bearer signed-test-token",
                "Idempotency-Key": "request-0004",
            },
            json={
                "player_profile_id": "profile_command",
                "play_session_id": session_id,
                "input_mode": "contextual_action",
                "action_id": "inspect",
                "actor_id": "profile_command",
                "expected_world_version": 0,
            },
        )
        assert response.status_code == 202
        status_response = api.get(
            response.json()["status_url"],
            headers={"Authorization": "Bearer signed-test-token"},
        )
        assert status_response.status_code == 200
        assert status_response.json()["command"]["status"] == "submitted"

        limited = CommandApplicationService(
            uow_factory,
            SQLAlchemyCommandRepository(engine),
            clock=lambda: now,
            commands_per_minute=4,
        )
        with pytest.raises(ProductConflictError) as rate_limited:
            limited.submit(
                principal=context,
                world_instance_id=instance_id,
                request=_request("request-0005", session_id),
            )
        assert rate_limited.value.code == "command_rate_limited"
    finally:
        engine.dispose()
        with base_engine.begin() as connection:
            connection.execute(text(f"drop schema if exists {quoted_schema} cascade"))
        base_engine.dispose()


def _seed_product(uow_factory, now):
    service = ProductApplicationService(uow_factory, clock=lambda: now)
    principal = ProductPrincipal(
        id="principal_command",
        identity_provider="test_oidc",
        external_subject="subject_command",
        roles=(PrincipalRole.PLAYER, PrincipalRole.CONTENT_AUTHOR),
        created_at=now,
    )
    profile = PlayerProfile(
        id="profile_command",
        principal_id=principal.id,
        display_name="Command Traveler",
        created_at=now,
        updated_at=now,
    )
    context = PrincipalContext(principal_id=principal.id, roles=principal.roles)
    service.provision_identity(principal=principal, profile=profile)
    service.register_world_content(
        principal=context,
        definition=WorldDefinition(id="world_command", name="Command World", created_at=now),
        content_version=WorldContentVersion(
            id="content_command",
            world_definition_id="world_command",
            schema_version="1.0",
            content_hash="c" * 64,
            status=ContentVersionStatus.PUBLISHED,
            created_at=now,
            published_at=now,
        ),
    )
    created = service.create_world_instance(
        principal=context,
        request=CreateWorldInstanceRequest(
            world_definition_id="world_command",
            content_version_id="content_command",
            player_profile_id=profile.id,
            initial_world_state=WorldState(
                schema_version="1.0",
                world_id="world_command",
                name="Command World",
                summary="Durable command world.",
                locations=(),
                factions=(),
                entities=(),
                resources=(),
                canon_facts=(),
                player=PlayerContext(summary="Command player."),
            ),
        ),
    )
    session = service.start_or_resume_session(
        principal=context,
        world_instance_id=created.world_instance.id,
        player_profile_id=profile.id,
    )
    return principal, context, created.world_instance.id, session.id


def _request(key: str, session_id: str, *, action_id: str = "inspect"):
    return SubmitPlayerCommand(
        idempotency_key=key,
        player_profile_id="profile_command",
        play_session_id=session_id,
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id=action_id,
        actor_id="profile_command",
        expected_world_version=0,
    )


class _StaticVerifier:
    def __init__(self, subject: str) -> None:
        self._subject = subject

    def verify_subject(self, _token: str) -> str:
        return self._subject
