from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

import aethelis.db.command_models  # noqa: F401
import aethelis.db.product_models  # noqa: F401
from aethelis.api.app import create_app
from aethelis.api.auth import PrincipalResolver
from aethelis.db.command_models import (
    COMMAND_TABLE_NAMES,
    ProductCommandGovernanceRecord,
    ProductCommandResultRecord,
)
from aethelis.db.command_repository import SQLAlchemyCommandRepository
from aethelis.db.connection import create_db_engine, load_database_settings
from aethelis.db.models import Base
from aethelis.db.product_models import PRODUCT_TABLE_NAMES, ProductWorldSnapshotRecord
from aethelis.db.product_repository import sqlalchemy_product_uow_factory
from aethelis.product.command_contracts import (
    CommandInputMode,
    PlayerCommandStatus,
    SubmitPlayerCommand,
)
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
from aethelis.product.governance_worker import GovernanceWorker
from aethelis.product.projections import ProjectionService
from aethelis.product.services import ProductApplicationService
from aethelis.schemas.world import (
    CanonFact,
    CanonVisibility,
    Location,
    PlayerContext,
    ResourceKind,
    WorldResource,
    WorldState,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("AETHELIS_RUN_DB_TESTS") != "1",
    reason="set AETHELIS_RUN_DB_TESTS=1 for isolated PostgreSQL integration tests",
)


def test_governed_action_atomic_snapshot_safe_projection_and_stale_rejection() -> None:
    base_engine = create_db_engine(load_database_settings())
    schema = f"aethelis_governed_test_{uuid4().hex}"
    quoted_schema = f'"{schema}"'
    with base_engine.begin() as connection:
        connection.execute(text(f"create schema {quoted_schema}"))
    engine = base_engine.execution_options(schema_translate_map={None: schema})
    tables = [Base.metadata.tables[name] for name in (*PRODUCT_TABLE_NAMES, *COMMAND_TABLE_NAMES)]
    try:
        Base.metadata.create_all(engine, tables=tables)
        now = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
        uow_factory = sqlalchemy_product_uow_factory(engine)
        context, instance_id, session_id = _seed(uow_factory, now)
        repository = SQLAlchemyCommandRepository(engine)
        ids = iter(
            (
                "command_loop_1",
                "command_loop_2",
                "command_loop_3",
                "command_loop_4",
                "command_loop_5",
            )
        )
        commands = CommandApplicationService(
            uow_factory,
            repository,
            clock=lambda: now,
            id_factory=lambda: next(ids),
        )
        projections = ProjectionService(uow_factory)

        initial_scene = projections.scene(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_loop",
        )
        assert initial_scene.visible_resources == ()
        assert initial_scene.public_facts == ("The archive plaza is open to visitors.",)
        assert tuple(item.action_id for item in initial_scene.contextual_actions) == (
            "investigate_area",
        )

        first = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_request("loop-request-0001", session_id, expected_version=0),
        )
        stale = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_request("loop-request-0002", session_id, expected_version=0),
        )
        parser = CommandWorker(
            repository,
            StructuredIntentParser(),
            worker_id="parser-loop",
            clock=lambda: now,
        )
        assert parser.run_once().id == first.command.id
        assert parser.run_once().id == stale.command.id

        governance = GovernanceWorker(
            repository,
            worker_id="governance-loop",
            clock=lambda: now,
        )
        committed = governance.run_once()
        stale_result = governance.run_once()
        assert committed is not None and committed.status == "completed"
        assert committed.source_world_version == 0
        assert committed.resulting_world_version == 1
        assert committed.consequences == ("Discovered resource: Brass Survey Token",)
        assert stale_result is not None and stale_result.status == "rejected"
        assert "world changed" in stale_result.message.lower()

        no_discovery = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_request("loop-request-0003", session_id, expected_version=1),
        )
        assert parser.run_once().id == no_discovery.command.id
        rejected = governance.run_once()
        assert rejected is not None and rejected.status == "rejected"
        assert rejected.resulting_world_version is None
        assert "nothing new" in rejected.message.lower()

        cancel_candidate = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_request("loop-request-0004", session_id, expected_version=1),
        )
        assert parser.run_once().id == cancel_candidate.command.id
        cancelled = commands.cancel(
            principal=context,
            world_instance_id=instance_id,
            command_id=cancel_candidate.command.id,
        )
        assert cancelled.command.status == "cancelled"
        assert cancelled.result is not None
        assert cancelled.result.status == "cancelled"
        assert cancelled.result.resulting_world_version is None
        assert governance.run_once() is None

        duplicate = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_request("loop-request-0001", session_id, expected_version=0),
        )
        assert duplicate.command.id == first.command.id
        assert duplicate.result is not None
        assert duplicate.result.resulting_world_version == 1
        terminal_cancel = commands.cancel(
            principal=context,
            world_instance_id=instance_id,
            command_id=first.command.id,
        )
        assert terminal_cancel.command.status == "completed"

        interrupted = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_request("loop-request-0005", session_id, expected_version=1),
        )
        claimed = repository.claim_next(
            worker_id="parser-interrupted",
            now=now,
            lease_duration=timedelta(seconds=90),
        )
        assert claimed is not None and claimed[0].id == interrupted.command.id
        cancellation_pending = commands.cancel(
            principal=context,
            world_instance_id=instance_id,
            command_id=interrupted.command.id,
        )
        assert cancellation_pending.command.status == "interpreting"
        assert cancellation_pending.command.cancellation_requested
        finished = repository.finish_attempt(
            command_id=interrupted.command.id,
            worker_id="parser-interrupted",
            now=now,
            status=PlayerCommandStatus.FAILED,
            error_code="provider_interrupted",
            error_message="The provider was interrupted.",
            retryable=True,
        )
        assert finished.status == "cancelled"
        interrupted_receipt = commands.get(
            principal=context,
            world_instance_id=instance_id,
            command_id=interrupted.command.id,
        )
        assert interrupted_receipt.result is not None
        assert interrupted_receipt.result.status == "cancelled"

        with engine.connect() as connection:
            assert (
                connection.scalar(select(func.count()).select_from(ProductWorldSnapshotRecord)) == 2
            )
            assert (
                connection.scalar(select(func.count()).select_from(ProductCommandGovernanceRecord))
                == 3
            )
            assert (
                connection.scalar(select(func.count()).select_from(ProductCommandResultRecord)) == 5
            )

        scene = projections.scene(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_loop",
        )
        resume = projections.resume_summary(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_loop",
        )
        assert scene.world_version == 1
        assert tuple(item.name for item in scene.visible_resources) == ("Brass Survey Token",)
        assert scene.contextual_actions == ()
        assert "The sealed wing contains a hidden mechanism." not in scene.public_facts
        assert resume.world_version == 1
        assert resume.last_save_reason == "auto"
        assert resume.visible_resource_count == 1

        api = TestClient(
            create_app(
                command_service=CommandApplicationService(
                    uow_factory, SQLAlchemyCommandRepository(engine), clock=lambda: now
                ),
                product_service=ProductApplicationService(uow_factory, clock=lambda: now),
                principal_resolver=PrincipalResolver(
                    _StaticVerifier(), uow_factory, provider_id="test_oidc"
                ),
                projection_service=ProjectionService(uow_factory),
            )
        )
        headers = {"Authorization": "Bearer signed-test-token"}
        scene_response = api.get(
            f"/api/v1/world-instances/{instance_id}/scene",
            params={"player_profile_id": "profile_loop"},
            headers=headers,
        )
        result_response = api.get(
            f"/api/v1/world-instances/{instance_id}/commands/{first.command.id}",
            headers=headers,
        )
        assert scene_response.status_code == 200
        assert scene_response.json()["visible_resources"][0]["name"] == "Brass Survey Token"
        assert result_response.status_code == 200
        assert result_response.json()["result"]["status"] == "completed"
    finally:
        engine.dispose()
        with base_engine.begin() as connection:
            connection.execute(text(f"drop schema if exists {quoted_schema} cascade"))
        base_engine.dispose()


def _seed(uow_factory, now):
    service = ProductApplicationService(uow_factory, clock=lambda: now)
    principal = ProductPrincipal(
        id="principal_loop",
        identity_provider="test_oidc",
        external_subject="subject_loop",
        roles=(PrincipalRole.PLAYER, PrincipalRole.CONTENT_AUTHOR),
        created_at=now,
    )
    profile = PlayerProfile(
        id="profile_loop",
        principal_id=principal.id,
        display_name="Loop Traveler",
        created_at=now,
        updated_at=now,
    )
    context = PrincipalContext(principal_id=principal.id, roles=principal.roles)
    service.provision_identity(principal=principal, profile=profile)
    service.register_world_content(
        principal=context,
        definition=WorldDefinition(id="world_loop", name="Loop World", created_at=now),
        content_version=WorldContentVersion(
            id="content_loop",
            world_definition_id="world_loop",
            schema_version="1.0",
            content_hash="e" * 64,
            status=ContentVersionStatus.PUBLISHED,
            created_at=now,
            published_at=now,
        ),
    )
    created = service.create_world_instance(
        principal=context,
        request=CreateWorldInstanceRequest(
            world_definition_id="world_loop",
            content_version_id="content_loop",
            player_profile_id=profile.id,
            initial_world_state=WorldState(
                schema_version="1.0",
                world_id="world_loop",
                name="Loop World",
                summary="A bounded governed product world.",
                locations=(
                    Location(
                        id="archive_plaza",
                        name="Archive Plaza",
                        summary="A public stone plaza.",
                    ),
                ),
                factions=(),
                entities=(),
                resources=(
                    WorldResource(
                        id="survey_token",
                        name="Brass Survey Token",
                        kind=ResourceKind.KEY_ITEM,
                        quantity=1,
                        location_id="archive_plaza",
                        summary="A numbered token hidden under a survey marker.",
                    ),
                ),
                canon_facts=(
                    CanonFact(
                        id="public_fact",
                        statement="The archive plaza is open to visitors.",
                        visibility=CanonVisibility.PUBLIC,
                    ),
                    CanonFact(
                        id="hidden_fact",
                        statement="The sealed wing contains a hidden mechanism.",
                        visibility=CanonVisibility.HIDDEN_CANON,
                    ),
                ),
                player=PlayerContext(
                    summary="A visitor investigating the archive.",
                    current_location_id="archive_plaza",
                ),
            ),
        ),
    )
    session = service.start_or_resume_session(
        principal=context,
        world_instance_id=created.world_instance.id,
        player_profile_id=profile.id,
    )
    return context, created.world_instance.id, session.id


def _request(key: str, session_id: str, *, expected_version: int) -> SubmitPlayerCommand:
    return SubmitPlayerCommand(
        idempotency_key=key,
        player_profile_id="profile_loop",
        play_session_id=session_id,
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id="investigate_area",
        actor_id="profile_loop",
        location_id="archive_plaza",
        expected_world_version=expected_version,
    )


class _StaticVerifier:
    def verify_subject(self, _token: str) -> str:
        return "subject_loop"
