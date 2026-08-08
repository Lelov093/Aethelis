from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
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
)
from aethelis.db.command_repository import SQLAlchemyCommandRepository
from aethelis.db.connection import create_db_engine, load_database_settings
from aethelis.db.models import Base
from aethelis.db.product_models import (
    PRODUCT_TABLE_NAMES,
    ProductWorldContentPackageRecord,
    ProductWorldSnapshotRecord,
)
from aethelis.db.product_repository import sqlalchemy_product_uow_factory
from aethelis.product.command_contracts import CommandInputMode, SubmitPlayerCommand
from aethelis.product.command_service import CommandApplicationService
from aethelis.product.command_worker import CommandWorker, StructuredIntentParser
from aethelis.product.content_loader import ProductContentPackageLoader
from aethelis.product.contracts import (
    PlayerProfile,
    PrincipalContext,
    PrincipalRole,
    ProductPrincipal,
    SaveReason,
)
from aethelis.product.governance_worker import GovernanceWorker
from aethelis.product.projections import ProjectionService
from aethelis.product.services import ProductApplicationService

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.environ.get("AETHELIS_RUN_DB_TESTS") != "1",
    reason="set AETHELIS_RUN_DB_TESTS=1 for isolated PostgreSQL integration tests",
)


def test_mistgate_package_creation_movement_map_journal_and_restart() -> None:
    base_engine = create_db_engine(load_database_settings())
    schema = f"aethelis_mistgate_product_test_{uuid4().hex}"
    quoted_schema = f'"{schema}"'
    with base_engine.begin() as connection:
        connection.execute(text(f"create schema {quoted_schema}"))
    engine = base_engine.execution_options(schema_translate_map={None: schema})
    tables = [Base.metadata.tables[name] for name in (*PRODUCT_TABLE_NAMES, *COMMAND_TABLE_NAMES)]
    try:
        Base.metadata.create_all(engine, tables=tables)
        now = datetime(2026, 8, 1, 15, 0, tzinfo=UTC)
        uow_factory = sqlalchemy_product_uow_factory(engine)
        context = _provision(uow_factory, now)
        product = ProductApplicationService(uow_factory, clock=lambda: now)
        package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
        version = product.publish_content_package(principal=context, package=package)

        with uow_factory() as uow:
            stored = uow.catalog.get_content_package(version.id)
            assert stored is not None
            assert stored.content_hash == version.content_hash
            assert (
                stored.package.blueprint.source_seed_sha256 == package.blueprint.source_seed_sha256
            )

        created = product.create_world_instance_from_content(
            principal=context,
            content_version_id=version.id,
            player_profile_id="profile_mistgate",
        )
        instance_id = created.world_instance.id
        session = product.start_or_resume_session(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        projections = ProjectionService(uow_factory)
        scene = projections.scene(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        map_view = projections.map(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        journal = projections.journal(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )

        assert scene.location_name == "议会广场"
        assert tuple(entity.name for entity in scene.visible_entities) == ("罗文·凯斯特",)
        assert {action.target_id for action in scene.contextual_actions if action.target_id} == {
            "central_archive",
            "market_row",
            "old_aqueduct",
            "workshop_lane",
        }
        assert any(not action.command_required for action in scene.contextual_actions)
        assert len(map_view.locations) == 5
        assert sum(location.is_reachable for location in map_view.locations) == 4
        assert journal.entries[0].startswith("你以外来调查者")
        assert all("workshop safe" not in fact.lower() for fact in journal.confirmed_facts)
        assert journal.resources == ()
        assert tuple(item.title for item in journal.opportunities) == (
            "查阅旧维修记录",
            "追踪稀缺零件",
            "查明校准钥匙去向",
            "检查失稳信号",
        )
        assert all(not item.is_at_location for item in journal.opportunities)
        assert journal.situation.phase == "unstable"
        assert len(journal.current_objectives) == 3
        assert "私下" not in scene.model_dump_json()

        command_ids = iter(
            (
                "mistgate_move_1",
                "mistgate_move_2",
                "mistgate_move_3",
                "mistgate_move_4",
                "mistgate_dialogue_1",
                "mistgate_dialogue_2",
                "mistgate_exchange_1",
                "mistgate_move_5",
                "mistgate_move_6",
                "mistgate_repair_1",
                "mistgate_move_7",
                "mistgate_move_8",
                "mistgate_investigate_key",
                "mistgate_release_key",
                "mistgate_move_9",
                "mistgate_move_10",
                "mistgate_investigate_lens",
                "mistgate_validate_lens",
                "mistgate_final_repair",
                "mistgate_world_response",
                "mistgate_fork_breach",
                "mistgate_fork_move_1",
                "mistgate_fork_move_2",
                "mistgate_fork_repair",
                "mistgate_fork_move_3",
                "mistgate_fork_move_4",
                "mistgate_fork_investigate_key",
                "mistgate_fork_release_key",
                "mistgate_fork_move_5",
                "mistgate_fork_move_6",
                "mistgate_fork_investigate_lens",
                "mistgate_fork_validate_lens",
                "mistgate_fork_final_repair",
                "mistgate_fork_world_response",
            )
        )
        repository = SQLAlchemyCommandRepository(engine)
        commands = CommandApplicationService(
            uow_factory,
            repository,
            clock=lambda: now,
            id_factory=lambda: next(command_ids),
        )
        parser = CommandWorker(
            repository,
            StructuredIntentParser(),
            worker_id="mistgate-parser",
            clock=lambda: now,
        )
        governance = GovernanceWorker(
            repository,
            worker_id="mistgate-governance",
            clock=lambda: now,
        )
        move = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_move_request(
                "mistgate-move-request-1",
                session.id,
                expected_version=0,
                source="council_square",
                destination="central_archive",
            ),
        )
        assert parser.run_once().id == move.command.id
        moved = governance.run_once()
        assert moved is not None and moved.status == "completed"
        assert moved.resulting_world_version == 1

        archive_scene = projections.scene(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        assert archive_scene.location_name == "中央档案馆"
        assert {entity.name for entity in archive_scene.visible_entities} == {
            "米拉·维尔",
            "谐波调谐器",
        }

        blocked = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_move_request(
                "mistgate-move-request-2",
                session.id,
                expected_version=1,
                source="central_archive",
                destination="market_row",
            ),
        )
        assert parser.run_once().id == blocked.command.id
        rejected = governance.run_once()
        assert rejected is not None and rejected.status == "rejected"
        assert rejected.resulting_world_version is None

        restarted_product = ProductApplicationService(uow_factory, clock=lambda: now)
        restarted = restarted_product.load_resume_state(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        assert restarted.world_instance.current_world_version == 1
        assert restarted.snapshot.world_state.player.current_location_id == "central_archive"

        for key, expected_version, source, destination in (
            ("mistgate-move-request-3", 1, "central_archive", "council_square"),
            ("mistgate-move-request-4", 2, "council_square", "market_row"),
        ):
            submitted = commands.submit(
                principal=context,
                world_instance_id=instance_id,
                request=_move_request(
                    key,
                    session.id,
                    expected_version=expected_version,
                    source=source,
                    destination=destination,
                ),
            )
            assert parser.run_once().id == submitted.command.id
            result = governance.run_once()
            assert result is not None and result.status == "completed"

        for key, expected_version, character_id in (
            ("mistgate-dialogue-request-1", 3, "selka"),
            ("mistgate-dialogue-request-2", 4, "nara"),
        ):
            submitted = commands.submit(
                principal=context,
                world_instance_id=instance_id,
                request=_dialogue_request(
                    key,
                    session.id,
                    expected_version=expected_version,
                    character_id=character_id,
                ),
            )
            assert parser.run_once().id == submitted.command.id
            result = governance.run_once()
            assert result is not None and result.status == "completed"

        social_journal = projections.journal(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        assert tuple(item.kind for item in social_journal.knowledge) == (
            "confirmed_fact",
            "rumor",
        )
        assert {item.character_id for item in social_journal.relationships} == {"selka", "nara"}
        assert all(item.standing_label == "初步信任" for item in social_journal.relationships)
        exchange_ready_scene = projections.scene(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        assert any(
            action.action_id == "negotiate_resource" and action.target_id == "selka"
            for action in exchange_ready_scene.contextual_actions
        )

        exchange = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_exchange_request(
                "mistgate-exchange-request-1",
                session.id,
                expected_version=5,
            ),
        )
        assert parser.run_once().id == exchange.command.id
        exchanged = governance.run_once()
        assert exchanged is not None and exchanged.status == "completed"
        assert exchanged.resulting_world_version == 6

        exchange_journal = projections.journal(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        held_parts = next(item for item in exchange_journal.resources if item.is_player_owned)
        assert held_parts.source_resource_id == "stabilizer_parts"
        assert held_parts.quantity == 1
        assert held_parts.custody_label == "由你持有"
        assert len(exchange_journal.commitments) == 1
        assert exchange_journal.commitments[0].status == "active"
        exchanged_scene = projections.scene(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        assert all(
            action.action_id != "negotiate_resource"
            for action in exchanged_scene.contextual_actions
        )

        social_save = product.create_save_point(
            principal=context,
            world_instance_id=instance_id,
            reason=SaveReason.MANUAL,
            play_session_id=session.id,
            name="集市交谈之后",
        )
        forked = product.fork_world_from_save(
            principal=context,
            player_profile_id="profile_mistgate",
            source_world_instance_id=instance_id,
            save_point_id=social_save.id,
            name="集市分支",
        )
        assert forked.world_instance.current_world_version == 0
        assert forked.world_instance.forked_from_save_point_id == social_save.id
        assert len(forked.snapshot.world_state.player.knowledge) == 2
        assert len(forked.snapshot.world_state.player.relationships) == 2
        assert len(forked.snapshot.world_state.player.dialogue_history) == 3
        assert len(forked.snapshot.world_state.player.inventory) == 1
        assert len(forked.snapshot.world_state.player.commitments) == 1
        stabilizer_stock = next(
            resource
            for resource in forked.snapshot.world_state.resources
            if resource.id == "stabilizer_parts"
        )
        assert stabilizer_stock.quantity == 2

        for key, expected_version, source, destination in (
            ("mistgate-move-request-5", 6, "market_row", "council_square"),
            ("mistgate-move-request-6", 7, "council_square", "old_aqueduct"),
        ):
            submitted = commands.submit(
                principal=context,
                world_instance_id=instance_id,
                request=_move_request(
                    key,
                    session.id,
                    expected_version=expected_version,
                    source=source,
                    destination=destination,
                ),
            )
            assert parser.run_once().id == submitted.command.id
            result = governance.run_once()
            assert result is not None and result.status == "completed"

        repair_ready_scene = projections.scene(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        assert any(
            action.action_id == "repair_regulator" and action.target_id == "dawn_regulator"
            for action in repair_ready_scene.contextual_actions
        )
        repair = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_repair_request(
                "mistgate-repair-request-1",
                session.id,
                expected_version=8,
            ),
        )
        assert parser.run_once().id == repair.command.id
        repaired = governance.run_once()
        assert repaired is not None and repaired.status == "completed"
        assert repaired.resulting_world_version == 9

        repaired_journal = projections.journal(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        assert repaired_journal.commitments[0].status == "fulfilled"
        assert all(not item.is_player_owned for item in repaired_journal.resources)
        assert tuple(item.id for item in repaired_journal.outcomes) == ("outcome_city_holds",)

        for key, expected_version, source, destination in (
            ("mistgate-move-request-7", 9, "old_aqueduct", "council_square"),
            ("mistgate-move-request-8", 10, "council_square", "workshop_lane"),
        ):
            submitted = commands.submit(
                principal=context,
                world_instance_id=instance_id,
                request=_move_request(
                    key,
                    session.id,
                    expected_version=expected_version,
                    source=source,
                    destination=destination,
                ),
            )
            assert parser.run_once().id == submitted.command.id
            result = governance.run_once()
            assert result is not None and result.status == "completed"

        for request in (
            _action_request(
                "mistgate-investigate-key-request",
                session.id,
                expected_version=11,
                action_id="investigate_area",
                location_id="workshop_lane",
            ),
            _action_request(
                "mistgate-release-key-request",
                session.id,
                expected_version=12,
                action_id="request_calibration_key",
                location_id="workshop_lane",
                target_id="ivo",
            ),
        ):
            submitted = commands.submit(
                principal=context,
                world_instance_id=instance_id,
                request=request,
            )
            assert parser.run_once().id == submitted.command.id
            result = governance.run_once()
            assert result is not None and result.status == "completed"

        for key, expected_version, source, destination in (
            ("mistgate-move-request-9", 13, "workshop_lane", "council_square"),
            ("mistgate-move-request-10", 14, "council_square", "old_aqueduct"),
        ):
            submitted = commands.submit(
                principal=context,
                world_instance_id=instance_id,
                request=_move_request(
                    key,
                    session.id,
                    expected_version=expected_version,
                    source=source,
                    destination=destination,
                ),
            )
            assert parser.run_once().id == submitted.command.id
            result = governance.run_once()
            assert result is not None and result.status == "completed"

        for request in (
            _action_request(
                "mistgate-investigate-lens-request",
                session.id,
                expected_version=15,
                action_id="investigate_area",
                location_id="old_aqueduct",
            ),
            _action_request(
                "mistgate-validate-lens-request",
                session.id,
                expected_version=16,
                action_id="validate_gate_lens",
                location_id="old_aqueduct",
                target_id="gate_lens",
            ),
            _action_request(
                "mistgate-final-repair-request",
                session.id,
                expected_version=17,
                action_id="stabilize_regulator",
                location_id="old_aqueduct",
                target_id="dawn_regulator",
            ),
        ):
            submitted = commands.submit(
                principal=context,
                world_instance_id=instance_id,
                request=request,
            )
            assert parser.run_once().id == submitted.command.id
            result = governance.run_once()
            assert result is not None and result.status == "completed"

        ending_journal = projections.journal(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        assert ending_journal.commitments[0].status == "fulfilled"
        assert all(
            not item.is_player_owned or item.source_resource_id != "calibration_key"
            for item in ending_journal.resources
        )
        assert tuple(item.id for item in ending_journal.outcomes) == (
            "outcome_regulator_stabilized",
        )

        response_command = commands.submit(
            principal=context,
            world_instance_id=instance_id,
            request=_action_request(
                "mistgate-world-response-request",
                session.id,
                expected_version=18,
                action_id="wait_for_world_response",
                location_id="old_aqueduct",
            ),
        )
        assert parser.run_once().id == response_command.command.id
        response_result = governance.run_once()
        assert response_result is not None and response_result.status == "completed"
        assert response_result.resulting_world_version == 19

        response_journal = projections.journal(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        assert response_journal.world_responses[0].response_kind == "civic_support"
        assert response_journal.world_responses[0].actor_name == "塞尔卡·奥林"

        fork_session = product.start_or_resume_session(
            principal=context,
            world_instance_id=forked.world_instance.id,
            player_profile_id="profile_mistgate",
        )
        fork_requests = (
            _action_request(
                "mistgate-fork-breach-request",
                fork_session.id,
                expected_version=0,
                action_id="break_commitment",
                location_id="market_row",
                target_id="selka",
            ),
            _move_request(
                "mistgate-fork-move-request-1",
                fork_session.id,
                expected_version=1,
                source="market_row",
                destination="council_square",
            ),
            _move_request(
                "mistgate-fork-move-request-2",
                fork_session.id,
                expected_version=2,
                source="council_square",
                destination="old_aqueduct",
            ),
            _repair_request(
                "mistgate-fork-repair-request",
                fork_session.id,
                expected_version=3,
            ),
            _move_request(
                "mistgate-fork-move-request-3",
                fork_session.id,
                expected_version=4,
                source="old_aqueduct",
                destination="council_square",
            ),
            _move_request(
                "mistgate-fork-move-request-4",
                fork_session.id,
                expected_version=5,
                source="council_square",
                destination="workshop_lane",
            ),
            _action_request(
                "mistgate-fork-investigate-key-request",
                fork_session.id,
                expected_version=6,
                action_id="investigate_area",
                location_id="workshop_lane",
            ),
            _action_request(
                "mistgate-fork-release-key-request",
                fork_session.id,
                expected_version=7,
                action_id="request_calibration_key",
                location_id="workshop_lane",
                target_id="ivo",
            ),
            _move_request(
                "mistgate-fork-move-request-5",
                fork_session.id,
                expected_version=8,
                source="workshop_lane",
                destination="council_square",
            ),
            _move_request(
                "mistgate-fork-move-request-6",
                fork_session.id,
                expected_version=9,
                source="council_square",
                destination="old_aqueduct",
            ),
            _action_request(
                "mistgate-fork-investigate-lens-request",
                fork_session.id,
                expected_version=10,
                action_id="investigate_area",
                location_id="old_aqueduct",
            ),
            _action_request(
                "mistgate-fork-validate-lens-request",
                fork_session.id,
                expected_version=11,
                action_id="validate_gate_lens",
                location_id="old_aqueduct",
                target_id="gate_lens",
            ),
            _action_request(
                "mistgate-fork-final-repair-request",
                fork_session.id,
                expected_version=12,
                action_id="stabilize_regulator",
                location_id="old_aqueduct",
                target_id="dawn_regulator",
            ),
            _action_request(
                "mistgate-fork-world-response-request",
                fork_session.id,
                expected_version=13,
                action_id="wait_for_world_response",
                location_id="old_aqueduct",
            ),
        )
        for request in fork_requests:
            submitted = commands.submit(
                principal=context,
                world_instance_id=forked.world_instance.id,
                request=request,
            )
            assert parser.run_once().id == submitted.command.id
            result = governance.run_once()
            assert result is not None and result.status == "completed"

        broken_journal = projections.journal(
            principal=context,
            world_instance_id=forked.world_instance.id,
            player_profile_id="profile_mistgate",
        )
        assert broken_journal.commitments[0].status == "broken"
        assert broken_journal.world_responses[0].response_kind == "social_withdrawal"
        unchanged_source = projections.journal(
            principal=context,
            world_instance_id=instance_id,
            player_profile_id="profile_mistgate",
        )
        assert unchanged_source.commitments[0].status == "fulfilled"
        assert unchanged_source.world_responses[0].response_kind == "civic_support"

        api = TestClient(
            create_app(
                command_service=CommandApplicationService(
                    uow_factory,
                    SQLAlchemyCommandRepository(engine),
                    clock=lambda: now,
                ),
                product_service=restarted_product,
                principal_resolver=PrincipalResolver(
                    _StaticVerifier(), uow_factory, provider_id="test_oidc"
                ),
                projection_service=ProjectionService(uow_factory),
            )
        )
        headers = {"Authorization": "Bearer signed-test-token"}
        catalog_response = api.get("/api/v1/world-definitions", headers=headers)
        map_response = api.get(
            f"/api/v1/world-instances/{instance_id}/map",
            params={"player_profile_id": "profile_mistgate"},
            headers=headers,
        )
        journal_response = api.get(
            f"/api/v1/world-instances/{instance_id}/journal",
            params={"player_profile_id": "profile_mistgate"},
            headers=headers,
        )
        create_response = api.post(
            "/api/v1/world-instances",
            json={
                "content_version_id": version.id,
                "player_profile_id": "profile_mistgate",
            },
            headers=headers,
        )
        assert catalog_response.status_code == 200
        assert catalog_response.json()[0]["world_name"] == "雾门档案城"
        assert map_response.status_code == 200
        assert map_response.json()["current_location_id"] == "old_aqueduct"
        assert journal_response.status_code == 200
        assert "canon_key_in_workshop_safe" not in journal_response.text
        assert journal_response.json()["outcomes"][0]["id"] == "outcome_regulator_stabilized"
        assert journal_response.json()["world_responses"][0]["response_kind"] == "civic_support"
        assert create_response.status_code == 201
        created_api_instance = create_response.json()["world_instance"]["id"]
        session_response = api.post(
            f"/api/v1/world-instances/{created_api_instance}/sessions",
            json={"player_profile_id": "profile_mistgate"},
            headers=headers,
        )
        assert session_response.status_code == 200

        with engine.connect() as connection:
            assert (
                connection.scalar(
                    select(func.count()).select_from(ProductWorldContentPackageRecord)
                )
                == 1
            )
            assert (
                connection.scalar(select(func.count()).select_from(ProductWorldSnapshotRecord))
                == 36
            )
            assert (
                connection.scalar(select(func.count()).select_from(ProductCommandGovernanceRecord))
                == 34
            )
    finally:
        engine.dispose()
        with base_engine.begin() as connection:
            connection.execute(text(f"drop schema if exists {quoted_schema} cascade"))
        base_engine.dispose()


def test_living_world_dialogue_claim_and_agent_turn_persist_in_postgres() -> None:
    base_engine = create_db_engine(load_database_settings())
    schema = f"aethelis_living_world_test_{uuid4().hex}"
    quoted_schema = f'"{schema}"'
    with base_engine.begin() as connection:
        connection.execute(text(f"create schema {quoted_schema}"))
    engine = base_engine.execution_options(schema_translate_map={None: schema})
    tables = [Base.metadata.tables[name] for name in (*PRODUCT_TABLE_NAMES, *COMMAND_TABLE_NAMES)]
    try:
        Base.metadata.create_all(engine, tables=tables)
        now = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
        uow_factory = sqlalchemy_product_uow_factory(engine)
        context = _provision(uow_factory, now)
        product = ProductApplicationService(uow_factory, clock=lambda: now)
        package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
        version = product.publish_content_package(principal=context, package=package)
        created = product.create_world_instance_from_content(
            principal=context,
            content_version_id=version.id,
            player_profile_id="profile_mistgate",
        )
        session = product.start_or_resume_session(
            principal=context,
            world_instance_id=created.world_instance.id,
            player_profile_id="profile_mistgate",
        )
        command_ids = iter(("living_hello", "living_claim", "living_advance"))
        repository = SQLAlchemyCommandRepository(engine)
        commands = CommandApplicationService(
            uow_factory,
            repository,
            clock=lambda: now,
            id_factory=lambda: next(command_ids),
        )
        parser = CommandWorker(
            repository,
            StructuredIntentParser(),
            worker_id="living-parser",
            clock=lambda: now,
        )
        governance = GovernanceWorker(
            repository,
            worker_id="living-governance",
            clock=lambda: now,
        )

        for key, expected_version, utterance in (
            ("living-hello-request", 0, "你好"),
            ("living-claim-request", 1, "我听说校准钥匙在市场。"),
        ):
            submitted = commands.submit(
                principal=context,
                world_instance_id=created.world_instance.id,
                request=SubmitPlayerCommand(
                    idempotency_key=key,
                    player_profile_id="profile_mistgate",
                    play_session_id=session.id,
                    input_mode=CommandInputMode.NATURAL_LANGUAGE_INTENT,
                    text=utterance,
                    actor_id="profile_mistgate",
                    target_ids=("rowan",),
                    target_hints={"rowan": "罗文·凯斯特"},
                    location_id="council_square",
                    expected_world_version=expected_version,
                    locale="zh-CN",
                ),
            )
            assert parser.run_once().id == submitted.command.id
            result = governance.run_once()
            assert result is not None and result.status == "completed"

        advance = commands.submit(
            principal=context,
            world_instance_id=created.world_instance.id,
            request=_action_request(
                "living-advance-request",
                session.id,
                expected_version=2,
                action_id="advance_world",
                location_id="council_square",
            ),
        )
        assert parser.run_once().id == advance.command.id
        advanced = governance.run_once()
        assert advanced is not None and advanced.status == "completed"
        assert advanced.resulting_world_version == 3

        resumed = product.load_resume_state(
            principal=context,
            world_instance_id=created.world_instance.id,
            player_profile_id="profile_mistgate",
        )
        world = resumed.snapshot.world_state
        assert world.clock.turn == 1
        assert len(world.agent_claims) == 1
        assert world.world_activities[-1].activity_kind == "knowledge_propagation"
        assert len(world.player.dialogue_history) == 2
    finally:
        engine.dispose()
        with base_engine.begin() as connection:
            connection.execute(text(f"drop schema if exists {quoted_schema} cascade"))
        base_engine.dispose()


def _provision(uow_factory, now: datetime) -> PrincipalContext:
    product = ProductApplicationService(uow_factory, clock=lambda: now)
    principal = ProductPrincipal(
        id="principal_mistgate",
        identity_provider="test_oidc",
        external_subject="subject_mistgate",
        roles=(PrincipalRole.PLAYER, PrincipalRole.CONTENT_AUTHOR),
        created_at=now,
    )
    profile = PlayerProfile(
        id="profile_mistgate",
        principal_id=principal.id,
        display_name="雾门旅人",
        locale="zh-CN",
        created_at=now,
        updated_at=now,
    )
    product.provision_identity(principal=principal, profile=profile)
    return PrincipalContext(principal_id=principal.id, roles=principal.roles)


def _move_request(
    key: str,
    session_id: str,
    *,
    expected_version: int,
    source: str,
    destination: str,
) -> SubmitPlayerCommand:
    return SubmitPlayerCommand(
        idempotency_key=key,
        player_profile_id="profile_mistgate",
        play_session_id=session_id,
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id="move_to_location",
        actor_id="profile_mistgate",
        target_ids=(destination,),
        location_id=source,
        expected_world_version=expected_version,
        locale="zh-CN",
    )


def _dialogue_request(
    key: str,
    session_id: str,
    *,
    expected_version: int,
    character_id: str,
) -> SubmitPlayerCommand:
    return SubmitPlayerCommand(
        idempotency_key=key,
        player_profile_id="profile_mistgate",
        play_session_id=session_id,
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id="ask_character",
        actor_id="profile_mistgate",
        target_ids=(character_id,),
        location_id="market_row",
        expected_world_version=expected_version,
        locale="zh-CN",
    )


def _exchange_request(
    key: str,
    session_id: str,
    *,
    expected_version: int,
) -> SubmitPlayerCommand:
    return SubmitPlayerCommand(
        idempotency_key=key,
        player_profile_id="profile_mistgate",
        play_session_id=session_id,
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id="negotiate_resource",
        actor_id="profile_mistgate",
        target_ids=("selka",),
        location_id="market_row",
        expected_world_version=expected_version,
        locale="zh-CN",
    )


def _repair_request(
    key: str,
    session_id: str,
    *,
    expected_version: int,
) -> SubmitPlayerCommand:
    return SubmitPlayerCommand(
        idempotency_key=key,
        player_profile_id="profile_mistgate",
        play_session_id=session_id,
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id="repair_regulator",
        actor_id="profile_mistgate",
        target_ids=("dawn_regulator",),
        location_id="old_aqueduct",
        expected_world_version=expected_version,
        locale="zh-CN",
    )


def _action_request(
    key: str,
    session_id: str,
    *,
    expected_version: int,
    action_id: str,
    location_id: str,
    target_id: str | None = None,
) -> SubmitPlayerCommand:
    return SubmitPlayerCommand(
        idempotency_key=key,
        player_profile_id="profile_mistgate",
        play_session_id=session_id,
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id=action_id,
        actor_id="profile_mistgate",
        target_ids=(target_id,) if target_id else (),
        location_id=location_id,
        expected_world_version=expected_version,
        locale="zh-CN",
    )


class _StaticVerifier:
    def verify_subject(self, _token: str) -> str:
        return "subject_mistgate"
