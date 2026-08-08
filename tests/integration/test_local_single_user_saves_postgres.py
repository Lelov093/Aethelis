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
from aethelis.api.auth import LocalSingleUserPrincipalResolver
from aethelis.db.command_models import COMMAND_TABLE_NAMES
from aethelis.db.command_repository import SQLAlchemyCommandRepository
from aethelis.db.connection import create_db_engine, load_database_settings
from aethelis.db.models import Base
from aethelis.db.product_models import (
    PRODUCT_TABLE_NAMES,
    ProductPlayerProfileRecord,
    ProductPrincipalRecord,
    ProductSavePointRecord,
    ProductWorldContentPackageRecord,
    ProductWorldInstanceRecord,
    ProductWorldSnapshotRecord,
)
from aethelis.db.product_repository import sqlalchemy_product_uow_factory
from aethelis.product.command_service import CommandApplicationService
from aethelis.product.command_worker import CommandWorker, StructuredIntentParser
from aethelis.product.governance_worker import GovernanceWorker
from aethelis.product.local_mode import bootstrap_local_single_user
from aethelis.product.projections import ProjectionService
from aethelis.product.services import ProductApplicationService

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.environ.get("AETHELIS_RUN_DB_TESTS") != "1",
    reason="set AETHELIS_RUN_DB_TESTS=1 for isolated PostgreSQL integration tests",
)


def test_local_bootstrap_named_save_fork_archive_and_restart() -> None:
    base_engine = create_db_engine(load_database_settings())
    schema = f"aethelis_local_save_test_{uuid4().hex}"
    quoted_schema = f'"{schema}"'
    with base_engine.begin() as connection:
        connection.execute(text(f"create schema {quoted_schema}"))
    engine = base_engine.execution_options(schema_translate_map={None: schema})
    tables = [Base.metadata.tables[name] for name in (*PRODUCT_TABLE_NAMES, *COMMAND_TABLE_NAMES)]
    try:
        Base.metadata.create_all(engine, tables=tables)
        now = datetime(2026, 8, 2, 8, 0, tzinfo=UTC)
        uow_factory = sqlalchemy_product_uow_factory(engine)
        context, profile = bootstrap_local_single_user(
            uow_factory=uow_factory,
            repository_root=ROOT,
            principal_id="principal_local_player",
            profile_id="profile_local_player",
            display_name="雾门旅人",
            locale="zh-CN",
            now=now,
        )
        repeated_context, repeated_profile = bootstrap_local_single_user(
            uow_factory=uow_factory,
            repository_root=ROOT,
            principal_id="principal_local_player",
            profile_id="profile_local_player",
            display_name="雾门旅人",
            locale="zh-CN",
            now=now,
        )
        assert repeated_context == context
        assert repeated_profile == profile

        product = ProductApplicationService(uow_factory, clock=lambda: now)
        command_repository = SQLAlchemyCommandRepository(engine)
        commands = CommandApplicationService(
            uow_factory,
            command_repository,
            clock=lambda: now,
        )
        app = create_app(
            command_service=commands,
            product_service=product,
            principal_resolver=LocalSingleUserPrincipalResolver(
                uow_factory,
                principal_id=context.principal_id,
            ),
            projection_service=ProjectionService(uow_factory),
            allowed_origins=("http://localhost:5173",),
        )
        client = TestClient(app)

        me = client.get("/api/v1/me")
        empty = client.get("/api/v1/world-instances")
        created = client.post(
            "/api/v1/world-instances",
            json={
                "content_version_id": "mistgate_product_v1_7_0",
                "player_profile_id": profile.id,
                "name": "第一次抵达",
            },
        )
        assert me.status_code == 200
        assert me.json()["locale"] == "zh-CN"
        assert empty.status_code == 200 and empty.json() == []
        assert created.status_code == 201
        source_id = created.json()["world_instance"]["id"]

        manual = client.post(
            f"/api/v1/world-instances/{source_id}/saves",
            json={"name": "进入档案馆之前"},
        )
        assert manual.status_code == 201
        save_id = manual.json()["id"]
        forked = client.post(
            f"/api/v1/world-instances/{source_id}/saves/{save_id}/fork",
            json={"player_profile_id": profile.id, "name": "另一条线索"},
        )
        assert forked.status_code == 201
        forked_data = forked.json()["world_instance"]
        assert forked_data["id"] != source_id
        assert forked_data["forked_from_world_instance_id"] == source_id
        assert forked_data["forked_from_save_point_id"] == save_id
        assert forked_data["current_world_version"] == 0

        session = client.post(
            f"/api/v1/world-instances/{source_id}/sessions",
            json={"player_profile_id": profile.id},
        )
        submitted = client.post(
            f"/api/v1/world-instances/{source_id}/commands",
            headers={"Idempotency-Key": "local-archive-pending-1"},
            json={
                "player_profile_id": profile.id,
                "play_session_id": session.json()["id"],
                "input_mode": "contextual_action",
                "action_id": "move_to_location",
                "actor_id": profile.id,
                "target_ids": ["central_archive"],
                "location_id": "council_square",
                "expected_world_version": 0,
                "locale": "zh-CN",
            },
        )
        assert submitted.status_code == 202
        parser = CommandWorker(
            command_repository,
            StructuredIntentParser(),
            worker_id="local-parser",
            clock=lambda: now,
        )
        assert parser.run_once() is not None
        archived = client.post(f"/api/v1/world-instances/{source_id}/archive")
        rejected = GovernanceWorker(
            command_repository,
            worker_id="local-governance",
            clock=lambda: now,
        ).run_once()
        visible = client.get("/api/v1/world-instances")
        all_timelines = client.get(
            "/api/v1/world-instances",
            params={"include_archived": "true"},
        )
        archived_saves = client.get(f"/api/v1/world-instances/{source_id}/saves")
        assert archived.status_code == 200
        assert archived.json()["status"] == "archived"
        assert rejected is not None and rejected.status == "rejected"
        assert rejected.resulting_world_version is None
        assert [item["name"] for item in visible.json()] == ["另一条线索"]
        assert {item["name"] for item in all_timelines.json()} == {
            "第一次抵达",
            "另一条线索",
        }
        assert {item["name"] for item in archived_saves.json()} == {
            "时间线起点",
            "进入档案馆之前",
        }

        restarted = TestClient(
            create_app(
                command_service=CommandApplicationService(
                    uow_factory,
                    SQLAlchemyCommandRepository(engine),
                    clock=lambda: now,
                ),
                product_service=ProductApplicationService(uow_factory, clock=lambda: now),
                principal_resolver=LocalSingleUserPrincipalResolver(
                    uow_factory,
                    principal_id=context.principal_id,
                ),
                projection_service=ProjectionService(uow_factory),
            )
        )
        restarted_timelines = restarted.get(
            "/api/v1/world-instances",
            params={"include_archived": "true"},
        )
        assert restarted_timelines.status_code == 200
        assert len(restarted_timelines.json()) == 2

        with engine.connect() as connection:
            counts = {
                "principals": connection.scalar(select(func.count(ProductPrincipalRecord.id))),
                "profiles": connection.scalar(select(func.count(ProductPlayerProfileRecord.id))),
                "packages": connection.scalar(
                    select(func.count(ProductWorldContentPackageRecord.content_version_id))
                ),
                "worlds": connection.scalar(select(func.count(ProductWorldInstanceRecord.id))),
                "snapshots": connection.scalar(select(func.count(ProductWorldSnapshotRecord.id))),
                "saves": connection.scalar(select(func.count(ProductSavePointRecord.id))),
            }
        assert counts == {
            "principals": 1,
            "profiles": 1,
            "packages": 1,
            "worlds": 2,
            "snapshots": 2,
            "saves": 3,
        }
    finally:
        engine.dispose()
        with base_engine.begin() as connection:
            connection.execute(text(f"drop schema if exists {quoted_schema} cascade"))
        base_engine.dispose()
