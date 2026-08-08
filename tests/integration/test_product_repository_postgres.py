from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import text

import aethelis.db.product_models  # noqa: F401
from aethelis.db.connection import create_db_engine, load_database_settings
from aethelis.db.models import Base
from aethelis.db.product_models import PRODUCT_TABLE_NAMES
from aethelis.db.product_repository import sqlalchemy_product_uow_factory
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
from aethelis.product.services import ProductApplicationService
from aethelis.schemas.world import PlayerContext, WorldState

pytestmark = pytest.mark.skipif(
    os.environ.get("AETHELIS_RUN_DB_TESTS") != "1",
    reason="set AETHELIS_RUN_DB_TESTS=1 for isolated PostgreSQL integration tests",
)


def test_product_world_session_and_restart_readback() -> None:
    base_engine = create_db_engine(load_database_settings())
    schema = f"aethelis_product_test_{uuid4().hex}"
    quoted_schema = f'"{schema}"'
    with base_engine.begin() as connection:
        connection.execute(text(f"create schema {quoted_schema}"))
    engine = base_engine.execution_options(schema_translate_map={None: schema})
    product_tables = [Base.metadata.tables[name] for name in PRODUCT_TABLE_NAMES]
    try:
        Base.metadata.create_all(engine, tables=product_tables)
        now = datetime(2026, 8, 1, 8, 0, tzinfo=UTC)
        service = ProductApplicationService(
            sqlalchemy_product_uow_factory(engine),
            clock=lambda: now,
        )
        principal = ProductPrincipal(
            id="principal_db",
            identity_provider="test_oidc",
            external_subject="subject_db",
            roles=(PrincipalRole.PLAYER, PrincipalRole.CONTENT_AUTHOR),
            created_at=now,
        )
        profile = PlayerProfile(
            id="profile_db",
            principal_id=principal.id,
            display_name="Database Traveler",
            created_at=now,
            updated_at=now,
        )
        context = PrincipalContext(
            principal_id=principal.id,
            roles=(PrincipalRole.PLAYER, PrincipalRole.CONTENT_AUTHOR),
        )
        service.provision_identity(principal=principal, profile=profile)
        service.register_world_content(
            principal=context,
            definition=WorldDefinition(id="world_db", name="DB World", created_at=now),
            content_version=WorldContentVersion(
                id="content_db",
                world_definition_id="world_db",
                schema_version="1.0",
                content_hash="b" * 64,
                status=ContentVersionStatus.PUBLISHED,
                created_at=now,
                published_at=now,
            ),
        )
        created = service.create_world_instance(
            principal=context,
            request=CreateWorldInstanceRequest(
                world_definition_id="world_db",
                content_version_id="content_db",
                player_profile_id=profile.id,
                initial_world_state=WorldState(
                    schema_version="1.0",
                    world_id="world_db",
                    name="DB World",
                    summary="Persisted integration world.",
                    locations=(),
                    factions=(),
                    entities=(),
                    resources=(),
                    canon_facts=(),
                    player=PlayerContext(summary="Persisted player context."),
                ),
            ),
        )
        session = service.start_or_resume_session(
            principal=context,
            world_instance_id=created.world_instance.id,
            player_profile_id=profile.id,
        )
        service.suspend_session(principal=context, session_id=session.id)

        restarted_service = ProductApplicationService(sqlalchemy_product_uow_factory(engine))
        readback = restarted_service.load_resume_state(
            principal=context,
            world_instance_id=created.world_instance.id,
            player_profile_id=profile.id,
        )

        assert readback.world_instance.id == created.world_instance.id
        assert readback.snapshot.world_state.world_id == "world_db"
        assert readback.latest_save_point.snapshot_id == created.snapshot.id
        assert readback.play_session is not None
        assert readback.play_session.status == "suspended"
    finally:
        engine.dispose()
        with base_engine.begin() as connection:
            connection.execute(text(f"drop schema if exists {quoted_schema} cascade"))
        base_engine.dispose()
