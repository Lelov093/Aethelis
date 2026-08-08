from __future__ import annotations

from datetime import UTC, datetime
from types import TracebackType

import pytest

from aethelis.product.contracts import (
    ContentVersionStatus,
    CreateWorldInstanceRequest,
    PlayerProfile,
    PlaySessionStatus,
    PrincipalContext,
    PrincipalRole,
    ProductPrincipal,
    SaveReason,
    WorldContentVersion,
    WorldDefinition,
)
from aethelis.product.errors import ProductAccessDeniedError, ProductCompatibilityError
from aethelis.product.services import ProductApplicationService
from aethelis.schemas.world import PlayerContext, WorldState

NOW = datetime(2026, 8, 1, 8, 0, tzinfo=UTC)


class Store:
    def __init__(self) -> None:
        self.principals = {}
        self.profiles = {}
        self.definitions = {}
        self.versions = {}
        self.grants = {}
        self.instances = {}
        self.snapshots = {}
        self.sessions = {}
        self.saves = {}


class FakeRepositories:
    def __init__(self, store: Store) -> None:
        self.store = store

    def add_principal(self, value):
        self.store.principals[value.id] = value

    def get_principal(self, value_id):
        return self.store.principals.get(value_id)

    def add_profile(self, value):
        self.store.profiles[value.id] = value

    def get_profile(self, value_id):
        return self.store.profiles.get(value_id)

    def get_profile_for_principal(self, principal_id):
        return next(
            (
                profile
                for profile in self.store.profiles.values()
                if profile.principal_id == principal_id
            ),
            None,
        )

    def add_definition(self, value):
        self.store.definitions[value.id] = value

    def get_definition(self, value_id):
        return self.store.definitions.get(value_id)

    def add_content_version(self, value):
        self.store.versions[value.id] = value

    def get_content_version(self, value_id):
        return self.store.versions.get(value_id)

    def add_grant(self, value):
        self.store.grants[(value.principal_id, value.world_instance_id)] = value

    def get_grant(self, principal_id, world_instance_id):
        return self.store.grants.get((principal_id, world_instance_id))

    def add_instance(self, value):
        self.store.instances[value.id] = value

    def get_instance(self, value_id):
        return self.store.instances.get(value_id)

    def list_instances_for_owner(self, owner_principal_id, *, include_archived=False):
        return tuple(
            instance
            for instance in self.store.instances.values()
            if instance.owner_principal_id == owner_principal_id
            and (include_archived or instance.status.value != "archived")
        )

    def update_instance(self, value):
        self.store.instances[value.id] = value

    def add_snapshot(self, value):
        self.store.snapshots[value.id] = value

    def get_snapshot(self, value_id):
        return self.store.snapshots.get(value_id)

    def add_session(self, value):
        self.store.sessions[value.id] = value

    def get_session(self, value_id):
        return self.store.sessions.get(value_id)

    def find_resumable(self, world_instance_id, player_profile_id):
        matches = [
            item
            for item in self.store.sessions.values()
            if item.world_instance_id == world_instance_id
            and item.player_profile_id == player_profile_id
            and item.status in (PlaySessionStatus.ACTIVE, PlaySessionStatus.SUSPENDED)
        ]
        return max(matches, key=lambda item: item.last_active_at, default=None)

    def update_session(self, value):
        self.store.sessions[value.id] = value

    def add_save_point(self, value):
        self.store.saves[value.id] = value

    def get_latest(self, world_instance_id):
        matches = [
            item
            for item in self.store.saves.values()
            if item.world_instance_id == world_instance_id
        ]
        return max(matches, key=lambda item: (item.created_at, item.id), default=None)

    def get_save_point(self, save_point_id):
        return self.store.saves.get(save_point_id)

    def list_save_points(self, world_instance_id):
        return tuple(
            item
            for item in self.store.saves.values()
            if item.world_instance_id == world_instance_id
        )


class FakeUnitOfWork:
    def __init__(self, store: Store) -> None:
        repositories = FakeRepositories(store)
        self.identities = repositories
        self.catalog = repositories
        self.access = repositories
        self.worlds = repositories
        self.sessions = repositories
        self.saves = repositories
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        return None


def world_state() -> WorldState:
    return WorldState(
        schema_version="1.0",
        world_id="world_alpha",
        name="Alpha",
        summary="A test world.",
        locations=(),
        factions=(),
        entities=(),
        resources=(),
        canon_facts=(),
        player=PlayerContext(summary="A newly arrived player."),
    )


def seeded_service(*, roles=(PrincipalRole.PLAYER,)):
    store = Store()
    sequence = iter(range(100))
    service = ProductApplicationService(
        lambda: FakeUnitOfWork(store),
        clock=lambda: NOW,
        id_factory=lambda prefix: f"{prefix}_{next(sequence)}",
    )
    principal = ProductPrincipal(
        id="principal_1",
        identity_provider="test_oidc",
        external_subject="subject_1",
        roles=roles,
        created_at=NOW,
    )
    profile = PlayerProfile(
        id="profile_1",
        principal_id=principal.id,
        display_name="Traveler",
        created_at=NOW,
        updated_at=NOW,
    )
    service.provision_identity(principal=principal, profile=profile)
    context = PrincipalContext(principal_id=principal.id, roles=roles)
    return store, service, context


def register_world(service, context):
    definition = WorldDefinition(id="world_alpha", name="Alpha", created_at=NOW)
    version = WorldContentVersion(
        id="content_1",
        world_definition_id=definition.id,
        schema_version="1.0",
        content_hash="a" * 64,
        status=ContentVersionStatus.PUBLISHED,
        created_at=NOW,
        published_at=NOW,
    )
    service.register_world_content(
        principal=context,
        definition=definition,
        content_version=version,
    )


def test_create_suspend_resume_and_save_boundaries() -> None:
    store, service, context = seeded_service(
        roles=(PrincipalRole.PLAYER, PrincipalRole.CONTENT_AUTHOR)
    )
    register_world(service, context)

    created = service.create_world_instance(
        principal=context,
        request=CreateWorldInstanceRequest(
            world_definition_id="world_alpha",
            content_version_id="content_1",
            player_profile_id="profile_1",
            initial_world_state=world_state(),
        ),
    )
    session = service.start_or_resume_session(
        principal=context,
        world_instance_id=created.world_instance.id,
        player_profile_id="profile_1",
    )
    manual = service.create_save_point(
        principal=context,
        world_instance_id=created.world_instance.id,
        play_session_id=session.id,
    )
    suspended = service.suspend_session(principal=context, session_id=session.id)
    resumed = service.start_or_resume_session(
        principal=context,
        world_instance_id=created.world_instance.id,
        player_profile_id="profile_1",
    )
    resume_state = service.load_resume_state(
        principal=context,
        world_instance_id=created.world_instance.id,
        player_profile_id="profile_1",
    )

    assert created.world_instance.current_world_version == 0
    assert created.snapshot.world_state == world_state()
    assert manual.reason == SaveReason.MANUAL
    assert suspended.status == PlaySessionStatus.SUSPENDED
    assert resumed.id == session.id
    assert resumed.status == PlaySessionStatus.ACTIVE
    assert resume_state.play_session == resumed
    assert len(store.saves) == 3


def test_player_cannot_register_content_or_use_another_profile() -> None:
    _, service, player = seeded_service()
    with pytest.raises(ProductAccessDeniedError) as denied:
        register_world(service, player)
    assert denied.value.code == "content_registration_forbidden"


def test_content_author_can_publish_new_version_of_same_stable_definition() -> None:
    store, service, context = seeded_service(
        roles=(PrincipalRole.PLAYER, PrincipalRole.CONTENT_AUTHOR)
    )
    register_world(service, context)
    definition = WorldDefinition(
        id="world_alpha",
        name="Alpha",
        created_at=datetime(2026, 8, 2, 8, 0, tzinfo=UTC),
    )
    version = WorldContentVersion(
        id="content_2",
        world_definition_id=definition.id,
        schema_version="1.0",
        content_hash="b" * 64,
        status=ContentVersionStatus.PUBLISHED,
        created_at=definition.created_at,
        published_at=definition.created_at,
    )

    service.register_world_content(
        principal=context,
        definition=definition,
        content_version=version,
    )

    assert set(store.versions) == {"content_1", "content_2"}
    assert store.definitions["world_alpha"].created_at == NOW


def test_context_cannot_elevate_persisted_principal_roles() -> None:
    _, service, player = seeded_service()
    elevated = player.model_copy(
        update={"roles": (PrincipalRole.PLAYER, PrincipalRole.CONTENT_AUTHOR)}
    )
    with pytest.raises(ProductAccessDeniedError) as denied:
        register_world(service, elevated)
    assert denied.value.code == "principal_role_escalation"


def test_world_state_schema_must_match_content_version() -> None:
    _, service, context = seeded_service(roles=(PrincipalRole.PLAYER, PrincipalRole.CONTENT_AUTHOR))
    register_world(service, context)
    incompatible = world_state().model_copy(update={"schema_version": "2.0"})

    with pytest.raises(ProductCompatibilityError) as mismatch:
        service.create_world_instance(
            principal=context,
            request=CreateWorldInstanceRequest(
                world_definition_id="world_alpha",
                content_version_id="content_1",
                player_profile_id="profile_1",
                initial_world_state=incompatible,
            ),
        )
    assert mismatch.value.code == "world_state_schema_mismatch"


def test_named_save_forks_independent_timeline_and_supports_archive() -> None:
    store, service, context = seeded_service(
        roles=(PrincipalRole.PLAYER, PrincipalRole.CONTENT_AUTHOR)
    )
    register_world(service, context)
    created = service.create_world_instance(
        principal=context,
        request=CreateWorldInstanceRequest(
            world_definition_id="world_alpha",
            content_version_id="content_1",
            player_profile_id="profile_1",
            initial_world_state=world_state(),
            name="原始时间线",
        ),
    )
    manual = service.create_save_point(
        principal=context,
        world_instance_id=created.world_instance.id,
        name="做出选择之前",
    )

    forked = service.fork_world_from_save(
        principal=context,
        player_profile_id="profile_1",
        source_world_instance_id=created.world_instance.id,
        save_point_id=manual.id,
        name="另一种选择",
    )
    archived = service.archive_world_instance(
        principal=context,
        world_instance_id=created.world_instance.id,
    )
    visible = service.list_world_timelines(principal=context)
    all_timelines = service.list_world_timelines(principal=context, include_archived=True)
    saves = service.list_save_points(
        principal=context,
        world_instance_id=created.world_instance.id,
    )

    assert forked.world_instance.id != created.world_instance.id
    assert forked.world_instance.name == "另一种选择"
    assert forked.world_instance.current_world_version == 0
    assert forked.world_instance.forked_from_world_instance_id == created.world_instance.id
    assert forked.world_instance.forked_from_save_point_id == manual.id
    assert forked.world_instance.forked_from_snapshot_id == manual.snapshot_id
    assert forked.snapshot.world_state.model_copy(update={"player": None}) == (
        created.snapshot.world_state.model_copy(update={"player": None})
    )
    assert forked.snapshot.world_state.player.id == "profile_1"
    assert store.instances[forked.world_instance.id].status.value == "active"
    assert archived.status.value == "archived"
    assert [timeline.name for timeline in visible] == ["另一种选择"]
    assert {timeline.name for timeline in all_timelines} == {"原始时间线", "另一种选择"}
    assert {save.name for save in saves} == {"时间线起点", "做出选择之前"}

    restored = service.restore_world_instance(
        principal=context,
        world_instance_id=created.world_instance.id,
    )
    assert restored.status.value == "active"
