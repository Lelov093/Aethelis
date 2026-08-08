from __future__ import annotations

from types import TracebackType
from typing import Protocol, Self

from aethelis.product.content_contracts import InstalledContentPackage
from aethelis.product.contracts import (
    PlayerProfile,
    PlaySession,
    ProductPrincipal,
    SavePoint,
    WorldAccessGrant,
    WorldContentVersion,
    WorldDefinition,
    WorldInstance,
    WorldSnapshotEnvelope,
)


class IdentityRepository(Protocol):
    def add_principal(self, principal: ProductPrincipal) -> None: ...

    def get_principal(self, principal_id: str) -> ProductPrincipal | None: ...

    def get_principal_by_external_subject(
        self, identity_provider: str, external_subject: str
    ) -> ProductPrincipal | None: ...

    def add_profile(self, profile: PlayerProfile) -> None: ...

    def get_profile(self, profile_id: str) -> PlayerProfile | None: ...

    def get_profile_for_principal(self, principal_id: str) -> PlayerProfile | None: ...


class CatalogRepository(Protocol):
    def add_definition(self, definition: WorldDefinition) -> None: ...

    def get_definition(self, definition_id: str) -> WorldDefinition | None: ...

    def add_content_version(self, version: WorldContentVersion) -> None: ...

    def get_content_version(self, version_id: str) -> WorldContentVersion | None: ...

    def list_published_content_versions(self) -> tuple[WorldContentVersion, ...]: ...

    def add_content_package(self, package: InstalledContentPackage) -> None: ...

    def get_content_package(self, content_version_id: str) -> InstalledContentPackage | None: ...


class AccessRepository(Protocol):
    def add_grant(self, grant: WorldAccessGrant) -> None: ...

    def get_grant(self, principal_id: str, world_instance_id: str) -> WorldAccessGrant | None: ...


class WorldInstanceRepository(Protocol):
    def add_instance(self, instance: WorldInstance) -> None: ...

    def get_instance(self, instance_id: str) -> WorldInstance | None: ...

    def list_instances_for_owner(
        self, owner_principal_id: str, *, include_archived: bool = False
    ) -> tuple[WorldInstance, ...]: ...

    def update_instance(self, instance: WorldInstance) -> None: ...

    def add_snapshot(self, snapshot: WorldSnapshotEnvelope) -> None: ...

    def get_snapshot(self, snapshot_id: str) -> WorldSnapshotEnvelope | None: ...


class PlaySessionRepository(Protocol):
    def add_session(self, session: PlaySession) -> None: ...

    def get_session(self, session_id: str) -> PlaySession | None: ...

    def find_resumable(
        self, world_instance_id: str, player_profile_id: str
    ) -> PlaySession | None: ...

    def update_session(self, session: PlaySession) -> None: ...


class SavePointRepository(Protocol):
    def add_save_point(self, save_point: SavePoint) -> None: ...

    def get_latest(self, world_instance_id: str) -> SavePoint | None: ...

    def get_save_point(self, save_point_id: str) -> SavePoint | None: ...

    def list_save_points(self, world_instance_id: str) -> tuple[SavePoint, ...]: ...


class ProductUnitOfWork(Protocol):
    identities: IdentityRepository
    catalog: CatalogRepository
    access: AccessRepository
    worlds: WorldInstanceRepository
    sessions: PlaySessionRepository
    saves: SavePointRepository

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...
