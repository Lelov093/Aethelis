from __future__ import annotations

from types import TracebackType

from pydantic import BaseModel
from sqlalchemy import Select, desc, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import aethelis.db.product_models as records
from aethelis.product.content_contracts import InstalledContentPackage, ProductContentPackage
from aethelis.product.contracts import (
    ContentVersionStatus,
    PlayerProfile,
    PlaySession,
    PlaySessionStatus,
    PrincipalRole,
    PrincipalStatus,
    ProductPrincipal,
    SavePoint,
    SaveReason,
    WorldAccessGrant,
    WorldAccessLevel,
    WorldContentVersion,
    WorldDefinition,
    WorldDefinitionStatus,
    WorldInstance,
    WorldInstanceStatus,
    WorldSnapshotEnvelope,
)
from aethelis.schemas.world import WorldState


class SQLAlchemyIdentityRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_principal(self, principal: ProductPrincipal) -> None:
        data = principal.model_dump(mode="python", exclude={"roles"})
        data["roles_json"] = {"roles": [role.value for role in principal.roles]}
        self._session.add(records.ProductPrincipalRecord(**data))
        self._session.flush()

    def get_principal(self, principal_id: str) -> ProductPrincipal | None:
        row = self._session.get(records.ProductPrincipalRecord, principal_id)
        return _principal(row) if row else None

    def get_principal_by_external_subject(
        self, identity_provider: str, external_subject: str
    ) -> ProductPrincipal | None:
        row = self._session.scalar(
            select(records.ProductPrincipalRecord).where(
                records.ProductPrincipalRecord.identity_provider == identity_provider,
                records.ProductPrincipalRecord.external_subject == external_subject,
            )
        )
        return _principal(row) if row else None

    def add_profile(self, profile: PlayerProfile) -> None:
        data = profile.model_dump(mode="python")
        data["accessibility_preferences"] = dict(profile.accessibility_preferences)
        self._session.add(records.ProductPlayerProfileRecord(**data))

    def get_profile(self, profile_id: str) -> PlayerProfile | None:
        row = self._session.get(records.ProductPlayerProfileRecord, profile_id)
        return _profile(row) if row else None

    def get_profile_for_principal(self, principal_id: str) -> PlayerProfile | None:
        row = self._session.scalar(
            select(records.ProductPlayerProfileRecord).where(
                records.ProductPlayerProfileRecord.principal_id == principal_id
            )
        )
        return _profile(row) if row else None


class SQLAlchemyCatalogRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_definition(self, definition: WorldDefinition) -> None:
        self._session.add(
            records.ProductWorldDefinitionRecord(**definition.model_dump(mode="python"))
        )
        self._session.flush()

    def get_definition(self, definition_id: str) -> WorldDefinition | None:
        row = self._session.get(records.ProductWorldDefinitionRecord, definition_id)
        return _definition(row) if row else None

    def add_content_version(self, version: WorldContentVersion) -> None:
        self._session.add(
            records.ProductWorldContentVersionRecord(**version.model_dump(mode="python"))
        )
        self._session.flush()

    def get_content_version(self, version_id: str) -> WorldContentVersion | None:
        row = self._session.get(records.ProductWorldContentVersionRecord, version_id)
        return _content_version(row) if row else None

    def list_published_content_versions(self) -> tuple[WorldContentVersion, ...]:
        rows = self._session.scalars(
            select(records.ProductWorldContentVersionRecord)
            .where(records.ProductWorldContentVersionRecord.status == "published")
            .order_by(
                records.ProductWorldContentVersionRecord.world_definition_id,
                records.ProductWorldContentVersionRecord.published_at.desc(),
                records.ProductWorldContentVersionRecord.id,
            )
        )
        return tuple(_content_version(row) for row in rows)

    def add_content_package(self, package: InstalledContentPackage) -> None:
        self._session.add(
            records.ProductWorldContentPackageRecord(
                content_version_id=package.content_version_id,
                package_id=package.package_id,
                content_hash=package.content_hash,
                package_json=package.package.model_dump(mode="json"),
                created_at=package.created_at,
            )
        )
        self._session.flush()

    def get_content_package(self, content_version_id: str) -> InstalledContentPackage | None:
        row = self._session.get(records.ProductWorldContentPackageRecord, content_version_id)
        if row is None:
            return None
        return InstalledContentPackage(
            content_version_id=row.content_version_id,
            package_id=row.package_id,
            content_hash=row.content_hash,
            package=ProductContentPackage.model_validate(row.package_json),
            created_at=row.created_at,
        )


class SQLAlchemyAccessRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_grant(self, grant: WorldAccessGrant) -> None:
        self._session.add(records.ProductWorldAccessGrantRecord(**grant.model_dump(mode="python")))

    def get_grant(self, principal_id: str, world_instance_id: str) -> WorldAccessGrant | None:
        row = self._session.scalar(
            select(records.ProductWorldAccessGrantRecord).where(
                records.ProductWorldAccessGrantRecord.principal_id == principal_id,
                records.ProductWorldAccessGrantRecord.world_instance_id == world_instance_id,
            )
        )
        return _grant(row) if row else None


class SQLAlchemyWorldInstanceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_instance(self, instance: WorldInstance) -> None:
        self._session.add(records.ProductWorldInstanceRecord(**instance.model_dump(mode="python")))
        self._session.flush()

    def get_instance(self, instance_id: str) -> WorldInstance | None:
        row = self._session.get(records.ProductWorldInstanceRecord, instance_id)
        return _instance(row) if row else None

    def list_instances_for_owner(
        self, owner_principal_id: str, *, include_archived: bool = False
    ) -> tuple[WorldInstance, ...]:
        statement = select(records.ProductWorldInstanceRecord).where(
            records.ProductWorldInstanceRecord.owner_principal_id == owner_principal_id
        )
        if not include_archived:
            statement = statement.where(records.ProductWorldInstanceRecord.status != "archived")
        rows = self._session.scalars(
            statement.order_by(
                desc(records.ProductWorldInstanceRecord.updated_at),
                records.ProductWorldInstanceRecord.id,
            )
        )
        return tuple(_instance(row) for row in rows)

    def update_instance(self, instance: WorldInstance) -> None:
        row = self._required(records.ProductWorldInstanceRecord, instance.id)
        _assign(row, instance.model_dump(mode="python"), exclude={"id"})

    def add_snapshot(self, snapshot: WorldSnapshotEnvelope) -> None:
        data = snapshot.model_dump(mode="python", exclude={"world_state"})
        data["world_state_json"] = snapshot.world_state.model_dump(mode="json")
        self._session.add(records.ProductWorldSnapshotRecord(**data))
        self._session.flush()

    def get_snapshot(self, snapshot_id: str) -> WorldSnapshotEnvelope | None:
        row = self._session.get(records.ProductWorldSnapshotRecord, snapshot_id)
        return _snapshot(row) if row else None

    def _required(self, record_type: type[object], record_id: str) -> object:
        row = self._session.get(record_type, record_id)
        if row is None:
            raise LookupError(f"record not found: {record_id}")
        return row


class SQLAlchemyPlaySessionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_session(self, play_session: PlaySession) -> None:
        self._session.add(
            records.ProductPlaySessionRecord(**play_session.model_dump(mode="python"))
        )
        self._session.flush()

    def get_session(self, session_id: str) -> PlaySession | None:
        row = self._session.get(records.ProductPlaySessionRecord, session_id)
        return _play_session(row) if row else None

    def find_resumable(self, world_instance_id: str, player_profile_id: str) -> PlaySession | None:
        statement: Select[tuple[records.ProductPlaySessionRecord]] = (
            select(records.ProductPlaySessionRecord)
            .where(
                records.ProductPlaySessionRecord.world_instance_id == world_instance_id,
                records.ProductPlaySessionRecord.player_profile_id == player_profile_id,
                records.ProductPlaySessionRecord.status.in_(("active", "suspended")),
            )
            .order_by(desc(records.ProductPlaySessionRecord.last_active_at))
            .limit(1)
        )
        row = self._session.scalar(statement)
        return _play_session(row) if row else None

    def update_session(self, play_session: PlaySession) -> None:
        row = self._session.get(records.ProductPlaySessionRecord, play_session.id)
        if row is None:
            raise LookupError(f"record not found: {play_session.id}")
        _assign(row, play_session.model_dump(mode="python"), exclude={"id"})


class SQLAlchemySavePointRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_save_point(self, save_point: SavePoint) -> None:
        self._session.add(records.ProductSavePointRecord(**save_point.model_dump(mode="python")))

    def get_latest(self, world_instance_id: str) -> SavePoint | None:
        row = self._session.scalar(
            select(records.ProductSavePointRecord)
            .where(records.ProductSavePointRecord.world_instance_id == world_instance_id)
            .order_by(
                desc(records.ProductSavePointRecord.world_version),
                desc(records.ProductSavePointRecord.created_at),
                desc(records.ProductSavePointRecord.id),
            )
            .limit(1)
        )
        return _save_point(row) if row else None

    def get_save_point(self, save_point_id: str) -> SavePoint | None:
        row = self._session.get(records.ProductSavePointRecord, save_point_id)
        return _save_point(row) if row else None

    def list_save_points(self, world_instance_id: str) -> tuple[SavePoint, ...]:
        rows = self._session.scalars(
            select(records.ProductSavePointRecord)
            .where(records.ProductSavePointRecord.world_instance_id == world_instance_id)
            .order_by(
                desc(records.ProductSavePointRecord.world_version),
                desc(records.ProductSavePointRecord.created_at),
                desc(records.ProductSavePointRecord.id),
            )
        )
        return tuple(_save_point(row) for row in rows)


class SQLAlchemyProductUnitOfWork:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self._session: Session | None = None

    def __enter__(self) -> SQLAlchemyProductUnitOfWork:
        self._session = self._session_factory()
        self.identities = SQLAlchemyIdentityRepository(self._session)
        self.catalog = SQLAlchemyCatalogRepository(self._session)
        self.access = SQLAlchemyAccessRepository(self._session)
        self.worlds = SQLAlchemyWorldInstanceRepository(self._session)
        self.sessions = SQLAlchemyPlaySessionRepository(self._session)
        self.saves = SQLAlchemySavePointRepository(self._session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._session is None:
            return
        if exc_type is not None:
            self._session.rollback()
        self._session.close()
        self._session = None

    def commit(self) -> None:
        self._require_session().commit()

    def rollback(self) -> None:
        self._require_session().rollback()

    def _require_session(self) -> Session:
        if self._session is None:
            raise RuntimeError("unit of work is not active")
        return self._session


def sqlalchemy_product_uow_factory(engine: Engine):  # type: ignore[no-untyped-def]
    factory = sessionmaker(engine, expire_on_commit=False)
    return lambda: SQLAlchemyProductUnitOfWork(factory)


def _assign(row: object, values: dict[str, object], *, exclude: set[str]) -> None:
    for key, value in values.items():
        if key not in exclude:
            setattr(row, key, value)


def _principal(row: records.ProductPrincipalRecord) -> ProductPrincipal:
    return ProductPrincipal(
        id=row.id,
        identity_provider=row.identity_provider,
        external_subject=row.external_subject,
        roles=tuple(PrincipalRole(role) for role in row.roles_json["roles"]),
        status=PrincipalStatus(row.status),
        created_at=row.created_at,
    )


def _profile(row: records.ProductPlayerProfileRecord) -> PlayerProfile:
    return PlayerProfile.model_validate(_row_data(row, PlayerProfile))


def _definition(row: records.ProductWorldDefinitionRecord) -> WorldDefinition:
    data = _row_data(row, WorldDefinition)
    data["status"] = WorldDefinitionStatus(row.status)
    return WorldDefinition.model_validate(data)


def _content_version(row: records.ProductWorldContentVersionRecord) -> WorldContentVersion:
    data = _row_data(row, WorldContentVersion)
    data["status"] = ContentVersionStatus(row.status)
    return WorldContentVersion.model_validate(data)


def _grant(row: records.ProductWorldAccessGrantRecord) -> WorldAccessGrant:
    data = _row_data(row, WorldAccessGrant)
    data["access_level"] = WorldAccessLevel(row.access_level)
    return WorldAccessGrant.model_validate(data)


def _instance(row: records.ProductWorldInstanceRecord) -> WorldInstance:
    data = _row_data(row, WorldInstance)
    data["status"] = WorldInstanceStatus(row.status)
    return WorldInstance.model_validate(data)


def _snapshot(row: records.ProductWorldSnapshotRecord) -> WorldSnapshotEnvelope:
    data = _row_data(row, WorldSnapshotEnvelope, exclude={"world_state"})
    data["world_state"] = WorldState.model_validate(row.world_state_json)
    return WorldSnapshotEnvelope.model_validate(data)


def _play_session(row: records.ProductPlaySessionRecord) -> PlaySession:
    data = _row_data(row, PlaySession)
    data["status"] = PlaySessionStatus(row.status)
    return PlaySession.model_validate(data)


def _save_point(row: records.ProductSavePointRecord) -> SavePoint:
    data = _row_data(row, SavePoint)
    data["reason"] = SaveReason(row.reason)
    return SavePoint.model_validate(data)


def _row_data(
    row: object, model_type: type[BaseModel], *, exclude: set[str] | None = None
) -> dict[str, object]:
    fields = model_type.model_fields
    return {name: getattr(row, name) for name in fields if name not in (exclude or set())}
