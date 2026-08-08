from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from aethelis.product.content_contracts import (
    AvailableWorldContent,
    InstalledContentPackage,
    ProductContentPackage,
    installed_content_package,
    product_content_hash_candidates,
)
from aethelis.product.contracts import (
    ContentVersionStatus,
    CreateWorldInstanceRequest,
    PlayerProfile,
    PlaySession,
    PlaySessionStatus,
    PrincipalContext,
    PrincipalRole,
    PrincipalStatus,
    ProductPrincipal,
    ResumeState,
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
    utc_timestamp,
)
from aethelis.product.errors import (
    ProductAccessDeniedError,
    ProductCompatibilityError,
    ProductConflictError,
    ProductNotFoundError,
)
from aethelis.product.ports import ProductUnitOfWork
from aethelis.product.projection_contracts import SavePointView, WorldTimelineView
from aethelis.schemas.world import WorldState

UnitOfWorkFactory = Callable[[], ProductUnitOfWork]
Clock = Callable[[], datetime]
IdFactory = Callable[[str], str]

_ACCESS_RANK = {
    WorldAccessLevel.VIEW: 1,
    WorldAccessLevel.PLAY: 2,
    WorldAccessLevel.MANAGE: 3,
}


def system_clock() -> datetime:
    return datetime.now(UTC)


def random_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class ProductApplicationService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        *,
        clock: Clock = system_clock,
        id_factory: IdFactory = random_id,
    ) -> None:
        self._uow_factory = uow_factory
        self._clock = clock
        self._id_factory = id_factory

    def get_player_profile(self, *, principal: PrincipalContext) -> PlayerProfile:
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            profile = uow.identities.get_profile_for_principal(principal.principal_id)
            if profile is None:
                raise ProductNotFoundError(
                    "player_profile_not_found",
                    "Player profile was not found.",
                )
            return profile

    def provision_identity(
        self,
        *,
        principal: ProductPrincipal,
        profile: PlayerProfile,
    ) -> None:
        if profile.principal_id != principal.id:
            raise ProductCompatibilityError(
                "profile_principal_mismatch",
                "Player profile does not belong to the supplied principal.",
            )
        with self._uow_factory() as uow:
            if uow.identities.get_principal(principal.id) is not None:
                raise ProductConflictError(
                    "principal_already_exists",
                    "Principal already exists.",
                )
            if uow.identities.get_profile(profile.id) is not None:
                raise ProductConflictError(
                    "player_profile_already_exists",
                    "Player profile already exists.",
                )
            uow.identities.add_principal(principal)
            uow.identities.add_profile(profile)
            uow.commit()

    def register_world_content(
        self,
        *,
        principal: PrincipalContext,
        definition: WorldDefinition,
        content_version: WorldContentVersion,
        content_package: InstalledContentPackage | None = None,
    ) -> None:
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            if not principal.has_any_role(
                PrincipalRole.CONTENT_AUTHOR,
                PrincipalRole.OPERATOR,
                PrincipalRole.ADMINISTRATOR,
            ):
                raise ProductAccessDeniedError(
                    "content_registration_forbidden",
                    "Principal cannot register world content.",
                )
            if content_version.world_definition_id != definition.id:
                raise ProductCompatibilityError(
                    "content_definition_mismatch",
                    "Content version does not belong to the supplied world definition.",
                )
            existing_definition = uow.catalog.get_definition(definition.id)
            if existing_definition is not None and (
                existing_definition.name != definition.name
                or existing_definition.status != definition.status
            ):
                raise ProductConflictError(
                    "world_definition_conflict",
                    "World definition ID already refers to different content.",
                )
            if uow.catalog.get_content_version(content_version.id) is not None:
                raise ProductConflictError(
                    "content_version_already_exists",
                    "World content version already exists.",
                )
            if content_package is not None:
                if content_package.content_version_id != content_version.id:
                    raise ProductCompatibilityError(
                        "package_content_version_mismatch",
                        "Content package does not belong to the supplied version.",
                    )
                if content_package.package.blueprint.world_definition_id != definition.id:
                    raise ProductCompatibilityError(
                        "package_world_definition_mismatch",
                        "Content package does not belong to the supplied definition.",
                    )
                if content_package.content_hash != content_version.content_hash:
                    raise ProductCompatibilityError(
                        "package_content_hash_mismatch",
                        "Content package hash does not match its version record.",
                    )
            if existing_definition is None:
                uow.catalog.add_definition(definition)
            uow.catalog.add_content_version(content_version)
            if content_package is not None:
                uow.catalog.add_content_package(content_package)
            uow.commit()

    def publish_content_package(
        self,
        *,
        principal: PrincipalContext,
        package: ProductContentPackage,
    ) -> WorldContentVersion:
        now = utc_timestamp(self._clock())
        installed = installed_content_package(package, created_at=now)
        definition = WorldDefinition(
            id=package.blueprint.world_definition_id,
            name=package.text(
                package.blueprint.world_name_key,
                package.blueprint.default_locale,
            ),
            created_at=now,
        )
        version = WorldContentVersion(
            id=package.blueprint.content_version_id,
            world_definition_id=package.blueprint.world_definition_id,
            schema_version=package.blueprint.engine_schema_version,
            content_hash=installed.content_hash,
            status=ContentVersionStatus.PUBLISHED,
            created_at=now,
            published_at=now,
        )
        self.register_world_content(
            principal=principal,
            definition=definition,
            content_version=version,
            content_package=installed,
        )
        return version

    def create_world_instance_from_content(
        self,
        *,
        principal: PrincipalContext,
        content_version_id: str,
        player_profile_id: str,
        name: str = "新的雾门时间线",
    ) -> ResumeState:
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            profile = self._require_profile(uow, player_profile_id, principal.principal_id)
            version = uow.catalog.get_content_version(content_version_id)
            package_record = uow.catalog.get_content_package(content_version_id)
            if version is None or package_record is None:
                raise ProductNotFoundError(
                    "product_content_package_not_found",
                    "Published product content package was not found.",
                )
            package = package_record.package
            if version.content_hash not in product_content_hash_candidates(package):
                raise ProductCompatibilityError(
                    "stored_content_hash_mismatch",
                    "Stored product content no longer matches its published version.",
                )
            world = package.initial_world_state.model_copy(
                update={
                    "player": package.initial_world_state.player.model_copy(
                        update={"id": profile.id}
                    )
                    if package.initial_world_state.player
                    else None
                }
            )
            request = CreateWorldInstanceRequest(
                world_definition_id=package.blueprint.world_definition_id,
                content_version_id=content_version_id,
                player_profile_id=profile.id,
                initial_world_state=world,
                name=name,
            )
        return self.create_world_instance(principal=principal, request=request)

    def list_available_world_content(
        self,
        *,
        principal: PrincipalContext,
    ) -> tuple[AvailableWorldContent, ...]:
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            available = []
            included_definitions: set[str] = set()
            for version in uow.catalog.list_published_content_versions():
                if version.world_definition_id in included_definitions:
                    continue
                package_record = uow.catalog.get_content_package(version.id)
                definition = uow.catalog.get_definition(version.world_definition_id)
                if package_record is None or definition is None:
                    continue
                blueprint = package_record.package.blueprint
                available.append(
                    AvailableWorldContent(
                        world_definition_id=definition.id,
                        world_name=definition.name,
                        content_version_id=version.id,
                        package_id=blueprint.package_id,
                        default_locale=blueprint.default_locale,
                        supported_locales=blueprint.supported_locales,
                    )
                )
                included_definitions.add(version.world_definition_id)
            return tuple(available)

    def create_world_instance(
        self,
        *,
        principal: PrincipalContext,
        request: CreateWorldInstanceRequest,
    ) -> ResumeState:
        now = utc_timestamp(self._clock())
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            self._require_profile(uow, request.player_profile_id, principal.principal_id)
            definition = uow.catalog.get_definition(request.world_definition_id)
            if definition is None:
                raise ProductNotFoundError(
                    "world_definition_not_found",
                    "World definition was not found.",
                )
            version = uow.catalog.get_content_version(request.content_version_id)
            if version is None:
                raise ProductNotFoundError(
                    "content_version_not_found",
                    "World content version was not found.",
                )
            self._validate_creatable_content(definition, version, request)

            instance_id = self._id_factory("world_instance")
            snapshot_id = self._id_factory("world_snapshot")
            instance = WorldInstance(
                id=instance_id,
                owner_principal_id=principal.principal_id,
                world_definition_id=definition.id,
                content_version_id=version.id,
                name=request.name,
                forked_from_world_instance_id=request.forked_from_world_instance_id,
                forked_from_save_point_id=request.forked_from_save_point_id,
                forked_from_snapshot_id=request.forked_from_snapshot_id,
                status=WorldInstanceStatus.ACTIVE,
                current_world_version=0,
                current_snapshot_id=snapshot_id,
                created_at=now,
                updated_at=now,
            )
            snapshot = WorldSnapshotEnvelope(
                id=snapshot_id,
                world_instance_id=instance_id,
                world_version=0,
                content_version_id=version.id,
                engine_schema_version=request.initial_world_state.schema_version,
                state_sha256=_world_state_hash(request.initial_world_state),
                world_state=request.initial_world_state,
                created_at=now,
            )
            grant = WorldAccessGrant(
                id=self._id_factory("world_grant"),
                principal_id=principal.principal_id,
                world_instance_id=instance_id,
                access_level=WorldAccessLevel.MANAGE,
                granted_by_principal_id=principal.principal_id,
                created_at=now,
            )
            initial_save = SavePoint(
                id=self._id_factory("save_point"),
                world_instance_id=instance_id,
                world_version=0,
                snapshot_id=snapshot_id,
                content_version_id=version.id,
                name="时间线起点",
                reason=SaveReason.INSTANCE_CREATED,
                created_at=now,
            )
            uow.worlds.add_instance(instance)
            uow.worlds.add_snapshot(snapshot)
            uow.access.add_grant(grant)
            uow.saves.add_save_point(initial_save)
            uow.commit()
            return ResumeState(
                world_instance=instance,
                snapshot=snapshot,
                latest_save_point=initial_save,
                play_session=None,
            )

    def start_or_resume_session(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
        player_profile_id: str,
    ) -> PlaySession:
        now = utc_timestamp(self._clock())
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            self._require_profile(uow, player_profile_id, principal.principal_id)
            instance = self._require_world_access(
                uow,
                principal,
                world_instance_id,
                WorldAccessLevel.PLAY,
            )
            existing = uow.sessions.find_resumable(world_instance_id, player_profile_id)
            if existing is not None:
                resumed = existing.model_copy(
                    update={
                        "status": PlaySessionStatus.ACTIVE,
                        "last_observed_world_version": instance.current_world_version,
                        "last_active_at": now,
                        "suspended_at": None,
                    }
                )
                uow.sessions.update_session(resumed)
                uow.commit()
                return resumed

            session = PlaySession(
                id=self._id_factory("play_session"),
                world_instance_id=instance.id,
                player_profile_id=player_profile_id,
                status=PlaySessionStatus.ACTIVE,
                entry_world_version=instance.current_world_version,
                last_observed_world_version=instance.current_world_version,
                started_at=now,
                last_active_at=now,
            )
            uow.sessions.add_session(session)
            uow.commit()
            return session

    def create_save_point(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
        reason: SaveReason = SaveReason.MANUAL,
        play_session_id: str | None = None,
        name: str | None = None,
    ) -> SavePoint:
        now = utc_timestamp(self._clock())
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            instance = self._require_world_access(
                uow,
                principal,
                world_instance_id,
                WorldAccessLevel.PLAY,
            )
            if play_session_id is not None:
                self._require_session(uow, play_session_id, instance.id)
            snapshot = uow.worlds.get_snapshot(instance.current_snapshot_id)
            if snapshot is None:
                raise ProductConflictError(
                    "world_head_snapshot_missing",
                    "World instance head snapshot is missing.",
                )
            save = SavePoint(
                id=self._id_factory("save_point"),
                world_instance_id=instance.id,
                world_version=instance.current_world_version,
                snapshot_id=snapshot.id,
                content_version_id=instance.content_version_id,
                play_session_id=play_session_id,
                name=name
                or _default_save_name(
                    snapshot.world_state,
                    instance.current_world_version,
                ),
                reason=reason,
                created_at=now,
            )
            uow.saves.add_save_point(save)
            uow.commit()
            return save

    def list_world_timelines(
        self,
        *,
        principal: PrincipalContext,
        include_archived: bool = False,
    ) -> tuple[WorldTimelineView, ...]:
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            return tuple(
                self._timeline_view(uow, instance)
                for instance in uow.worlds.list_instances_for_owner(
                    principal.principal_id,
                    include_archived=include_archived,
                )
            )

    def list_save_points(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
    ) -> tuple[SavePointView, ...]:
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            self._require_world_access(
                uow,
                principal,
                world_instance_id,
                WorldAccessLevel.VIEW,
                allow_archived=True,
            )
            return tuple(
                self._save_view(uow, save) for save in uow.saves.list_save_points(world_instance_id)
            )

    def fork_world_from_save(
        self,
        *,
        principal: PrincipalContext,
        player_profile_id: str,
        source_world_instance_id: str,
        save_point_id: str,
        name: str,
    ) -> ResumeState:
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            profile = self._require_profile(uow, player_profile_id, principal.principal_id)
            source = self._require_world_access(
                uow,
                principal,
                source_world_instance_id,
                WorldAccessLevel.PLAY,
                allow_archived=True,
            )
            save = uow.saves.get_save_point(save_point_id)
            if save is None or save.world_instance_id != source.id:
                raise ProductNotFoundError(
                    "save_point_not_found",
                    "Save point was not found in the source timeline.",
                )
            snapshot = uow.worlds.get_snapshot(save.snapshot_id)
            if snapshot is None:
                raise ProductConflictError(
                    "save_snapshot_missing",
                    "Save point snapshot is missing.",
                )
            if save.content_version_id != source.content_version_id:
                raise ProductCompatibilityError(
                    "save_content_version_mismatch",
                    "Save point content version does not match its source timeline.",
                )
            forked_world = snapshot.world_state.model_copy(
                update={
                    "player": snapshot.world_state.player.model_copy(update={"id": profile.id})
                    if snapshot.world_state.player
                    else None
                }
            )
            request = CreateWorldInstanceRequest(
                world_definition_id=source.world_definition_id,
                content_version_id=source.content_version_id,
                player_profile_id=profile.id,
                initial_world_state=forked_world,
                name=name,
                forked_from_world_instance_id=source.id,
                forked_from_save_point_id=save.id,
                forked_from_snapshot_id=snapshot.id,
            )
        return self.create_world_instance(principal=principal, request=request)

    def archive_world_instance(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
    ) -> WorldInstance:
        return self._set_world_status(
            principal=principal,
            world_instance_id=world_instance_id,
            status=WorldInstanceStatus.ARCHIVED,
        )

    def restore_world_instance(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
    ) -> WorldInstance:
        return self._set_world_status(
            principal=principal,
            world_instance_id=world_instance_id,
            status=WorldInstanceStatus.ACTIVE,
        )

    def suspend_session(
        self,
        *,
        principal: PrincipalContext,
        session_id: str,
    ) -> PlaySession:
        return self._transition_session(
            principal=principal,
            session_id=session_id,
            status=PlaySessionStatus.SUSPENDED,
        )

    def close_session(
        self,
        *,
        principal: PrincipalContext,
        session_id: str,
    ) -> PlaySession:
        return self._transition_session(
            principal=principal,
            session_id=session_id,
            status=PlaySessionStatus.CLOSED,
        )

    def load_resume_state(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
        player_profile_id: str,
    ) -> ResumeState:
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            self._require_profile(uow, player_profile_id, principal.principal_id)
            instance = self._require_world_access(
                uow,
                principal,
                world_instance_id,
                WorldAccessLevel.PLAY,
            )
            snapshot = uow.worlds.get_snapshot(instance.current_snapshot_id)
            save = uow.saves.get_latest(instance.id)
            if snapshot is None or save is None:
                raise ProductConflictError(
                    "world_resume_boundary_missing",
                    "World instance does not have a complete resumable boundary.",
                )
            session = uow.sessions.find_resumable(instance.id, player_profile_id)
            return ResumeState(
                world_instance=instance,
                snapshot=snapshot,
                latest_save_point=save,
                play_session=session,
            )

    def _set_world_status(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
        status: WorldInstanceStatus,
    ) -> WorldInstance:
        now = utc_timestamp(self._clock())
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            instance = self._require_world_access(
                uow,
                principal,
                world_instance_id,
                WorldAccessLevel.MANAGE,
                allow_archived=True,
            )
            if instance.status == status:
                return instance
            updated = instance.model_copy(update={"status": status, "updated_at": now})
            uow.worlds.update_instance(updated)
            uow.commit()
            return updated

    @staticmethod
    def _save_view(uow: ProductUnitOfWork, save: SavePoint) -> SavePointView:
        snapshot = uow.worlds.get_snapshot(save.snapshot_id)
        if snapshot is None:
            raise ProductConflictError(
                "save_snapshot_missing",
                "Save point snapshot is missing.",
            )
        return SavePointView(
            id=save.id,
            world_instance_id=save.world_instance_id,
            name=save.name or _default_save_name(snapshot.world_state, save.world_version),
            world_version=save.world_version,
            reason=save.reason,
            location_name=_current_location_name(snapshot.world_state),
            created_at=save.created_at,
        )

    @classmethod
    def _timeline_view(
        cls,
        uow: ProductUnitOfWork,
        instance: WorldInstance,
    ) -> WorldTimelineView:
        definition = uow.catalog.get_definition(instance.world_definition_id)
        save = uow.saves.get_latest(instance.id)
        if definition is None or save is None:
            raise ProductConflictError(
                "world_timeline_boundary_missing",
                "World timeline is missing its definition or save boundary.",
            )
        save_view = cls._save_view(uow, save)
        return WorldTimelineView(
            id=instance.id,
            name=instance.name,
            status=instance.status,
            world_name=definition.name,
            world_version=instance.current_world_version,
            location_name=save_view.location_name,
            latest_save=save_view,
            forked_from_world_instance_id=instance.forked_from_world_instance_id,
            forked_from_save_point_id=instance.forked_from_save_point_id,
            updated_at=instance.updated_at,
        )

    def _transition_session(
        self,
        *,
        principal: PrincipalContext,
        session_id: str,
        status: PlaySessionStatus,
    ) -> PlaySession:
        now = utc_timestamp(self._clock())
        with self._uow_factory() as uow:
            self._require_principal(uow, principal)
            session = uow.sessions.get_session(session_id)
            if session is None:
                raise ProductNotFoundError("play_session_not_found", "Play session was not found.")
            instance = self._require_world_access(
                uow,
                principal,
                session.world_instance_id,
                WorldAccessLevel.PLAY,
            )
            if session.status == PlaySessionStatus.CLOSED:
                raise ProductConflictError(
                    "play_session_already_closed",
                    "Closed play session cannot transition again.",
                )
            update: dict[str, object] = {
                "status": status,
                "last_active_at": now,
                "last_observed_world_version": instance.current_world_version,
            }
            save_reason = SaveReason.SESSION_SUSPENDED
            if status == PlaySessionStatus.SUSPENDED:
                update["suspended_at"] = now
            else:
                update["ended_at"] = now
                save_reason = SaveReason.SESSION_CLOSED
            transitioned = session.model_copy(update=update)
            snapshot = uow.worlds.get_snapshot(instance.current_snapshot_id)
            if snapshot is None:
                raise ProductConflictError(
                    "world_head_snapshot_missing",
                    "World instance head snapshot is missing.",
                )
            save = SavePoint(
                id=self._id_factory("save_point"),
                world_instance_id=instance.id,
                world_version=instance.current_world_version,
                snapshot_id=snapshot.id,
                content_version_id=instance.content_version_id,
                play_session_id=session.id,
                reason=save_reason,
                created_at=now,
            )
            uow.sessions.update_session(transitioned)
            uow.saves.add_save_point(save)
            uow.commit()
            return transitioned

    @staticmethod
    def _require_principal(uow: ProductUnitOfWork, principal: PrincipalContext) -> None:
        stored = uow.identities.get_principal(principal.principal_id)
        if stored is None:
            raise ProductAccessDeniedError(
                "principal_not_provisioned",
                "Authenticated principal is not provisioned.",
            )
        if stored.status != PrincipalStatus.ACTIVE:
            raise ProductAccessDeniedError(
                "principal_not_active",
                "Authenticated principal is not active.",
            )
        if not set(principal.roles).issubset(stored.roles):
            raise ProductAccessDeniedError(
                "principal_role_escalation",
                "Authenticated role context exceeds persisted principal roles.",
            )

    @staticmethod
    def _require_profile(
        uow: ProductUnitOfWork,
        profile_id: str,
        principal_id: str,
    ) -> PlayerProfile:
        profile = uow.identities.get_profile(profile_id)
        if profile is None:
            raise ProductNotFoundError("player_profile_not_found", "Player profile was not found.")
        if profile.principal_id != principal_id:
            raise ProductAccessDeniedError(
                "player_profile_forbidden",
                "Player profile is not owned by the authenticated principal.",
            )
        return profile

    @staticmethod
    def _require_world_access(
        uow: ProductUnitOfWork,
        principal: PrincipalContext,
        instance_id: str,
        required: WorldAccessLevel,
        *,
        allow_archived: bool = False,
    ) -> WorldInstance:
        instance = uow.worlds.get_instance(instance_id)
        if instance is None:
            raise ProductNotFoundError("world_instance_not_found", "World instance was not found.")
        if instance.status == WorldInstanceStatus.ARCHIVED and not allow_archived:
            raise ProductConflictError(
                "world_instance_archived",
                "Archived world instance cannot be played.",
            )
        if principal.has_any_role(PrincipalRole.ADMINISTRATOR):
            return instance
        grant = uow.access.get_grant(principal.principal_id, instance.id)
        if grant is None or _ACCESS_RANK[grant.access_level] < _ACCESS_RANK[required]:
            raise ProductAccessDeniedError(
                "world_access_forbidden",
                "Principal does not have the required world access.",
            )
        return instance

    @staticmethod
    def _require_session(
        uow: ProductUnitOfWork,
        session_id: str,
        world_instance_id: str,
    ) -> PlaySession:
        session = uow.sessions.get_session(session_id)
        if session is None:
            raise ProductNotFoundError("play_session_not_found", "Play session was not found.")
        if session.world_instance_id != world_instance_id:
            raise ProductCompatibilityError(
                "play_session_world_mismatch",
                "Play session belongs to another world instance.",
            )
        return session

    @staticmethod
    def _validate_creatable_content(
        definition: WorldDefinition,
        version: WorldContentVersion,
        request: CreateWorldInstanceRequest,
    ) -> None:
        if definition.status != WorldDefinitionStatus.ACTIVE:
            raise ProductConflictError(
                "world_definition_not_active",
                "World definition is not active.",
            )
        if version.status != ContentVersionStatus.PUBLISHED:
            raise ProductConflictError(
                "content_version_not_published",
                "World content version is not published.",
            )
        if version.world_definition_id != definition.id:
            raise ProductCompatibilityError(
                "content_definition_mismatch",
                "Content version does not belong to the requested world definition.",
            )
        if request.initial_world_state.world_id != definition.id:
            raise ProductCompatibilityError(
                "world_state_definition_mismatch",
                "Initial WorldState does not match the requested world definition.",
            )
        if request.initial_world_state.schema_version != version.schema_version:
            raise ProductCompatibilityError(
                "world_state_schema_mismatch",
                "Initial WorldState schema does not match the content version.",
            )


def _world_state_hash(world_state: WorldState) -> str:
    payload = world_state.model_dump(mode="json")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _current_location_name(world_state: WorldState) -> str | None:
    location_id = world_state.player.current_location_id if world_state.player else None
    location = next((item for item in world_state.locations if item.id == location_id), None)
    return location.name if location else None


def _default_save_name(world_state: WorldState, world_version: int) -> str:
    location = _current_location_name(world_state) or "未知地点"
    return f"{location} · 世界版本 {world_version}"
