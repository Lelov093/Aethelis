from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.world import WorldState


class PrincipalRole(StrEnum):
    PLAYER = "player"
    CONTENT_AUTHOR = "content_author"
    DEVELOPER = "developer"
    OPERATOR = "operator"
    ADMINISTRATOR = "administrator"


class PrincipalStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DISABLED = "disabled"


class WorldAccessLevel(StrEnum):
    VIEW = "view"
    PLAY = "play"
    MANAGE = "manage"


class WorldDefinitionStatus(StrEnum):
    ACTIVE = "active"
    RETIRED = "retired"


class ContentVersionStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    RETIRED = "retired"


class WorldInstanceStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


class PlaySessionStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class SaveReason(StrEnum):
    INSTANCE_CREATED = "instance_created"
    AUTO = "auto"
    MANUAL = "manual"
    SESSION_SUSPENDED = "session_suspended"
    SESSION_CLOSED = "session_closed"


class ProductPrincipal(AethelisModel):
    id: Identifier
    identity_provider: str = Field(min_length=1, max_length=80)
    external_subject: str = Field(min_length=1, max_length=255)
    roles: tuple[PrincipalRole, ...] = Field(default=(PrincipalRole.PLAYER,), min_length=1)
    status: PrincipalStatus = PrincipalStatus.ACTIVE
    created_at: AwareDatetime

    @model_validator(mode="after")
    def validate_unique_roles(self) -> ProductPrincipal:
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("principal roles must be unique")
        return self


class PrincipalContext(AethelisModel):
    principal_id: Identifier
    roles: tuple[PrincipalRole, ...] = Field(min_length=1)
    auth_session_id: Identifier | None = None

    def has_any_role(self, *roles: PrincipalRole) -> bool:
        return bool(set(self.roles).intersection(roles))


class PlayerProfile(AethelisModel):
    id: Identifier
    principal_id: Identifier
    display_name: str = Field(min_length=1, max_length=80)
    locale: str = Field(default="en", min_length=2, max_length=35)
    accessibility_preferences: dict[str, object] = Field(default_factory=dict)
    created_at: AwareDatetime
    updated_at: AwareDatetime


class WorldAccessGrant(AethelisModel):
    id: Identifier
    principal_id: Identifier
    world_instance_id: Identifier
    access_level: WorldAccessLevel
    granted_by_principal_id: Identifier
    created_at: AwareDatetime


class WorldDefinition(AethelisModel):
    id: Identifier
    name: str = Field(min_length=1, max_length=160)
    status: WorldDefinitionStatus = WorldDefinitionStatus.ACTIVE
    created_at: AwareDatetime


class WorldContentVersion(AethelisModel):
    id: Identifier
    world_definition_id: Identifier
    schema_version: str = Field(min_length=1, max_length=80)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ContentVersionStatus = ContentVersionStatus.DRAFT
    created_at: AwareDatetime
    published_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_publication_state(self) -> WorldContentVersion:
        if self.status == ContentVersionStatus.PUBLISHED and self.published_at is None:
            raise ValueError("published content version requires published_at")
        return self


class WorldInstance(AethelisModel):
    id: Identifier
    owner_principal_id: Identifier
    world_definition_id: Identifier
    content_version_id: Identifier
    name: str = Field(default="未命名时间线", min_length=1, max_length=120)
    forked_from_world_instance_id: Identifier | None = None
    forked_from_save_point_id: Identifier | None = None
    forked_from_snapshot_id: Identifier | None = None
    status: WorldInstanceStatus = WorldInstanceStatus.ACTIVE
    current_world_version: int = Field(ge=0)
    current_snapshot_id: Identifier
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_fork_provenance(self) -> WorldInstance:
        provenance = (
            self.forked_from_world_instance_id,
            self.forked_from_save_point_id,
            self.forked_from_snapshot_id,
        )
        if any(provenance) and not all(provenance):
            raise ValueError("fork provenance must be complete")
        return self


class WorldSnapshotEnvelope(AethelisModel):
    id: Identifier
    world_instance_id: Identifier
    world_version: int = Field(ge=0)
    previous_snapshot_id: Identifier | None = None
    source_command_id: Identifier | None = None
    source_committed_event_id: Identifier | None = None
    content_version_id: Identifier
    engine_schema_version: str = Field(min_length=1, max_length=80)
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_state: WorldState
    created_at: AwareDatetime


class PlaySession(AethelisModel):
    id: Identifier
    world_instance_id: Identifier
    player_profile_id: Identifier
    status: PlaySessionStatus = PlaySessionStatus.ACTIVE
    entry_world_version: int = Field(ge=0)
    last_observed_world_version: int = Field(ge=0)
    last_acknowledged_command_id: Identifier | None = None
    started_at: AwareDatetime
    last_active_at: AwareDatetime
    suspended_at: AwareDatetime | None = None
    ended_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_session_timestamps(self) -> PlaySession:
        if self.status == PlaySessionStatus.SUSPENDED and self.suspended_at is None:
            raise ValueError("suspended session requires suspended_at")
        if self.status == PlaySessionStatus.CLOSED and self.ended_at is None:
            raise ValueError("closed session requires ended_at")
        return self


class SavePoint(AethelisModel):
    id: Identifier
    world_instance_id: Identifier
    world_version: int = Field(ge=0)
    snapshot_id: Identifier
    content_version_id: Identifier
    play_session_id: Identifier | None = None
    command_id: Identifier | None = None
    name: str | None = Field(default=None, min_length=1, max_length=120)
    reason: SaveReason
    created_at: AwareDatetime


class CreateWorldInstanceRequest(AethelisModel):
    world_definition_id: Identifier
    content_version_id: Identifier
    player_profile_id: Identifier
    initial_world_state: WorldState
    name: str = Field(default="新的雾门时间线", min_length=1, max_length=120)
    forked_from_world_instance_id: Identifier | None = None
    forked_from_save_point_id: Identifier | None = None
    forked_from_snapshot_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_fork_provenance(self) -> CreateWorldInstanceRequest:
        provenance = (
            self.forked_from_world_instance_id,
            self.forked_from_save_point_id,
            self.forked_from_snapshot_id,
        )
        if any(provenance) and not all(provenance):
            raise ValueError("fork provenance must be complete")
        return self


class ResumeState(AethelisModel):
    world_instance: WorldInstance
    snapshot: WorldSnapshotEnvelope
    latest_save_point: SavePoint
    play_session: PlaySession | None = None


def utc_timestamp(value: datetime) -> datetime:
    """Type-narrowing helper retained for application clock adapters."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("application timestamps must be timezone-aware")
    return value
