from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from aethelis.schemas.common import AethelisModel, ConfidenceBand, Identifier


class VisibilityScope(StrEnum):
    PUBLIC = "public"
    LOCATION_VISIBLE = "location_visible"
    OWNER_PRIVATE = "owner_private"
    FACTION_LIMITED = "faction_limited"
    SECRET = "secret"


class PermissionTag(StrEnum):
    ARCHIVE_SEAL = "archive_seal"
    REPAIR_PERMIT = "repair_permit"
    GUARD_CLEARANCE = "guard_clearance"


class PublicFact(AethelisModel):
    id: Identifier
    claim: str = Field(min_length=1)
    subject_ids: tuple[Identifier, ...] = ()
    object_ids: tuple[Identifier, ...] = ()
    location_id: Identifier | None = None
    visibility_scope: VisibilityScope = VisibilityScope.PUBLIC


class RumorRecord(AethelisModel):
    id: Identifier
    claim: str = Field(min_length=1)
    source_agent_id: Identifier | None = None
    source_label: str | None = None
    confidence: ConfidenceBand
    subject_ids: tuple[Identifier, ...] = ()
    object_ids: tuple[Identifier, ...] = ()
    location_id: Identifier | None = None
    visibility_scope: VisibilityScope = VisibilityScope.LOCATION_VISIBLE

    @model_validator(mode="after")
    def validate_source(self) -> RumorRecord:
        if self.source_agent_id is None and not self.source_label:
            raise ValueError("rumor requires source_agent_id or source_label")
        return self


class PressureSeed(AethelisModel):
    id: Identifier
    pressure_type: Identifier
    level: int = Field(ge=0, le=10)
    location_id: Identifier | None = None
    resource_id: Identifier | None = None
    description: str = Field(min_length=1)


class ActionMetadata(AethelisModel):
    id: Identifier
    action_type: Identifier
    allowed_actor_types: tuple[str, ...]
    required_permission_tags: tuple[PermissionTag, ...] = ()
    target_types: tuple[str, ...] = ()
    description: str = Field(min_length=1)


class MetadataSeed(AethelisModel):
    schema_version: str = Field(min_length=1)
    permission_tags: tuple[PermissionTag, ...] = ()
    public_facts: tuple[PublicFact, ...] = ()
    rumors: tuple[RumorRecord, ...] = ()
    pressure_seeds: tuple[PressureSeed, ...] = ()
    action_metadata: tuple[ActionMetadata, ...] = ()
