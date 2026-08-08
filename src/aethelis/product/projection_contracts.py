from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field

from aethelis.product.contracts import SaveReason, WorldInstanceStatus
from aethelis.schemas.common import AethelisModel, Identifier


class VisibleResourceView(AethelisModel):
    id: Identifier
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    quantity: int = Field(ge=0)


class VisibleEntityView(AethelisModel):
    id: Identifier
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    status: str = Field(min_length=1)


class ContextualActionView(AethelisModel):
    action_id: Identifier
    label: str = Field(min_length=1)
    location_id: Identifier | None = None
    target_id: Identifier | None = None
    command_required: bool = True


class SceneView(AethelisModel):
    world_instance_id: Identifier
    world_version: int = Field(ge=0)
    world_turn: int = Field(ge=0)
    elapsed_minutes: int = Field(ge=0)
    location_id: Identifier | None = None
    location_name: str | None = None
    visible_entities: tuple[VisibleEntityView, ...] = ()
    visible_resources: tuple[VisibleResourceView, ...] = ()
    public_facts: tuple[str, ...] = ()
    contextual_actions: tuple[ContextualActionView, ...] = ()
    content_version_id: Identifier
    supports_free_dialogue: bool = False
    supports_world_narrative: bool = False
    recommended_content_version_id: Identifier | None = None


class DialogueExchangeView(AethelisModel):
    id: Identifier
    input_kind: Literal["preset", "free"]
    player_text: str = Field(min_length=1)
    response_text: str = Field(min_length=1)
    requested_effect_status: str = Field(min_length=1)
    visible_effects: tuple[str, ...] = ()
    committed_event_id: Identifier


class DialogueInteractionView(AethelisModel):
    id: Identifier
    target_kind: Literal["character", "world_narrative"]
    target_id: Identifier | None = None
    target_name: str = Field(min_length=1)
    contains_free_expression: bool
    exchanges: tuple[DialogueExchangeView, ...] = ()


class ResumeSummaryView(AethelisModel):
    world_instance_id: Identifier
    world_version: int = Field(ge=0)
    world_name: str = Field(min_length=1)
    location_name: str | None = None
    last_save_reason: str = Field(min_length=1)
    visible_resource_count: int = Field(ge=0)
    resumable_session_id: Identifier | None = None


class MapLocationView(AethelisModel):
    id: Identifier
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    accessibility_label: str = Field(min_length=1)
    is_current: bool
    is_reachable: bool


class MapView(AethelisModel):
    world_instance_id: Identifier
    world_version: int = Field(ge=0)
    current_location_id: Identifier | None = None
    locations: tuple[MapLocationView, ...] = ()


class JournalResourceView(AethelisModel):
    id: Identifier
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    quantity: int = Field(ge=0)
    custody_label: str = Field(min_length=1)
    source_resource_id: Identifier | None = None
    is_player_owned: bool = False


class JournalCommitmentView(AethelisModel):
    id: Identifier
    counterparty_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: str = Field(min_length=1)


class JournalOutcomeView(AethelisModel):
    id: Identifier
    outcome_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)


class JournalWorldResponseView(AethelisModel):
    id: Identifier
    actor_name: str = Field(min_length=1)
    response_kind: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class JournalActivityView(AethelisModel):
    id: Identifier
    turn: int = Field(ge=1)
    actor_names: tuple[str, ...] = ()
    activity_kind: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class OpportunityView(AethelisModel):
    id: Identifier
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    location_id: Identifier
    location_name: str = Field(min_length=1)
    action_id: Identifier
    target_id: Identifier | None = None
    is_at_location: bool
    is_completed: bool = False
    is_optional: bool = False


class SituationView(AethelisModel):
    phase: Literal["unstable", "contained", "repaired"]
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    completed_steps: int = Field(ge=0)
    total_steps: int = Field(ge=1)
    recovery_guidance: tuple[str, ...] = ()


class JournalKnowledgeView(AethelisModel):
    id: Identifier
    kind: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    confidence: str = Field(min_length=1)


class JournalRelationshipView(AethelisModel):
    character_id: Identifier
    character_name: str = Field(min_length=1)
    trust: int = Field(ge=-5, le=5)
    standing_label: str = Field(min_length=1)
    interaction_count: int = Field(ge=0)


class JournalView(AethelisModel):
    world_instance_id: Identifier
    world_version: int = Field(ge=0)
    entries: tuple[str, ...] = ()
    confirmed_facts: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    current_objectives: tuple[str, ...] = ()
    resources: tuple[JournalResourceView, ...] = ()
    opportunities: tuple[OpportunityView, ...] = ()
    situation: SituationView
    knowledge: tuple[JournalKnowledgeView, ...] = ()
    relationships: tuple[JournalRelationshipView, ...] = ()
    commitments: tuple[JournalCommitmentView, ...] = ()
    outcomes: tuple[JournalOutcomeView, ...] = ()
    world_responses: tuple[JournalWorldResponseView, ...] = ()
    world_activities: tuple[JournalActivityView, ...] = ()
    dialogue_interactions: tuple[DialogueInteractionView, ...] = ()


class SavePointView(AethelisModel):
    id: Identifier
    world_instance_id: Identifier
    name: str = Field(min_length=1, max_length=120)
    world_version: int = Field(ge=0)
    reason: SaveReason
    location_name: str | None = None
    created_at: AwareDatetime


class WorldTimelineView(AethelisModel):
    id: Identifier
    name: str = Field(min_length=1, max_length=120)
    status: WorldInstanceStatus
    world_name: str = Field(min_length=1)
    world_version: int = Field(ge=0)
    location_name: str | None = None
    latest_save: SavePointView
    forked_from_world_instance_id: Identifier | None = None
    forked_from_save_point_id: Identifier | None = None
    updated_at: AwareDatetime
