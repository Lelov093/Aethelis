from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from aethelis.schemas.agents import AgentProfile, RelationshipRecord
from aethelis.schemas.common import AethelisModel, ConfidenceBand, Identifier, RecordStatus
from aethelis.schemas.ledger import BeliefCandidate, BeliefRecord, MemoryRecord


class CanonVisibility(StrEnum):
    PUBLIC = "public"
    HIDDEN_CANON = "hidden_canon"


class EntityKind(StrEnum):
    ARTIFACT = "artifact"
    DEVICE = "device"
    DOOR = "door"
    CONTAINER = "container"
    PERSON_REFERENCE = "person_reference"
    OTHER = "other"


class ResourceKind(StrEnum):
    KEY_ITEM = "key_item"
    MATERIAL = "material"
    INFORMATION = "information"
    SERVICE = "service"


class ResourceDiscoveryState(AethelisModel):
    discovered_by_agent_ids: tuple[Identifier, ...] = ()


class PlayerKnowledgeKind(StrEnum):
    CONFIRMED_FACT = "confirmed_fact"
    RUMOR = "rumor"


class PlayerKnowledgeRecord(AethelisModel):
    id: Identifier
    kind: PlayerKnowledgeKind
    statement: str = Field(min_length=1)
    source_entity_id: Identifier
    subject_ids: tuple[Identifier, ...] = ()
    confidence: ConfidenceBand
    committed_event_id: Identifier


class PlayerRelationshipState(AethelisModel):
    character_id: Identifier
    trust: int = Field(default=0, ge=-5, le=5)
    interaction_count: int = Field(default=0, ge=0)
    last_committed_event_id: Identifier


class DialogueTargetKind(StrEnum):
    CHARACTER = "character"
    WORLD_NARRATIVE = "world_narrative"


class DialogueActKind(StrEnum):
    AUTHORED_TOPIC = "authored_topic"
    GREETING = "greeting"
    QUESTION = "question"
    CLAIM = "claim"
    REQUEST = "request"
    WORLD_OBSERVATION = "world_observation"
    WORLD_ACTION = "world_action"


class RequestedEffectStatus(StrEnum):
    NONE = "none"
    COMMITTED = "committed"
    REJECTED = "rejected"
    NEEDS_CLARIFICATION = "needs_clarification"


class PlayerDialogueTurn(AethelisModel):
    id: Identifier
    interaction_id: Identifier | None = None
    character_id: Identifier | None = None
    dialogue_option_id: Identifier | None = None
    target_kind: DialogueTargetKind = DialogueTargetKind.CHARACTER
    dialogue_act: DialogueActKind = DialogueActKind.AUTHORED_TOPIC
    player_utterance: str | None = Field(default=None, min_length=1, max_length=2000)
    utterance: str = Field(min_length=1)
    knowledge_record_ids: tuple[Identifier, ...] = ()
    belief_candidate_ids: tuple[Identifier, ...] = ()
    requested_effect_status: RequestedEffectStatus = RequestedEffectStatus.NONE
    committed_event_id: Identifier
    expression_evidence: DialogueExpressionEvidence | None = None

    @model_validator(mode="after")
    def validate_target(self) -> PlayerDialogueTurn:
        if self.target_kind == DialogueTargetKind.CHARACTER and self.character_id is None:
            raise ValueError("character dialogue requires character_id")
        if self.target_kind == DialogueTargetKind.WORLD_NARRATIVE and self.character_id is not None:
            raise ValueError("world narrative dialogue cannot carry character_id")
        return self


class DialogueExpressionEvidence(AethelisModel):
    source: Literal["authored", "provider_reviewed", "authored_fallback"]
    provider_name: str | None = None
    model_names: tuple[str, ...] = ()
    raw_text_sha256: tuple[str, ...] = ()
    latency_ms: int = Field(default=0, ge=0)
    usage: dict[str, int] = Field(default_factory=dict)
    review_approved: bool | None = None
    unsupported_claim_count: int = Field(default=0, ge=0)
    failure_code: str | None = Field(default=None, max_length=120)


class PlayerCommitmentStatus(StrEnum):
    ACTIVE = "active"
    FULFILLED = "fulfilled"
    BROKEN = "broken"


class PlayerInventoryItem(AethelisModel):
    id: Identifier
    resource_id: Identifier
    quantity: int = Field(ge=1)
    acquired_from_entity_id: Identifier | None = None
    acquired_event_id: Identifier


class PlayerCommitment(AethelisModel):
    id: Identifier
    counterparty_entity_id: Identifier
    description: str = Field(min_length=1)
    status: PlayerCommitmentStatus = PlayerCommitmentStatus.ACTIVE
    related_resource_ids: tuple[Identifier, ...] = ()
    committed_event_id: Identifier
    resolved_event_id: Identifier | None = None


class PlayerWorldResponse(AethelisModel):
    id: Identifier
    response_option_id: Identifier
    actor_entity_id: Identifier
    response_kind: Literal["civic_support", "social_withdrawal"]
    summary: str = Field(min_length=1)
    committed_event_id: Identifier


class WorldClockState(AethelisModel):
    turn: int = Field(default=0, ge=0)
    elapsed_minutes: int = Field(default=0, ge=0)


class AgentClaimRecord(AethelisModel):
    id: Identifier
    speaker_id: Identifier
    listener_agent_id: Identifier
    statement: str = Field(min_length=1, max_length=2000)
    confidence: ConfidenceBand
    belief_candidate_id: Identifier
    committed_event_id: Identifier


class WorldActivityRecord(AethelisModel):
    id: Identifier
    turn: int = Field(ge=1)
    actor_agent_ids: tuple[Identifier, ...] = Field(min_length=1)
    activity_kind: Literal[
        "independent_action",
        "cooperation",
        "conflict",
        "knowledge_propagation",
    ]
    summary: str = Field(min_length=1, max_length=500)
    location_id: Identifier | None = None
    source_claim_id: Identifier | None = None
    committed_event_id: Identifier


class Location(AethelisModel):
    id: Identifier
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    facilities: tuple[str, ...] = ()


class Faction(AethelisModel):
    id: Identifier
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class Entity(AethelisModel):
    id: Identifier
    name: str = Field(min_length=1)
    kind: EntityKind
    location_id: Identifier | None = None
    status: RecordStatus = RecordStatus.ACTIVE
    summary: str = Field(min_length=1)
    tags: tuple[str, ...] = ()


class WorldResource(AethelisModel):
    id: Identifier
    name: str = Field(min_length=1)
    kind: ResourceKind
    quantity: int = Field(ge=0)
    location_id: Identifier | None = None
    owner_agent_id: Identifier | None = None
    owner_entity_id: Identifier | None = None
    discovery_state: ResourceDiscoveryState = Field(default_factory=ResourceDiscoveryState)
    pressure_weight: float = Field(default=0.0, ge=0.0)
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_single_owner(self) -> WorldResource:
        owners = [self.owner_agent_id, self.owner_entity_id]
        if sum(owner is not None for owner in owners) > 1:
            raise ValueError("resource cannot have both owner_agent_id and owner_entity_id")
        return self


class CanonFact(AethelisModel):
    """Verified world-state fact.

    This is intentionally separate from BeliefRecord, Rumor, SecretRecord, and
    RejectedClaim-like records. Canon facts may be hidden from agents, but they
    are still verified world state.
    """

    id: Identifier
    statement: str = Field(min_length=1)
    visibility: CanonVisibility
    subject_ids: tuple[Identifier, ...] = ()
    object_ids: tuple[Identifier, ...] = ()
    external_ref_ids: tuple[Identifier, ...] = ()
    location_id: Identifier | None = None
    source: str = Field(default="seed", min_length=1)
    tags: tuple[str, ...] = ()


class PlayerContext(AethelisModel):
    id: Identifier = "player"
    summary: str = Field(min_length=1)
    current_location_id: Identifier | None = None
    governance_notes: tuple[str, ...] = ()
    knowledge: tuple[PlayerKnowledgeRecord, ...] = ()
    relationships: tuple[PlayerRelationshipState, ...] = ()
    dialogue_history: tuple[PlayerDialogueTurn, ...] = ()
    inventory: tuple[PlayerInventoryItem, ...] = ()
    commitments: tuple[PlayerCommitment, ...] = ()
    world_responses: tuple[PlayerWorldResponse, ...] = ()


class WorldState(AethelisModel):
    schema_version: str = Field(min_length=1)
    world_id: Identifier
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    locations: tuple[Location, ...]
    factions: tuple[Faction, ...]
    entities: tuple[Entity, ...]
    resources: tuple[WorldResource, ...]
    canon_facts: tuple[CanonFact, ...]
    player: PlayerContext | None = None
    clock: WorldClockState = Field(default_factory=WorldClockState)
    agent_profiles: tuple[AgentProfile, ...] = ()
    agent_beliefs: tuple[BeliefRecord, ...] = ()
    agent_memories: tuple[MemoryRecord, ...] = ()
    agent_relationships: tuple[RelationshipRecord, ...] = ()
    agent_belief_candidates: tuple[BeliefCandidate, ...] = ()
    agent_claims: tuple[AgentClaimRecord, ...] = ()
    world_activities: tuple[WorldActivityRecord, ...] = ()
