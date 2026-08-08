from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from aethelis.schemas.common import AethelisModel, ConfidenceBand, Identifier, RecordStatus


class BeliefKind(StrEnum):
    BELIEF = "belief"
    RUMOR = "rumor"
    PRIVATE_BELIEF = "private_belief"
    REJECTED_CLAIM = "rejected_claim"


class BeliefTruthStatus(StrEnum):
    UNKNOWN = "unknown"
    FALSE = "false"
    CONTESTED = "contested"
    PARTIALLY_TRUE = "partially_true"


class MemoryKind(StrEnum):
    OBSERVATION = "observation"
    CONVERSATION = "conversation"
    TASK = "task"
    WARNING = "warning"


class BeliefRecord(AethelisModel):
    """Agent-local claim. It is not CanonFact and must not mutate canon directly."""

    id: Identifier
    owner_agent_id: Identifier
    kind: BeliefKind
    claim: str = Field(min_length=1)
    truth_status: BeliefTruthStatus = BeliefTruthStatus.UNKNOWN
    confidence: ConfidenceBand
    subject_ids: tuple[Identifier, ...] = ()
    object_ids: tuple[Identifier, ...] = ()
    related_canon_fact_ids: tuple[Identifier, ...] = ()
    source_memory_ids: tuple[Identifier, ...] = ()
    status: RecordStatus = RecordStatus.ACTIVE


class BeliefCandidate(AethelisModel):
    """Non-canon candidate claim routed from player input, rumor, or agent belief.

    A BeliefCandidate is not CanonFact and cannot mutate WorldState. It is a
    governed input to later belief assimilation.
    """

    id: Identifier
    source_type: str = Field(min_length=1)
    source_id: Identifier
    claim: str = Field(min_length=1)
    confidence: ConfidenceBand
    status: RecordStatus = RecordStatus.ACTIVE
    owner_agent_id: Identifier | None = None
    subject_ids: tuple[Identifier, ...] = ()
    object_ids: tuple[Identifier, ...] = ()
    trace_reference_id: Identifier | None = None
    canon_updated: bool = False
    world_state_modified: bool = False


class SecretRecord(AethelisModel):
    """Non-canon secret knowledge scoped to an owner agent or faction."""

    id: Identifier
    owner_agent_id: Identifier | None = None
    owner_faction_id: Identifier | None = None
    claim: str = Field(min_length=1)
    subject_ids: tuple[Identifier, ...] = ()
    object_ids: tuple[Identifier, ...] = ()
    confidence: ConfidenceBand


class MemoryRecord(AethelisModel):
    id: Identifier
    owner_agent_id: Identifier
    kind: MemoryKind
    summary: str = Field(min_length=1)
    related_location_id: Identifier | None = None
    related_agent_ids: tuple[Identifier, ...] = ()
    related_entity_ids: tuple[Identifier, ...] = ()
    related_resource_ids: tuple[Identifier, ...] = ()
    source_event_id: Identifier | None = None
    salience: int = Field(default=1, ge=1, le=5)
