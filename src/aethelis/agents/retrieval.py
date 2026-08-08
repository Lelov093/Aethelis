from __future__ import annotations

from pydantic import Field

from aethelis.agents.context import (
    CognitionContext,
    ObservationContext,
    build_agent_context,
)
from aethelis.algorithms.runtime_features import retrieval_rank_score
from aethelis.schemas.agents import RelationshipRecord
from aethelis.schemas.common import AethelisModel, Identifier, RecordStatus
from aethelis.schemas.ledger import BeliefKind, BeliefRecord, MemoryRecord
from aethelis.schemas.metadata import VisibilityScope
from aethelis.schemas.seed import SeedBundle


class CognitionRetrievalSummary(AethelisModel):
    agent_id: Identifier
    scenario_id: Identifier
    own_belief_count: int = 0
    own_memory_count: int = 0
    visible_relationship_count: int = 0
    pressure_context_available: bool = False
    evolution_context_available: bool = False
    hidden_context_used: bool = False
    provider_called: bool = False
    selected_belief_ids: tuple[Identifier, ...] = ()
    filtered_belief_ids: tuple[Identifier, ...] = ()
    selected_memory_ids: tuple[Identifier, ...] = ()
    available_memory_ids: tuple[Identifier, ...] = ()
    suppressed_memory_ids: tuple[Identifier, ...] = ()
    selected_relationship_ids: tuple[Identifier, ...] = ()
    filtered_private_belief_count: int = 0
    cross_agent_memory_filtered_count: int = 0
    filtered_hidden_canon_count: int = 0
    filtered_faction_limited_count: int = 0

    def safe_summary(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "scenario_id": self.scenario_id,
            "own_belief_count": self.own_belief_count,
            "own_memory_count": self.own_memory_count,
            "visible_relationship_count": self.visible_relationship_count,
            "pressure_context_available": self.pressure_context_available,
            "evolution_context_available": self.evolution_context_available,
            "hidden_context_used": self.hidden_context_used,
            "provider_called": self.provider_called,
            "selected_belief_count": len(self.selected_belief_ids),
            "filtered_belief_count": len(self.filtered_belief_ids),
            "selected_memory_count": len(self.selected_memory_ids),
            "available_memory_count": len(self.available_memory_ids),
            "suppressed_memory_count": len(self.suppressed_memory_ids),
            "selected_relationship_count": len(self.selected_relationship_ids),
            "filtered_private_belief_count": self.filtered_private_belief_count,
            "cross_agent_memory_filtered_count": self.cross_agent_memory_filtered_count,
            "filtered_hidden_canon_count": self.filtered_hidden_canon_count,
            "filtered_faction_limited_count": self.filtered_faction_limited_count,
            "selected_belief_ids": list(self.selected_belief_ids),
            "filtered_belief_ids": list(self.filtered_belief_ids),
            "selected_memory_ids": list(self.selected_memory_ids),
            "available_memory_ids": list(self.available_memory_ids),
            "suppressed_memory_ids": list(self.suppressed_memory_ids),
            "selected_relationship_ids": list(self.selected_relationship_ids),
        }


class RetrievedCognitionContext(AethelisModel):
    observation: ObservationContext
    cognition: CognitionContext
    visible_relationships: tuple[RelationshipRecord, ...] = ()
    pressure_context: dict[str, object] | None = None
    evolution_context: dict[str, object] | None = None
    summary: CognitionRetrievalSummary


class ContextSourceRecord(AethelisModel):
    source_id: Identifier
    source_type: Identifier
    score: float = Field(ge=0.0, le=1.0)
    selected: bool = True
    reason: Identifier
    suppress_reason: Identifier | None = None


class PerAgentObservationFrame(AethelisModel):
    agent_id: Identifier
    location_id: Identifier
    visible_entity_ids: tuple[Identifier, ...] = ()
    visible_resource_ids: tuple[Identifier, ...] = ()
    visible_agent_ids: tuple[Identifier, ...] = ()
    visible_public_fact_ids: tuple[Identifier, ...] = ()
    visible_rumor_ids: tuple[Identifier, ...] = ()


class PerAgentRetrievalFrame(AethelisModel):
    agent_id: Identifier
    selected_belief_ids: tuple[Identifier, ...] = ()
    filtered_belief_ids: tuple[Identifier, ...] = ()
    selected_memory_ids: tuple[Identifier, ...] = ()
    suppressed_memory_ids: tuple[Identifier, ...] = ()
    selected_relationship_ids: tuple[Identifier, ...] = ()
    source_records: tuple[ContextSourceRecord, ...] = ()


class ContextBoundaryFlags(AethelisModel):
    hidden_context_used: bool = False
    provider_called: bool = False
    private_cross_agent_beliefs_filtered: int = Field(default=0, ge=0)
    cross_agent_memories_filtered: int = Field(default=0, ge=0)
    hidden_canon_filtered: int = Field(default=0, ge=0)
    rejected_or_outdated_filtered_ids: tuple[Identifier, ...] = ()
    faction_limited_filtered: int = Field(default=0, ge=0)
    can_modify_world_state: bool = False
    can_mutate_canon: bool = False


class ActiveAgentContextFrame(AethelisModel):
    agent_id: Identifier
    scenario_id: Identifier
    observation: PerAgentObservationFrame
    retrieval: PerAgentRetrievalFrame
    packed_source_ids: tuple[Identifier, ...] = ()
    suppressed_source_ids: tuple[Identifier, ...] = ()
    context_budget: int = Field(ge=1)
    boundary_flags: ContextBoundaryFlags


class MultiAgentStepContext(AethelisModel):
    """Read-only active-set context. It cannot mutate WorldState or Canon."""

    step_id: Identifier
    scenario_id: Identifier
    active_agent_ids: tuple[Identifier, ...]
    frames: tuple[ActiveAgentContextFrame, ...]
    context_budget_per_agent: int = Field(ge=1)
    no_shared_omniscient_context: bool = True
    can_modify_world_state: bool = False
    can_mutate_canon: bool = False

    def frame_for(self, agent_id: str) -> ActiveAgentContextFrame:
        for frame in self.frames:
            if frame.agent_id == agent_id:
                return frame
        raise ValueError(f"Unknown active agent id: {agent_id}")

    def safe_summary(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "scenario_id": self.scenario_id,
            "active_agent_ids": list(self.active_agent_ids),
            "context_budget_per_agent": self.context_budget_per_agent,
            "no_shared_omniscient_context": self.no_shared_omniscient_context,
            "can_modify_world_state": self.can_modify_world_state,
            "can_mutate_canon": self.can_mutate_canon,
            "frames": [
                {
                    "agent_id": frame.agent_id,
                    "observation": frame.observation.model_dump(mode="json"),
                    "retrieval": frame.retrieval.model_dump(mode="json"),
                    "packed_source_ids": list(frame.packed_source_ids),
                    "suppressed_source_ids": list(frame.suppressed_source_ids),
                    "boundary_flags": frame.boundary_flags.model_dump(mode="json"),
                }
                for frame in self.frames
            ],
        }


class CognitionRetriever:
    """Retrieve local cognition while preserving knowledge boundaries."""

    def retrieve(
        self,
        bundle: SeedBundle,
        *,
        agent_id: str,
        scenario_id: str,
        pressure_context: dict[str, object] | None = None,
        evolution_context: dict[str, object] | None = None,
    ) -> RetrievedCognitionContext:
        observation, cognition = build_agent_context(
            bundle,
            agent_id=agent_id,
            scenario_id=scenario_id,
        )
        visible_relationships = tuple(
            relationship
            for relationship in bundle.agents.relationships
            if agent_id in {relationship.source_agent_id, relationship.target_agent_id}
        )
        ranked_beliefs, filtered_belief_ids = _rank_beliefs(cognition.owned_beliefs)
        ranked_memories, available_memory_ids, suppressed_memory_ids = _rank_memories(
            cognition.owned_memories
        )
        cognition = cognition.model_copy(
            update={
                "owned_beliefs": ranked_beliefs,
                "owned_memories": ranked_memories,
            }
        )
        summary = CognitionRetrievalSummary(
            agent_id=agent_id,
            scenario_id=scenario_id,
            own_belief_count=len(cognition.owned_beliefs),
            own_memory_count=len(cognition.owned_memories),
            visible_relationship_count=len(visible_relationships),
            pressure_context_available=pressure_context is not None,
            evolution_context_available=evolution_context is not None,
            selected_belief_ids=tuple(belief.id for belief in ranked_beliefs),
            filtered_belief_ids=filtered_belief_ids,
            selected_memory_ids=tuple(memory.id for memory in ranked_memories),
            available_memory_ids=available_memory_ids,
            suppressed_memory_ids=suppressed_memory_ids,
            selected_relationship_ids=tuple(
                relationship.id for relationship in visible_relationships
            ),
            filtered_private_belief_count=_private_cross_agent_belief_count(bundle, agent_id),
            cross_agent_memory_filtered_count=_cross_agent_memory_count(bundle, agent_id),
            filtered_hidden_canon_count=_hidden_canon_count(bundle),
            filtered_faction_limited_count=_faction_limited_filtered_count(bundle, agent_id),
        )
        return RetrievedCognitionContext(
            observation=observation,
            cognition=cognition,
            visible_relationships=visible_relationships,
            pressure_context=pressure_context,
            evolution_context=evolution_context,
            summary=summary,
        )


def build_multi_agent_step_context(
    bundle: SeedBundle,
    *,
    step_id: str,
    scenario_id: str,
    active_agent_ids: tuple[str, ...],
    context_budget_per_agent: int = 12,
    pressure_context: dict[str, object] | None = None,
    evolution_context: dict[str, object] | None = None,
) -> MultiAgentStepContext:
    if not active_agent_ids:
        raise ValueError("active_agent_ids must not be empty")
    retriever = CognitionRetriever()
    frames = tuple(
        _active_agent_frame(
            retriever.retrieve(
                bundle,
                agent_id=agent_id,
                scenario_id=scenario_id,
                pressure_context=pressure_context,
                evolution_context=evolution_context,
            ),
            context_budget=context_budget_per_agent,
        )
        for agent_id in active_agent_ids
    )
    return MultiAgentStepContext(
        step_id=step_id,
        scenario_id=scenario_id,
        active_agent_ids=active_agent_ids,
        frames=frames,
        context_budget_per_agent=context_budget_per_agent,
    )


def _active_agent_frame(
    retrieved: RetrievedCognitionContext,
    *,
    context_budget: int,
) -> ActiveAgentContextFrame:
    records = _context_source_records(retrieved)
    packed, suppressed = _pack_context_records(records, context_budget)
    observation = retrieved.observation
    summary = retrieved.summary
    return ActiveAgentContextFrame(
        agent_id=summary.agent_id,
        scenario_id=summary.scenario_id,
        observation=PerAgentObservationFrame(
            agent_id=summary.agent_id,
            location_id=observation.location.id,
            visible_entity_ids=tuple(item.id for item in observation.visible_entities),
            visible_resource_ids=tuple(item.id for item in observation.visible_resources),
            visible_agent_ids=observation.visible_agent_ids,
            visible_public_fact_ids=tuple(item.id for item in observation.visible_public_facts),
            visible_rumor_ids=tuple(item.id for item in observation.visible_rumors),
        ),
        retrieval=PerAgentRetrievalFrame(
            agent_id=summary.agent_id,
            selected_belief_ids=summary.selected_belief_ids,
            filtered_belief_ids=summary.filtered_belief_ids,
            selected_memory_ids=summary.selected_memory_ids,
            suppressed_memory_ids=summary.suppressed_memory_ids,
            selected_relationship_ids=summary.selected_relationship_ids,
            source_records=(*packed, *suppressed),
        ),
        packed_source_ids=tuple(record.source_id for record in packed),
        suppressed_source_ids=tuple(record.source_id for record in suppressed),
        context_budget=context_budget,
        boundary_flags=ContextBoundaryFlags(
            hidden_context_used=summary.hidden_context_used,
            provider_called=summary.provider_called,
            private_cross_agent_beliefs_filtered=summary.filtered_private_belief_count,
            cross_agent_memories_filtered=summary.cross_agent_memory_filtered_count,
            hidden_canon_filtered=summary.filtered_hidden_canon_count,
            rejected_or_outdated_filtered_ids=summary.filtered_belief_ids,
            faction_limited_filtered=summary.filtered_faction_limited_count,
        ),
    )


def _context_source_records(
    retrieved: RetrievedCognitionContext,
) -> tuple[ContextSourceRecord, ...]:
    observation = retrieved.observation
    cognition = retrieved.cognition
    selected_belief_ids = set(retrieved.summary.selected_belief_ids)
    selected_memory_ids = set(retrieved.summary.selected_memory_ids)
    records = [
        *(
            ContextSourceRecord(
                source_id=item.id,
                source_type="observation_entity",
                score=1.0,
                reason="visible_location_context",
            )
            for item in observation.visible_entities
        ),
        *(
            ContextSourceRecord(
                source_id=item.id,
                source_type="observation_resource",
                score=1.0,
                reason="visible_location_context",
            )
            for item in observation.visible_resources
        ),
        *(
            ContextSourceRecord(
                source_id=agent_id,
                source_type="observation_nearby_agent",
                score=0.85,
                reason="same_location_agent_visible",
            )
            for agent_id in observation.visible_agent_ids
        ),
        *(
            ContextSourceRecord(
                source_id=item.id,
                source_type="observation_public_fact",
                score=0.8,
                reason="public_or_location_fact",
            )
            for item in observation.visible_public_facts
        ),
        *(
            ContextSourceRecord(
                source_id=item.id,
                source_type="observation_rumor",
                score=0.65,
                reason="location_visible_rumor",
            )
            for item in observation.visible_rumors
        ),
        *(
            ContextSourceRecord(
                source_id=belief.id,
                source_type="belief",
                score=_belief_score(belief),
                reason="owned_active_belief",
            )
            for belief in cognition.owned_beliefs
            if belief.id in selected_belief_ids
        ),
        *(
            ContextSourceRecord(
                source_id=memory.id,
                source_type="memory",
                score=_memory_score(memory),
                reason="owned_memory_rerank",
            )
            for memory in cognition.owned_memories
            if memory.id in selected_memory_ids
        ),
        *(
            ContextSourceRecord(
                source_id=relationship.id,
                source_type="relationship",
                score=_relationship_score(relationship),
                reason="visible_agent_relationship",
            )
            for relationship in retrieved.visible_relationships
        ),
    ]
    return tuple(
        sorted(records, key=lambda record: (-record.score, record.source_type, record.source_id))
    )


def _pack_context_records(
    records: tuple[ContextSourceRecord, ...],
    context_budget: int,
) -> tuple[tuple[ContextSourceRecord, ...], tuple[ContextSourceRecord, ...]]:
    if context_budget < 1:
        raise ValueError("context_budget must be >= 1")
    packed = tuple(
        record.model_copy(update={"selected": True}) for record in records[:context_budget]
    )
    suppressed = tuple(
        record.model_copy(update={"selected": False, "suppress_reason": "context_budget"})
        for record in records[context_budget:]
    )
    return packed, suppressed


def _rank_beliefs(
    beliefs: tuple[BeliefRecord, ...],
) -> tuple[tuple[BeliefRecord, ...], tuple[Identifier, ...]]:
    selected = tuple(
        belief
        for belief in beliefs
        if belief.status == RecordStatus.ACTIVE and belief.kind != BeliefKind.REJECTED_CLAIM
    )
    filtered = tuple(belief.id for belief in beliefs if belief not in selected)
    return (
        tuple(sorted(selected, key=lambda belief: (-_belief_score(belief), belief.id))),
        filtered,
    )


def _belief_score(belief: BeliefRecord) -> float:
    confidence = {"low": 0.25, "medium": 0.55, "high": 0.85}
    return retrieval_rank_score(
        salience=0.6,
        recency=0.5,
        belief_confidence=confidence.get(belief.confidence.value, 0.5),
        suppression=0.0,
    )


def _rank_memories(
    memories: tuple[MemoryRecord, ...],
) -> tuple[tuple[MemoryRecord, ...], tuple[Identifier, ...], tuple[Identifier, ...]]:
    scored = tuple((_memory_score(memory), memory) for memory in memories)
    ranked_available = tuple(
        memory
        for score, memory in sorted(scored, key=lambda item: (-item[0], item[1].id))
        if score >= 0.45
    )
    suppressed = tuple(memory.id for score, memory in scored if score < 0.45)
    return (
        ranked_available,
        tuple(memory.id for _, memory in scored),
        suppressed,
    )


def _memory_score(memory) -> float:
    salience = memory.salience / 5
    recency = 1.0 if memory.source_event_id else 0.35
    reinforcement = 0.15 if memory.source_event_id else 0.0
    suppression = 0.35 if memory.salience <= 1 else 0.0
    semantic_proxy = 0.65 if memory.related_entity_ids or memory.related_resource_ids else 0.45
    return retrieval_rank_score(
        salience=salience,
        recency=recency,
        belief_confidence=min(1.0, semantic_proxy + reinforcement),
        suppression=suppression,
    )


def _relationship_score(relationship: RelationshipRecord) -> float:
    return max(0.0, min(1.0, (relationship.trust + 5) / 10))


def _private_cross_agent_belief_count(bundle: SeedBundle, agent_id: str) -> int:
    return sum(
        1
        for belief in bundle.beliefs.beliefs
        if belief.owner_agent_id != agent_id and belief.kind == BeliefKind.PRIVATE_BELIEF
    )


def _cross_agent_memory_count(bundle: SeedBundle, agent_id: str) -> int:
    return sum(1 for memory in bundle.memories.memories if memory.owner_agent_id != agent_id)


def _hidden_canon_count(bundle: SeedBundle) -> int:
    return sum(1 for fact in bundle.world.canon_facts if fact.visibility == "hidden_canon")


def _faction_limited_filtered_count(bundle: SeedBundle, agent_id: str) -> int:
    agent_faction = next(
        (agent.faction_id for agent in bundle.agents.agents if agent.id == agent_id),
        None,
    )
    metadata = bundle.metadata
    metadata_count = 0
    if metadata is not None:
        metadata_count = sum(
            1
            for item in (*metadata.public_facts, *metadata.rumors)
            if item.visibility_scope == VisibilityScope.FACTION_LIMITED
        )
    secret_count = sum(
        1
        for secret in bundle.beliefs.secrets
        if secret.owner_faction_id is not None and secret.owner_faction_id != agent_faction
    )
    return metadata_count + secret_count
