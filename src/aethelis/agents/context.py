from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import Field

from aethelis.algorithms.runtime_features import observation_context_score
from aethelis.schemas.agents import AgentProfile
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.ledger import BeliefRecord, MemoryRecord
from aethelis.schemas.metadata import PublicFact, RumorRecord
from aethelis.schemas.seed import SeedBundle
from aethelis.schemas.world import Entity, Location, WorldResource

IdentifiedT = TypeVar("IdentifiedT", bound="Identified")


class Identified(Protocol):
    id: str


class ObservationContext(AethelisModel):
    agent_id: Identifier
    location: Location
    visible_entities: tuple[Entity, ...] = ()
    visible_resources: tuple[WorldResource, ...] = ()
    visible_agent_ids: tuple[Identifier, ...] = ()
    visible_public_facts: tuple[PublicFact, ...] = ()
    visible_rumors: tuple[RumorRecord, ...] = ()

    def prompt_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "location": {
                "id": self.location.id,
                "name": self.location.name,
                "summary": self.location.summary,
            },
            "visible_entities": [
                {"id": item.id, "name": item.name, "kind": item.kind.value}
                for item in self.visible_entities
            ],
            "visible_resources": [
                {"id": item.id, "name": item.name, "kind": item.kind.value}
                for item in self.visible_resources
            ],
            "visible_agent_ids": list(self.visible_agent_ids),
            "visible_public_facts": [
                {
                    "id": item.id,
                    "claim": item.claim,
                    "subject_ids": list(item.subject_ids),
                    "object_ids": list(item.object_ids),
                }
                for item in self.visible_public_facts
            ],
            "visible_rumors": [
                {
                    "id": item.id,
                    "claim": item.claim,
                    "confidence": item.confidence.value,
                    "subject_ids": list(item.subject_ids),
                    "object_ids": list(item.object_ids),
                }
                for item in self.visible_rumors
            ],
        }

    def compact_prompt_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "location_id": self.location.id,
            "visible_entity_ids": [item.id for item in self.visible_entities],
            "visible_resource_ids": [item.id for item in self.visible_resources],
            "visible_agent_ids": list(self.visible_agent_ids),
            "visible_public_fact_ids": [item.id for item in self.visible_public_facts],
            "visible_rumor_ids": [item.id for item in self.visible_rumors],
        }


class CognitionContext(AethelisModel):
    agent: AgentProfile
    owned_beliefs: tuple[BeliefRecord, ...] = ()
    owned_memories: tuple[MemoryRecord, ...] = ()
    scenario_id: Identifier
    instructions: tuple[str, ...] = Field(default_factory=tuple)

    def prompt_dict(self) -> dict[str, object]:
        return {
            "agent": {
                "id": self.agent.id,
                "name": self.agent.name,
                "role": self.agent.role,
                "current_location_id": self.agent.current_location_id,
                "public_summary": self.agent.public_summary,
                "private_summary": self.agent.private_summary,
                "goals": [
                    {
                        "id": goal.id,
                        "description": goal.description,
                        "priority": goal.priority,
                    }
                    for goal in self.agent.cognitive_state.goals
                ],
                "knowledge_boundaries": list(self.agent.cognitive_state.knowledge_boundaries),
            },
            "owned_beliefs": [
                {
                    "id": belief.id,
                    "kind": belief.kind.value,
                    "claim": belief.claim,
                    "confidence": belief.confidence.value,
                    "truth_status": belief.truth_status.value,
                    "subject_ids": list(belief.subject_ids),
                    "object_ids": list(belief.object_ids),
                }
                for belief in self.owned_beliefs
            ],
            "owned_memories": [
                {
                    "id": memory.id,
                    "kind": memory.kind.value,
                    "summary": memory.summary,
                    "salience": memory.salience,
                }
                for memory in self.owned_memories
            ],
            "scenario_id": self.scenario_id,
            "instructions": list(self.instructions),
        }

    def compact_prompt_dict(self) -> dict[str, object]:
        return {
            "agent": {
                "id": self.agent.id,
                "role": self.agent.role,
                "current_location_id": self.agent.current_location_id,
                "private_summary": self.agent.private_summary,
                "goals": [
                    {
                        "id": goal.id,
                        "description": goal.description,
                        "priority": goal.priority,
                    }
                    for goal in self.agent.cognitive_state.goals
                ],
                "knowledge_boundaries": list(self.agent.cognitive_state.knowledge_boundaries),
            },
            "owned_beliefs": [
                {
                    "id": belief.id,
                    "claim": belief.claim,
                    "confidence": belief.confidence.value,
                    "object_ids": list(belief.object_ids),
                }
                for belief in self.owned_beliefs
            ],
            "owned_memories": [
                {
                    "id": memory.id,
                    "summary": memory.summary,
                }
                for memory in self.owned_memories
            ],
        }


def build_agent_context(
    bundle: SeedBundle,
    *,
    agent_id: str,
    scenario_id: str,
) -> tuple[ObservationContext, CognitionContext]:
    return ObservationBuilder().build_for_agent(
        bundle,
        agent_id=agent_id,
        scenario_id=scenario_id,
    )


class ObservationBuilder:
    """Build visible observation and owned cognition without scheduling agents."""

    def build_for_agent(
        self,
        bundle: SeedBundle,
        *,
        agent_id: str,
        scenario_id: str,
    ) -> tuple[ObservationContext, CognitionContext]:
        observation = self.build_observation(
            bundle,
            actor_id=agent_id,
            scenario_id=scenario_id,
            actor_type="agent",
        )
        agent = _find_one(bundle.agents.agents, agent_id, "agent")
        owned_beliefs = tuple(
            belief for belief in bundle.beliefs.beliefs if belief.owner_agent_id == agent.id
        )
        owned_memories = tuple(
            memory for memory in bundle.memories.memories if memory.owner_agent_id == agent.id
        )
        cognition = CognitionContext(
            agent=agent,
            owned_beliefs=owned_beliefs,
            owned_memories=owned_memories,
            scenario_id=scenario_id,
            instructions=_scenario_instructions(scenario_id),
        )
        return observation, cognition

    def build_observation(
        self,
        bundle: SeedBundle,
        *,
        actor_id: str,
        scenario_id: str,
        actor_type: str = "agent",
    ) -> ObservationContext:
        if actor_type == "player":
            if bundle.world.player is None or bundle.world.player.current_location_id is None:
                raise ValueError("player observation requires player.current_location_id")
            location_id = bundle.world.player.current_location_id
        elif actor_type == "agent":
            agent = _find_one(bundle.agents.agents, actor_id, "agent")
            location_id = agent.current_location_id
        else:
            raise ValueError(f"Unsupported actor_type: {actor_type}")

        location = _find_one(bundle.world.locations, location_id, "location")
        visible_entities = _rank_allowed_context(
            tuple(entity for entity in bundle.world.entities if entity.location_id == location.id),
            scenario_id=scenario_id,
            agent_id=actor_id,
            limit=4,
            min_score=0.45,
        )
        visible_resources = _rank_allowed_context(
            tuple(
            resource
            for resource in bundle.world.resources
            if resource.location_id == location.id and resource.owner_entity_id is None
            ),
            scenario_id=scenario_id,
            agent_id=actor_id,
            limit=4,
            min_score=0.45,
        )
        visible_agent_ids = (
            tuple(
                other.id
                for other in bundle.agents.agents
                if other.current_location_id == location.id and other.id != actor_id
            )
            if actor_type == "agent"
            else tuple(
                other.id
                for other in bundle.agents.agents
                if other.current_location_id == location.id
            )
        )
        public_facts = _rank_allowed_context(
            tuple(
            fact
            for fact in (bundle.metadata.public_facts if bundle.metadata is not None else ())
            if fact.location_id in {None, location.id}
            ),
            scenario_id=scenario_id,
            agent_id=actor_id,
            limit=2,
            min_score=0.50,
        )
        rumors = _rank_allowed_context(
            tuple(
            rumor
            for rumor in (bundle.metadata.rumors if bundle.metadata is not None else ())
            if rumor.location_id in {None, location.id}
            ),
            scenario_id=scenario_id,
            agent_id=actor_id,
            limit=2,
            min_score=0.50,
        )
        return ObservationContext(
            agent_id=actor_id,
            location=location,
            visible_entities=visible_entities,
            visible_resources=visible_resources,
            visible_agent_ids=visible_agent_ids,
            visible_public_facts=public_facts,
            visible_rumors=rumors,
        )


def _legacy_build_agent_context(
    bundle: SeedBundle,
    *,
    agent_id: str,
    scenario_id: str,
) -> tuple[ObservationContext, CognitionContext]:
    agent = _find_one(bundle.agents.agents, agent_id, "agent")
    location = _find_one(bundle.world.locations, agent.current_location_id, "location")
    visible_entities = tuple(
        entity for entity in bundle.world.entities if entity.location_id == location.id
    )
    visible_resources = tuple(
        resource
        for resource in bundle.world.resources
        if resource.location_id == location.id and resource.owner_entity_id is None
    )
    visible_agent_ids = tuple(
        other.id
        for other in bundle.agents.agents
        if other.current_location_id == location.id and other.id != agent.id
    )
    owned_beliefs = tuple(
        belief for belief in bundle.beliefs.beliefs if belief.owner_agent_id == agent.id
    )
    owned_memories = tuple(
        memory for memory in bundle.memories.memories if memory.owner_agent_id == agent.id
    )
    return (
        ObservationContext(
            agent_id=agent.id,
            location=location,
            visible_entities=visible_entities,
            visible_resources=visible_resources,
            visible_agent_ids=visible_agent_ids,
        ),
        CognitionContext(
            agent=agent,
            owned_beliefs=owned_beliefs,
            owned_memories=owned_memories,
            scenario_id=scenario_id,
            instructions=_scenario_instructions(scenario_id),
        ),
    )


def _scenario_instructions(scenario_id: str) -> tuple[str, ...]:
    if scenario_id == "inspect_workshop_safe":
        return (
            "Propose exactly one investigate action.",
            "The action must inspect workshop_safe.",
        )
    return ("Propose exactly one action consistent with the visible context.",)


def _rank_allowed_context(
    items: tuple[IdentifiedT, ...],
    *,
    scenario_id: str,
    agent_id: str,
    limit: int,
    min_score: float,
) -> tuple[IdentifiedT, ...]:
    ranked = tuple(
        item
        for score, item in sorted(
            ((_allowed_context_score(item, scenario_id, agent_id), item) for item in items),
            key=lambda scored: (-scored[0], scored[1].id),
        )
        if score >= min_score
    )
    return ranked[:limit]


def _allowed_context_score(item: Identified, scenario_id: str, agent_id: str) -> float:
    text = f"{item.id} {getattr(item, 'name', '')} {getattr(item, 'claim', '')}".lower()
    scenario_tokens = {token for token in scenario_id.lower().replace("_", " ").split() if token}
    relevance = 1.0 if any(token in text for token in scenario_tokens) else 0.35
    agent_relevance = 1.0 if agent_id.lower() in text else 0.5
    return observation_context_score(
        visibility=1.0,
        access=1.0,
        public_relevance=relevance,
        agent_relevance=agent_relevance,
        privacy_risk=0.0,
    )


def _find_one(items: tuple[IdentifiedT, ...], item_id: str, label: str) -> IdentifiedT:
    for item in items:
        if item.id == item_id:
            return item
    raise ValueError(f"Unknown {label} id: {item_id}")
