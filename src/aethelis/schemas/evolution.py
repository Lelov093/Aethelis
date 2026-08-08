from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from aethelis.schemas.common import AethelisModel, ConfidenceBand, Identifier
from aethelis.schemas.events import VerificationDecision
from aethelis.schemas.ledger import BeliefTruthStatus, MemoryKind


class CausalNodeType(StrEnum):
    COMMITTED_EVENT = "committed_event"
    STATE_DIFF = "state_diff"
    VERIFICATION_RESULT = "verification_result"
    WORLD_TARGET = "world_target"


class CausalEdgeType(StrEnum):
    VERIFIED_BY = "verified_by"
    CAUSED_STATE_DIFF = "caused_state_diff"
    AFFECTED_TARGET = "affected_target"
    SEQUENTIAL_COMMIT = "sequential_commit"


class CausalEventNode(AethelisModel):
    id: Identifier
    node_type: CausalNodeType
    source_id: Identifier
    label: str = Field(min_length=1)


class CausalEventEdge(AethelisModel):
    id: Identifier
    source_node_id: Identifier
    target_node_id: Identifier
    edge_type: CausalEdgeType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CausalEventGraphSummary(AethelisModel):
    nodes: tuple[CausalEventNode, ...] = ()
    edges: tuple[CausalEventEdge, ...] = ()


class PressureUpdateSummary(AethelisModel):
    id: Identifier
    pressure_type: Identifier
    source_step_id: Identifier
    source_event_id: Identifier | None = None
    source_state_diff_id: Identifier | None = None
    source_candidate_id: Identifier | None = None
    decision: VerificationDecision
    location_id: Identifier | None = None
    resource_id: Identifier | None = None
    before_level: int = Field(ge=0, le=10)
    delta: int = Field(ge=-10, le=10)
    after_level: int = Field(ge=0, le=10)
    applied: bool = False
    governance_basis: Identifier = "deterministic_runtime_signal"
    reason: str = Field(min_length=1)


class BeliefUpdateSummary(AethelisModel):
    id: Identifier
    owner_agent_id: Identifier
    source_belief_id: Identifier
    source_event_id: Identifier
    source_state_diff_id: Identifier | None = None
    truth_status_before: BeliefTruthStatus
    truth_status_after: BeliefTruthStatus
    confidence_before: ConfidenceBand
    confidence_after: ConfidenceBand
    canon_updated: bool = False
    governance_basis: Identifier = "commit_applied_state_diff"
    reason: str = Field(min_length=1)


class MemoryUpdateSummary(AethelisModel):
    id: Identifier
    owner_agent_id: Identifier
    memory_kind: MemoryKind
    source_event_id: Identifier
    source_state_diff_id: Identifier | None = None
    related_resource_ids: tuple[Identifier, ...] = ()
    related_entity_ids: tuple[Identifier, ...] = ()
    retained_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    reinforcement_delta: float | None = Field(default=None, ge=0.0, le=1.0)
    suppression_penalty: float | None = Field(default=None, ge=0.0, le=1.0)
    updated_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    summary: str = Field(min_length=1)
    applied: bool = False
    governance_basis: Identifier = "commit_applied_state_diff"
    reason: str = Field(default="Deterministic memory signal.", min_length=1)


class RelationshipUpdateSummary(AethelisModel):
    id: Identifier
    relationship_id: Identifier
    source_agent_id: Identifier
    target_agent_id: Identifier
    source_event_id: Identifier
    source_state_diff_id: Identifier | None = None
    trust_before: int = Field(ge=-5, le=5)
    trust_delta: int = Field(ge=-10, le=10)
    trust_after: int = Field(ge=-5, le=5)
    applied: bool = False
    governance_basis: Identifier = "commit_applied_state_diff"
    reason: str = Field(min_length=1)


class EvolutionUpdateSummary(AethelisModel):
    step_id: Identifier
    scenario_id: Identifier
    decision: VerificationDecision
    applied_update_count: int = Field(default=0, ge=0)
    causal_graph: CausalEventGraphSummary | None = None
    pressure_updates: tuple[PressureUpdateSummary, ...] = ()
    belief_updates: tuple[BeliefUpdateSummary, ...] = ()
    memory_updates: tuple[MemoryUpdateSummary, ...] = ()
    relationship_updates: tuple[RelationshipUpdateSummary, ...] = ()
    opportunity_route: Identifier | None = None
    opportunity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    opportunity_source_ids: tuple[Identifier, ...] = ()
    canon_updated: bool = False
    world_state_updated: bool = False

    def safe_dict(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "scenario_id": self.scenario_id,
            "decision": self.decision.value,
            "applied_update_count": self.applied_update_count,
            "causal_graph": (
                self.causal_graph.model_dump(mode="json") if self.causal_graph is not None else None
            ),
            "pressure_updates": [
                update.model_dump(mode="json") for update in self.pressure_updates
            ],
            "belief_updates": [
                {
                    "id": update.id,
                    "owner_agent_id": update.owner_agent_id,
                    "source_belief_id": update.source_belief_id,
                    "source_event_id": update.source_event_id,
                    "source_state_diff_id": update.source_state_diff_id,
                    "truth_status_before": update.truth_status_before.value,
                    "truth_status_after": update.truth_status_after.value,
                    "confidence_before": update.confidence_before.value,
                    "confidence_after": update.confidence_after.value,
                    "canon_updated": update.canon_updated,
                    "governance_basis": update.governance_basis,
                    "reason": update.reason,
                }
                for update in self.belief_updates
            ],
            "memory_updates": [
                {
                    "id": update.id,
                    "owner_agent_id": update.owner_agent_id,
                    "memory_kind": update.memory_kind.value,
                    "source_event_id": update.source_event_id,
                    "source_state_diff_id": update.source_state_diff_id,
                    "related_resource_ids": list(update.related_resource_ids),
                    "related_entity_ids": list(update.related_entity_ids),
                    "retained_strength": update.retained_strength,
                    "reinforcement_delta": update.reinforcement_delta,
                    "suppression_penalty": update.suppression_penalty,
                    "updated_strength": update.updated_strength,
                    "applied": update.applied,
                    "governance_basis": update.governance_basis,
                    "reason": update.reason,
                }
                for update in self.memory_updates
            ],
            "relationship_updates": [
                {
                    "id": update.id,
                    "relationship_id": update.relationship_id,
                    "source_agent_id": update.source_agent_id,
                    "target_agent_id": update.target_agent_id,
                    "source_event_id": update.source_event_id,
                    "source_state_diff_id": update.source_state_diff_id,
                    "trust_before": update.trust_before,
                    "trust_delta": update.trust_delta,
                    "trust_after": update.trust_after,
                    "applied": update.applied,
                    "governance_basis": update.governance_basis,
                    "reason": update.reason,
                }
                for update in self.relationship_updates
            ],
            "opportunity_route": self.opportunity_route,
            "opportunity_score": self.opportunity_score,
            "opportunity_source_ids": list(self.opportunity_source_ids),
            "canon_updated": self.canon_updated,
            "world_state_updated": self.world_state_updated,
        }


class EvolutionRuntimeState(AethelisModel):
    """Runtime-readable applied evolution state for one in-memory world run.

    This stores only applied commit-derived evolution updates. Trace-only
    non-commit pressure signals are intentionally excluded.
    """

    causal_nodes: tuple[CausalEventNode, ...] = ()
    causal_edges: tuple[CausalEventEdge, ...] = ()
    latest_committed_event_ids: tuple[Identifier, ...] = ()
    causal_update_journal: tuple[Identifier, ...] = ()
    pressure_levels: dict[Identifier, int] = Field(default_factory=dict)
    pressure_update_journal: tuple[Identifier, ...] = ()
    belief_update_journal: tuple[Identifier, ...] = ()
    memory_signal_journal: tuple[Identifier, ...] = ()
    relationship_signal_journal: tuple[Identifier, ...] = ()
    agent_belief_update_counts: dict[Identifier, int] = Field(default_factory=dict)
    agent_memory_signal_counts: dict[Identifier, int] = Field(default_factory=dict)
    relationship_signal_counts: dict[Identifier, int] = Field(default_factory=dict)
    pressure_updates: tuple[PressureUpdateSummary, ...] = ()
    belief_updates: tuple[BeliefUpdateSummary, ...] = ()
    memory_updates: tuple[MemoryUpdateSummary, ...] = ()
    relationship_updates: tuple[RelationshipUpdateSummary, ...] = ()

    def causal_runtime_summary(self) -> dict[str, object]:
        return {
            "causal_node_count": len(self.causal_nodes),
            "causal_edge_count": len(self.causal_edges),
            "latest_committed_event_ids": list(self.latest_committed_event_ids),
            "causal_update_count": len(self.causal_update_journal),
        }

    def pressure_runtime_summary(self) -> dict[str, object]:
        return {
            "pressure_update_count": len(self.pressure_updates),
            "pressure_keys": sorted(self.pressure_levels),
            "latest_pressure_levels": [
                {
                    "pressure_type": pressure_type,
                    "after_level": after_level,
                }
                for pressure_type, after_level in sorted(self.pressure_levels.items())
            ],
            "pressure_update_journal_count": len(self.pressure_update_journal),
        }

    def cognitive_runtime_summary(self) -> dict[str, object]:
        return {
            "belief_update_count": len(self.belief_updates),
            "memory_signal_count": len(self.memory_updates),
            "relationship_signal_count": len(self.relationship_updates),
            "agent_belief_update_counts": dict(sorted(self.agent_belief_update_counts.items())),
            "agent_memory_signal_counts": dict(sorted(self.agent_memory_signal_counts.items())),
            "relationship_signal_counts": dict(sorted(self.relationship_signal_counts.items())),
            "belief_update_journal_count": len(self.belief_update_journal),
            "memory_signal_journal_count": len(self.memory_signal_journal),
            "relationship_signal_journal_count": len(self.relationship_signal_journal),
            "latest_cognitive_update_refs": {
                "belief": list(self.belief_update_journal[-5:]),
                "memory": list(self.memory_signal_journal[-5:]),
                "relationship": list(self.relationship_signal_journal[-5:]),
            },
        }

    def safe_summary(self) -> dict[str, object]:
        return {
            "causal_node_count": len(self.causal_nodes),
            "causal_edge_count": len(self.causal_edges),
            "pressure_update_count": len(self.pressure_updates),
            "causal_runtime_summary": self.causal_runtime_summary(),
            "pressure_runtime_summary": self.pressure_runtime_summary(),
            "cognitive_runtime_summary": self.cognitive_runtime_summary(),
            "belief_update_count": len(self.belief_updates),
            "memory_update_count": len(self.memory_updates),
            "relationship_update_count": len(self.relationship_updates),
            "latest_pressure_levels": [
                {
                    "pressure_type": update.pressure_type,
                    "after_level": update.after_level,
                    "source_step_id": update.source_step_id,
                }
                for update in self.pressure_updates
            ],
            "belief_update_agents": sorted(
                {update.owner_agent_id for update in self.belief_updates}
            ),
            "memory_update_agents": sorted(
                {update.owner_agent_id for update in self.memory_updates}
            ),
            "relationship_update_ids": [
                update.relationship_id for update in self.relationship_updates
            ],
        }
