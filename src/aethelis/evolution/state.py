from __future__ import annotations

from aethelis.schemas.events import VerificationDecision
from aethelis.schemas.evolution import (
    CausalEdgeType,
    CausalEventEdge,
    CausalNodeType,
    EvolutionRuntimeState,
    EvolutionUpdateSummary,
)


def append_applied_evolution_update(
    state: EvolutionRuntimeState,
    update: EvolutionUpdateSummary | None,
) -> EvolutionRuntimeState:
    """Return a new runtime state with one applied commit evolution update.

    Non-commit, dry-run, and trace-only updates are ignored. This keeps runtime
    state stricter than trace projection.
    """

    if update is None:
        return state
    if update.decision != VerificationDecision.COMMIT:
        return state
    if not update.world_state_updated or update.applied_update_count <= 0:
        return state
    causal_nodes = state.causal_nodes
    causal_edges = state.causal_edges
    latest_committed_event_ids = state.latest_committed_event_ids
    causal_update_journal = state.causal_update_journal
    if update.causal_graph is not None:
        committed_node = _committed_event_node(update)
        sequential_edge = _sequential_commit_edge(
            previous_event_id=latest_committed_event_ids[-1]
            if latest_committed_event_ids
            else None,
            current_event_id=committed_node.source_id if committed_node is not None else None,
        )
        causal_nodes = _dedupe_by_id((*causal_nodes, *update.causal_graph.nodes))
        causal_edges = _dedupe_by_id(
            (
                *causal_edges,
                *update.causal_graph.edges,
                *((sequential_edge,) if sequential_edge is not None else ()),
            )
        )
        if committed_node is not None:
            latest_committed_event_ids = (*latest_committed_event_ids, committed_node.source_id)
            causal_update_journal = (*causal_update_journal, update.step_id)
    pressure_levels = dict(state.pressure_levels)
    applied_pressure_updates = tuple(
        pressure for pressure in update.pressure_updates if pressure.applied
    )
    for pressure in applied_pressure_updates:
        pressure_levels[pressure.pressure_type] = pressure.after_level
    belief_updates = tuple(belief for belief in update.belief_updates if not belief.canon_updated)
    memory_updates = tuple(memory for memory in update.memory_updates if memory.applied)
    relationship_updates = tuple(
        relationship for relationship in update.relationship_updates if relationship.applied
    )
    agent_belief_update_counts = _increment_counts(
        state.agent_belief_update_counts,
        (belief.owner_agent_id for belief in belief_updates),
    )
    agent_memory_signal_counts = _increment_counts(
        state.agent_memory_signal_counts,
        (memory.owner_agent_id for memory in memory_updates),
    )
    relationship_signal_counts = _increment_counts(
        state.relationship_signal_counts,
        (relationship.relationship_id for relationship in relationship_updates),
    )
    return EvolutionRuntimeState(
        causal_nodes=causal_nodes,
        causal_edges=causal_edges,
        latest_committed_event_ids=latest_committed_event_ids,
        causal_update_journal=causal_update_journal,
        pressure_levels=pressure_levels,
        pressure_update_journal=(
            *state.pressure_update_journal,
            *(pressure.id for pressure in applied_pressure_updates),
        ),
        belief_update_journal=(
            *state.belief_update_journal,
            *(belief.id for belief in belief_updates),
        ),
        memory_signal_journal=(
            *state.memory_signal_journal,
            *(memory.id for memory in memory_updates),
        ),
        relationship_signal_journal=(
            *state.relationship_signal_journal,
            *(relationship.id for relationship in relationship_updates),
        ),
        agent_belief_update_counts=agent_belief_update_counts,
        agent_memory_signal_counts=agent_memory_signal_counts,
        relationship_signal_counts=relationship_signal_counts,
        pressure_updates=(*state.pressure_updates, *applied_pressure_updates),
        belief_updates=(*state.belief_updates, *belief_updates),
        memory_updates=(*state.memory_updates, *memory_updates),
        relationship_updates=(*state.relationship_updates, *relationship_updates),
    )


def evolution_state_safe_context(state: EvolutionRuntimeState) -> dict[str, object]:
    """Safe read model for later activation/context foundations."""

    return state.safe_summary()


def _dedupe_by_id(items):
    deduped = {}
    for item in items:
        deduped[item.id] = item
    return tuple(deduped.values())


def _increment_counts(existing, keys):
    counts = dict(existing)
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return counts


def _committed_event_node(update: EvolutionUpdateSummary):
    if update.causal_graph is None:
        return None
    for node in update.causal_graph.nodes:
        if node.node_type == CausalNodeType.COMMITTED_EVENT:
            return node
    return None


def _sequential_commit_edge(
    *,
    previous_event_id: str | None,
    current_event_id: str | None,
) -> CausalEventEdge | None:
    if (
        previous_event_id is None
        or current_event_id is None
        or previous_event_id == current_event_id
    ):
        return None
    return CausalEventEdge(
        id=f"causal:edge:{previous_event_id}:then:{current_event_id}",
        source_node_id=f"causal:event:{previous_event_id}",
        target_node_id=f"causal:event:{current_event_id}",
        edge_type=CausalEdgeType.SEQUENTIAL_COMMIT,
    )
