from __future__ import annotations

from aethelis.algorithms.runtime_features import (
    belief_confidence_score,
    bounded_trust_update,
    causal_edge_confidence,
    decay_reinforcement_score,
    ewma_pressure_update,
    opportunity_score,
)
from aethelis.schemas.agents import RelationshipRecord
from aethelis.schemas.common import ConfidenceBand
from aethelis.schemas.events import EventCandidate, VerificationDecision, VerificationResult
from aethelis.schemas.evolution import (
    BeliefUpdateSummary,
    CausalEdgeType,
    CausalEventEdge,
    CausalEventGraphSummary,
    CausalEventNode,
    CausalNodeType,
    EvolutionUpdateSummary,
    MemoryUpdateSummary,
    PressureUpdateSummary,
    RelationshipUpdateSummary,
)
from aethelis.schemas.ledger import BeliefRecord, BeliefTruthStatus, MemoryKind, MemoryRecord
from aethelis.schemas.metadata import PressureSeed
from aethelis.schemas.seed import SeedBundle


class DeterministicEvolutionBuilder:
    """Build traceable post-verification evolution summaries."""

    def build_for_step(
        self,
        *,
        bundle: SeedBundle,
        step_id: str,
        scenario_id: str,
        agent_id: str,
        decision: VerificationDecision,
        committed_event_id: str | None,
        state_diff_id: str | None,
        verification_result_id: str | None,
        event_candidate_id: str | None,
        state_diff_applied: bool,
        verification_result: VerificationResult | None = None,
        event_candidate: EventCandidate | None = None,
    ) -> EvolutionUpdateSummary | None:
        if (
            decision == VerificationDecision.COMMIT
            and committed_event_id
            and state_diff_id
            and verification_result_id
        ):
            return _commit_evolution_summary(
                bundle=bundle,
                step_id=step_id,
                scenario_id=scenario_id,
                agent_id=agent_id,
                decision=decision,
                committed_event_id=committed_event_id,
                state_diff_id=state_diff_id,
                verification_result_id=verification_result_id,
                state_diff_applied=state_diff_applied,
                verification_result=verification_result,
                event_candidate=event_candidate,
            )
        pressure_updates = _non_commit_pressure_signals(
            bundle=bundle,
            step_id=step_id,
            scenario_id=scenario_id,
            decision=decision,
            event_candidate_id=event_candidate_id,
            verification_result=verification_result,
        )
        if not pressure_updates:
            return None
        return EvolutionUpdateSummary(
            step_id=step_id,
            scenario_id=scenario_id,
            decision=decision,
            applied_update_count=0,
            pressure_updates=pressure_updates,
            opportunity_route="trace_signal",
            opportunity_score=0.0,
            opportunity_source_ids=(event_candidate_id,) if event_candidate_id else (),
            canon_updated=False,
            world_state_updated=False,
        )


def _commit_evolution_summary(
    *,
    bundle: SeedBundle,
    step_id: str,
    scenario_id: str,
    agent_id: str,
    decision: VerificationDecision,
    committed_event_id: str,
    state_diff_id: str,
    verification_result_id: str,
    state_diff_applied: bool,
    verification_result: VerificationResult | None = None,
    event_candidate: EventCandidate | None = None,
) -> EvolutionUpdateSummary:
    quality = _verification_quality(verification_result, decision)
    belief_update = _belief_update(
        bundle,
        step_id,
        scenario_id,
        agent_id,
        committed_event_id,
        state_diff_id,
        quality,
    )
    memory_update = _memory_update(
        bundle,
        step_id,
        scenario_id,
        agent_id,
        committed_event_id,
        state_diff_id,
        quality,
    )
    relationship_update = _relationship_update(
        bundle,
        step_id,
        agent_id,
        committed_event_id,
        state_diff_id,
        quality,
        event_candidate,
    )
    opportunity_route, opportunity_value, pressure_updates = _commit_pressure_updates(
        bundle=bundle,
        step_id=step_id,
        decision=decision,
        committed_event_id=committed_event_id,
        state_diff_id=state_diff_id,
        verification_quality=quality,
    )
    applied_update_count = (
        len(pressure_updates)
        + int(belief_update is not None)
        + int(memory_update is not None)
        + int(relationship_update is not None)
    )
    return EvolutionUpdateSummary(
        step_id=step_id,
        scenario_id=scenario_id,
        decision=decision,
        applied_update_count=applied_update_count,
        causal_graph=_causal_graph(
            scenario_id=scenario_id,
            committed_event_id=committed_event_id,
            state_diff_id=state_diff_id,
            verification_result_id=verification_result_id,
            verification_quality=quality,
        ),
        pressure_updates=pressure_updates,
        belief_updates=(belief_update,) if belief_update is not None else (),
        memory_updates=(memory_update,) if memory_update is not None else (),
        relationship_updates=(relationship_update,) if relationship_update is not None else (),
        opportunity_route=opportunity_route,
        opportunity_score=opportunity_value,
        opportunity_source_ids=(committed_event_id, state_diff_id, verification_result_id),
        canon_updated=False,
        world_state_updated=state_diff_applied,
    )


def _causal_graph(
    *,
    scenario_id: str,
    committed_event_id: str,
    state_diff_id: str,
    verification_result_id: str,
    verification_quality: float,
) -> CausalEventGraphSummary:
    event_node_id = f"causal:event:{committed_event_id}"
    diff_node_id = f"causal:state_diff:{state_diff_id}"
    verification_node_id = f"causal:verification:{verification_result_id}"
    target_id, target_label = _causal_target(scenario_id)
    target_node_id = f"causal:resource:{target_id}"
    edge_confidence = causal_edge_confidence(
        temporal=1.0,
        entity_overlap=1.0 if target_id in state_diff_id else 0.7,
        pressure_linkage=verification_quality,
    )
    accepted_target_edges = (
        (
            CausalEventEdge(
                id=(
                    f"causal:edge:{state_diff_id}:affected:{target_id}:"
                    f"conf{round(edge_confidence * 100)}"
                ),
                source_node_id=diff_node_id,
                target_node_id=target_node_id,
                edge_type=CausalEdgeType.AFFECTED_TARGET,
                confidence=edge_confidence,
            ),
        )
        if edge_confidence >= 0.55
        else ()
    )
    return CausalEventGraphSummary(
        nodes=(
            CausalEventNode(
                id=event_node_id,
                node_type=CausalNodeType.COMMITTED_EVENT,
                source_id=committed_event_id,
                label=_causal_event_label(scenario_id),
            ),
            CausalEventNode(
                id=diff_node_id,
                node_type=CausalNodeType.STATE_DIFF,
                source_id=state_diff_id,
                label=f"{target_label} StateDiff",
            ),
            CausalEventNode(
                id=verification_node_id,
                node_type=CausalNodeType.VERIFICATION_RESULT,
                source_id=verification_result_id,
                label="Commit verification result",
            ),
            CausalEventNode(
                id=target_node_id,
                node_type=CausalNodeType.WORLD_TARGET,
                source_id=target_id,
                label=f"Affected {target_label} resource",
            ),
        ),
        edges=(
            CausalEventEdge(
                id=f"causal:edge:{verification_result_id}:verified:{committed_event_id}",
                source_node_id=verification_node_id,
                target_node_id=event_node_id,
                edge_type=CausalEdgeType.VERIFIED_BY,
                confidence=verification_quality,
            ),
            CausalEventEdge(
                id=f"causal:edge:{committed_event_id}:caused:{state_diff_id}",
                source_node_id=event_node_id,
                target_node_id=diff_node_id,
                edge_type=CausalEdgeType.CAUSED_STATE_DIFF,
                confidence=verification_quality,
            ),
            *accepted_target_edges,
        ),
    )


def _commit_pressure_updates(
    *,
    bundle: SeedBundle,
    step_id: str,
    decision: VerificationDecision,
    committed_event_id: str,
    state_diff_id: str,
    verification_quality: float,
) -> tuple[str, float, tuple[PressureUpdateSummary, ...]]:
    pressure_id = _commit_pressure_id(committed_event_id)
    seed = _pressure_seed(bundle, pressure_id)
    if seed is None:
        return "no_pressure_seed", 0.0, ()
    opportunity_value = _opportunity_value(seed.level, safety=verification_quality, goal=0.8)
    if not _opportunity_allows_update(seed.level, safety=verification_quality, goal=0.8):
        return "blocked", opportunity_value, ()
    update = _pressure_update(
        seed=seed,
        step_id=step_id,
        decision=decision,
        source_event_id=committed_event_id,
        source_state_diff_id=state_diff_id,
        target_impulse=0.35 + (0.45 * verification_quality),
        applied=True,
        governance_basis="commit_applied_state_diff",
        reason="EWMA pressure behavior used verified committed event impulse.",
    )
    return "record", opportunity_value, (update,)


def _non_commit_pressure_signals(
    *,
    bundle: SeedBundle,
    step_id: str,
    scenario_id: str,
    decision: VerificationDecision,
    event_candidate_id: str | None,
    verification_result: VerificationResult | None,
) -> tuple[PressureUpdateSummary, ...]:
    mapping = {
        "mira_search_archive_wrong_key": ("pressure_rumor_spread", -1),
        "niven_search_lantern_wrong_pass": ("pressure_gate_access", -1),
        "unsafe_force_open_safe": ("pressure_civic_trust", -1),
        "niven_force_quay_lock": ("pressure_gate_access", -1),
        "player_claim_key_in_hand": ("pressure_rumor_spread", 1),
        "player_claim_harbor_pass": ("pressure_gate_access", 1),
    }
    pressure_id_delta = mapping.get(scenario_id)
    if pressure_id_delta is None:
        return ()
    pressure_id, delta = pressure_id_delta
    seed = _pressure_seed(bundle, pressure_id)
    if seed is None:
        return ()
    quality = _verification_quality(verification_result, decision)
    safety = max(0.8 if delta < 0 else 0.55, quality)
    if not _opportunity_allows_update(seed.level, safety=safety, goal=0.5):
        return ()
    return (
        _pressure_update(
            seed=seed,
            step_id=step_id,
            decision=decision,
            source_candidate_id=event_candidate_id,
            target_impulse=(0.7 + quality * 0.2) if delta > 0 else max(0.15, quality * 0.35),
            applied=False,
            governance_basis="non_commit_trace_only",
            reason="Trace-only pressure signal from a non-commit governance path.",
        ),
    )


def _belief_update(
    bundle: SeedBundle,
    step_id: str,
    scenario_id: str,
    agent_id: str,
    committed_event_id: str,
    state_diff_id: str,
    verification_quality: float,
) -> BeliefUpdateSummary | None:
    belief = _first_belief_for_agent(bundle, agent_id)
    if belief is None:
        return None
    prior = _confidence_value(belief.confidence)
    contradiction_risk = 1.0 - verification_quality
    posterior = belief_confidence_score(
        prior=prior,
        source_reliability=max(0.35, verification_quality),
        verification_support=verification_quality,
        contradiction_risk=contradiction_risk,
        causal_support=max(0.35, verification_quality),
    )
    return BeliefUpdateSummary(
        id=f"belief_update:{step_id}:{agent_id}",
        owner_agent_id=belief.owner_agent_id,
        source_belief_id=belief.id,
        source_event_id=committed_event_id,
        source_state_diff_id=state_diff_id,
        truth_status_before=belief.truth_status,
        truth_status_after=BeliefTruthStatus.PARTIALLY_TRUE,
        confidence_before=belief.confidence,
        confidence_after=_confidence_band(posterior),
        canon_updated=False,
        governance_basis="commit_applied_state_diff",
        reason=(
            f"{_belief_update_reason(scenario_id)} belief_confidence_score={posterior:.2f}; "
            f"verification_quality={verification_quality:.2f}; "
            f"contradiction_risk={contradiction_risk:.2f}."
        ),
    )


def _memory_update(
    bundle: SeedBundle,
    step_id: str,
    scenario_id: str,
    agent_id: str,
    committed_event_id: str,
    state_diff_id: str,
    verification_quality: float,
) -> MemoryUpdateSummary:
    resource_ids, entity_ids = _memory_targets(scenario_id)
    existing = _top_memory_for_agent(bundle, agent_id)
    retained = (existing.salience / 5) if existing is not None else 0.45
    reinforcement = 0.15 + 0.35 * verification_quality
    suppression = 0.25 * (1.0 - verification_quality)
    strength = decay_reinforcement_score(
        retained=retained,
        reinforcement=reinforcement,
        suppression=suppression,
    )
    return MemoryUpdateSummary(
        id=f"memory_update:{step_id}:{agent_id}",
        owner_agent_id=agent_id,
        memory_kind=MemoryKind.OBSERVATION,
        source_event_id=committed_event_id,
        source_state_diff_id=state_diff_id,
        related_resource_ids=resource_ids,
        related_entity_ids=entity_ids,
        retained_strength=retained,
        reinforcement_delta=reinforcement,
        suppression_penalty=suppression,
        updated_strength=strength,
        summary=(
            f"{agent_id} remembers the verified outcome of {scenario_id}; "
            f"memory_strength={strength:.2f}."
        ),
        applied=True,
        governance_basis="commit_applied_state_diff",
        reason=(
            "Committed event creates a bounded observation memory signal. "
            f"retained={retained:.2f}; reinforcement={reinforcement:.2f}; "
            f"suppression={suppression:.2f}."
        ),
    )


def _relationship_update(
    bundle: SeedBundle,
    step_id: str,
    agent_id: str,
    committed_event_id: str,
    state_diff_id: str,
    verification_quality: float,
    event_candidate: EventCandidate | None,
) -> RelationshipUpdateSummary | None:
    relationship = _affected_relationship_for_event(bundle, agent_id, event_candidate)
    if relationship is None:
        return None
    conflict = 1.0 - verification_quality
    cooperation = verification_quality if "unsafe" not in committed_event_id else 0.25
    event_delta = cooperation - conflict
    trust_delta, trust_after = bounded_trust_update(
        trust_before=relationship.trust,
        event_delta=event_delta,
    )
    return RelationshipUpdateSummary(
        id=f"relationship_update:{step_id}:{relationship.id}",
        relationship_id=relationship.id,
        source_agent_id=relationship.source_agent_id,
        target_agent_id=relationship.target_agent_id,
        source_event_id=committed_event_id,
        source_state_diff_id=state_diff_id,
        trust_before=relationship.trust,
        trust_delta=trust_delta,
        trust_after=trust_after,
        applied=True,
        governance_basis="commit_applied_state_diff",
        reason=(
            f"Bounded relationship behavior used cooperation={cooperation:.2f}; "
            f"conflict={conflict:.2f}; event_delta={event_delta:.2f}."
        ),
    )


def _pressure_update(
    *,
    seed: PressureSeed,
    step_id: str,
    decision: VerificationDecision,
    target_impulse: float,
    applied: bool,
    reason: str,
    governance_basis: str,
    source_event_id: str | None = None,
    source_state_diff_id: str | None = None,
    source_candidate_id: str | None = None,
) -> PressureUpdateSummary:
    _, after_level = ewma_pressure_update(
        before_level=seed.level,
        event_impulse=target_impulse,
    )
    delta = after_level - seed.level
    return PressureUpdateSummary(
        id=f"pressure_update:{step_id}:{seed.pressure_type}",
        pressure_type=seed.pressure_type,
        source_step_id=step_id,
        source_event_id=source_event_id,
        source_state_diff_id=source_state_diff_id,
        source_candidate_id=source_candidate_id,
        decision=decision,
        location_id=seed.location_id,
        resource_id=seed.resource_id,
        before_level=seed.level,
        delta=delta,
        after_level=after_level,
        applied=applied,
        governance_basis=governance_basis,
        reason=f"{reason} target_impulse={target_impulse:.2f}.",
    )


def _confidence_value(confidence: ConfidenceBand) -> float:
    return {"low": 0.25, "medium": 0.55, "high": 0.85}.get(confidence.value, 0.5)


def _confidence_band(value: float) -> ConfidenceBand:
    if value >= 0.70:
        return ConfidenceBand.HIGH
    if value >= 0.40:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW


def _verification_quality(
    verification_result: VerificationResult | None,
    decision: VerificationDecision,
) -> float:
    if verification_result is None:
        return 0.85 if decision == VerificationDecision.COMMIT else 0.35
    total = max(len(verification_result.checks), 1)
    passed = sum(1 for check in verification_result.checks if check.passed)
    risk_penalty = min(len(verification_result.risk_flags) * 0.12, 0.6)
    decision_component = 1.0 if decision == VerificationDecision.COMMIT else 0.45
    return max(0.0, min(1.0, 0.65 * (passed / total) + 0.35 * decision_component - risk_penalty))


def _pressure_seed(bundle: SeedBundle, pressure_id: str) -> PressureSeed | None:
    if bundle.metadata is None:
        return None
    for seed in bundle.metadata.pressure_seeds:
        if seed.id == pressure_id:
            return seed
    return None


def _opportunity_allows_update(before_level: int, *, safety: float, goal: float) -> bool:
    return safety >= 0.5 and _opportunity_value(before_level, safety=safety, goal=goal) >= 0.35


def _opportunity_value(before_level: int, *, safety: float, goal: float) -> float:
    pressure_component = before_level / 10
    return opportunity_score(
        pressure_component=pressure_component,
        safety=safety,
        causal=0.75,
        goal=goal,
    )


def _causal_target(scenario_id: str) -> tuple[str, str]:
    if scenario_id == "elin_inspect_cargo_manifest_fixture":
        return "harbor_pass", "harbor pass discovery"
    if scenario_id == "selka_consume_stabilizer_part_fixture":
        return "stabilizer_parts", "stabilizer parts quantity"
    if scenario_id == "selka_restock_market_credit_fixture":
        return "market_credit", "market credit quantity"
    if scenario_id == "sora_release_relief_crates_fixture":
        return "relief_crates", "relief crates quantity"
    return "calibration_key", "calibration key discovery"


def _causal_event_label(scenario_id: str) -> str:
    if scenario_id == "elin_inspect_cargo_manifest_fixture":
        return "Committed harbor manifest inspection event"
    if scenario_id == "sora_release_relief_crates_fixture":
        return "Committed relief crate release event"
    return "Committed safe inspection event"


def _belief_update_reason(scenario_id: str) -> str:
    if scenario_id.startswith(("elin_", "sora_")):
        return "Committed Harbor event supports the actor's belief without writing a new CanonFact."
    return "Committed inspection supports Ivo's private belief without writing a new CanonFact."


def _commit_pressure_id(committed_event_id: str) -> str:
    if "elin_inspect_cargo_manifest_fixture" in committed_event_id:
        return "pressure_gate_access"
    if "sora_release_relief_crates_fixture" in committed_event_id:
        return "pressure_relief_delay"
    return "pressure_regulator_instability"


def _memory_targets(scenario_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if scenario_id == "elin_inspect_cargo_manifest_fixture":
        return ("harbor_pass",), ("cargo_manifest",)
    if scenario_id == "sora_release_relief_crates_fixture":
        return ("relief_crates",), ("quay_lock",)
    return ("calibration_key",), ("workshop_safe",)


def _first_belief_for_agent(bundle: SeedBundle, agent_id: str) -> BeliefRecord | None:
    for belief in bundle.beliefs.beliefs:
        if belief.owner_agent_id == agent_id:
            return belief
    return None


def _top_memory_for_agent(bundle: SeedBundle, agent_id: str) -> MemoryRecord | None:
    memories = tuple(
        memory for memory in bundle.memories.memories if memory.owner_agent_id == agent_id
    )
    if not memories:
        return None
    return max(memories, key=lambda memory: (memory.salience, bool(memory.source_event_id), memory.id))


def _affected_relationship_for_event(
    bundle: SeedBundle,
    agent_id: str,
    event_candidate: EventCandidate | None,
) -> RelationshipRecord | None:
    candidates = tuple(
        relationship
        for relationship in bundle.agents.relationships
        if agent_id in {relationship.source_agent_id, relationship.target_agent_id}
    )
    if not candidates:
        return None
    participant_ids = {agent_id}
    if event_candidate is not None:
        participant_ids.add(event_candidate.actor_agent_id)
        participant_ids.update(event_candidate.involved_entity_ids)
    direct = tuple(
        relationship
        for relationship in candidates
        if {relationship.source_agent_id, relationship.target_agent_id} & participant_ids
    )
    ranked = direct or candidates
    return sorted(
        ranked,
        key=lambda relationship: (-abs(relationship.trust), relationship.id),
    )[0]
