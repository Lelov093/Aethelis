from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aethelis.algorithms.mechanisms import (
    AlgorithmMechanismSummary,
    MechanismKind,
    load_algorithm_mechanism_config,
    run_algorithm_mechanism_experiment,
)
from aethelis.algorithms.runtime_features import (
    clamp01,
    decay_reinforcement_score,
    harmonic_governance_score,
    model_selection_score,
    opportunity_score,
    player_assimilation_score,
    proposal_confidence_score,
    retrieval_rank_score,
    weighted_scheduler_score,
)
from aethelis.runtime.single_step import SingleStepResult, StepContextSnapshot, build_step_context
from aethelis.schemas.evolution import EvolutionUpdateSummary
from aethelis.schemas.seed import SeedBundle
from aethelis.seeds.loader import SeedLoader

ALGORITHM_MODE_V11 = "v11_product05"
NOT_AVAILABLE = "not_available_from_current_runtime_object"
RUNTIME_MECHANISMS = (
    MechanismKind.AGENT_ACTIVATION,
    MechanismKind.OBSERVATION_SCOPE,
    MechanismKind.COGNITION_RETRIEVAL,
    MechanismKind.ACTION_PROPOSAL,
    MechanismKind.EVENT_CANDIDATE,
    MechanismKind.EVENT_VERIFICATION,
    MechanismKind.BELIEF_CONFIDENCE,
    MechanismKind.MEMORY_DECAY,
    MechanismKind.RELATIONSHIP_UPDATE,
    MechanismKind.WORLD_PRESSURE,
    MechanismKind.EVOLUTION_OPPORTUNITY,
    MechanismKind.CAUSAL_GRAPH,
    MechanismKind.PLAYER_INPUT,
    MechanismKind.EVALUATION_SCORING,
    MechanismKind.MODEL_COMBINATION,
)


@dataclass(frozen=True)
class RuntimeAlgorithmDecision:
    mechanism_id: str
    model_family: str
    formula: str
    score: float
    score_breakdown: dict[str, object]
    input_features: dict[str, object]
    decision: str
    runtime_object_type: str
    runtime_object_id: str


@dataclass(frozen=True)
class RuntimeModelingContext:
    bundle: SeedBundle
    step_context: StepContextSnapshot | None
    evolution_update: EvolutionUpdateSummary | None = None


def build_runtime_algorithm_decisions(
    *,
    seed_path: Path,
    result: SingleStepResult,
    config_path: Path | None,
    evolution_update: EvolutionUpdateSummary | None = None,
) -> tuple[RuntimeAlgorithmDecision, ...]:
    config = load_algorithm_mechanism_config(config_path)
    report = run_algorithm_mechanism_experiment(seed_path=seed_path, config=config)
    summaries = {summary.mechanism_id: summary for summary in report.summaries}
    modeling_context = _runtime_modeling_context(
        seed_path,
        result,
        evolution_update=evolution_update or result.evolution_update,
    )
    decisions: list[RuntimeAlgorithmDecision] = []
    for kind in RUNTIME_MECHANISMS:
        summary = summaries[kind]
        object_type, object_id = _runtime_link(kind, result, modeling_context)
        input_features = _input_features(kind, summary, result, modeling_context)
        score_breakdown = _score_breakdown(kind, summary, result, modeling_context, input_features)
        score = _score_from_breakdown(summary, score_breakdown)
        decisions.append(
            RuntimeAlgorithmDecision(
                mechanism_id=kind.value,
                model_family=summary.model_families[0],
                formula=_runtime_formula(kind, summary.formula),
                score=score,
                score_breakdown=score_breakdown,
                input_features=input_features,
                decision=_decision_label(kind, score, result),
                runtime_object_type=object_type,
                runtime_object_id=object_id,
            )
        )
    return tuple(decisions)


def _runtime_link(
    kind: MechanismKind,
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> tuple[str, str]:
    evolution = context.evolution_update
    if kind == MechanismKind.AGENT_ACTIVATION:
        return "agent", result.agent_id
    if kind == MechanismKind.OBSERVATION_SCOPE:
        return "observation_scope", result.agent_id
    if kind == MechanismKind.COGNITION_RETRIEVAL:
        return "retrieval_summary", result.agent_id
    if kind == MechanismKind.ACTION_PROPOSAL and result.action_proposal is not None:
        return "action_proposal", result.action_proposal.id
    if kind == MechanismKind.EVENT_CANDIDATE and result.event_candidate is not None:
        return "event_candidate", result.event_candidate.id
    if kind in {MechanismKind.EVENT_VERIFICATION, MechanismKind.BELIEF_CONFIDENCE}:
        return "verification_result", result.verification_result.id
    if kind == MechanismKind.MEMORY_DECAY:
        if evolution is not None and evolution.memory_updates:
            return "memory_update", evolution.memory_updates[0].id
        return "retrieval_summary", result.agent_id
    if kind == MechanismKind.RELATIONSHIP_UPDATE:
        if evolution is not None and evolution.relationship_updates:
            return "relationship_update", evolution.relationship_updates[0].id
        return "agent_relationships", result.agent_id
    if kind == MechanismKind.WORLD_PRESSURE and result.committed_event is not None:
        if evolution is not None and evolution.pressure_updates:
            return "pressure_update", evolution.pressure_updates[0].id
        return "state_diff", result.committed_event.state_diff.id
    if kind == MechanismKind.EVOLUTION_OPPORTUNITY:
        if evolution is not None and evolution.opportunity_route is not None:
            return "evolution_update", evolution.step_id
        return "world_evolution_opportunity", result.scenario_id
    if kind == MechanismKind.CAUSAL_GRAPH and result.committed_event is not None:
        edge = _first_evolution_edge(evolution)
        if edge is not None:
            return "causal_edge", edge.id
        return "committed_event", result.committed_event.id
    if kind == MechanismKind.MODEL_COMBINATION:
        return "algorithm_mode", ALGORITHM_MODE_V11
    return "run_step", f"{result.scenario_id}:{result.agent_id}"


def _decision_label(
    kind: MechanismKind,
    score: float,
    result: SingleStepResult,
) -> str:
    if kind == MechanismKind.EVENT_VERIFICATION and result.verification_result is not None:
        return f"support_{result.verification_result.decision.value}"
    if kind == MechanismKind.OBSERVATION_SCOPE:
        return "scope_guard_applied"
    if kind == MechanismKind.ACTION_PROPOSAL:
        return "ranked_accept" if score >= 0.55 else "ranked_hold"
    if kind == MechanismKind.EVENT_CANDIDATE:
        return "candidate_model_selected"
    if kind == MechanismKind.EVOLUTION_OPPORTUNITY:
        return "opportunity_recorded" if score >= 0.55 else "opportunity_hold"
    if kind == MechanismKind.PLAYER_INPUT:
        return "canon_guard_active"
    if kind == MechanismKind.EVALUATION_SCORING:
        return "score_pass" if score >= 0.55 else "score_fail"
    if kind == MechanismKind.MODEL_COMBINATION:
        return "runtime_model_family_selected"
    return "selected" if score >= 0.55 else "observed_low_score"


def _runtime_modeling_context(
    seed_path: Path,
    result: SingleStepResult,
    *,
    evolution_update: EvolutionUpdateSummary | None,
) -> RuntimeModelingContext:
    load_result = SeedLoader().load(seed_path)
    if load_result.bundle is None:
        raise ValueError("seed bundle unavailable for runtime algorithm modeling")
    step_context: StepContextSnapshot | None
    try:
        step_context = build_step_context(
            load_result.bundle,
            agent_id=result.agent_id,
            scenario_id=result.scenario_id,
        )
    except Exception:
        step_context = None
    return RuntimeModelingContext(
        bundle=load_result.bundle,
        step_context=step_context,
        evolution_update=evolution_update,
    )


def _input_features(
    kind: MechanismKind,
    summary: AlgorithmMechanismSummary,
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    features: dict[str, object] = dict(summary.trace)
    features.update(_common_runtime_features(result, context))
    features.update(_mechanism_features(kind, result, context))
    return features


def _score_breakdown(
    kind: MechanismKind,
    summary: AlgorithmMechanismSummary,
    result: SingleStepResult,
    context: RuntimeModelingContext,
    input_features: dict[str, object],
) -> dict[str, object]:
    trace = dict(input_features)
    runtime = _runtime_components(kind, result, context, trace)
    common = {
        "baseline_score": summary.baseline_score,
        "complex_score": summary.complex_score,
        "improvement": summary.improvement,
        "runtime_score": runtime.get("runtime_score", summary.complex_score),
        "trace": trace,
        "source_ids": trace.get("source_ids", ()),
        "output_object_ids": trace.get("output_object_ids", ()),
        "risk_flags": _mechanism_risk_flags(kind, result, runtime),
    }
    if kind == MechanismKind.OBSERVATION_SCOPE:
        return common | {
            "visible_scope_score": runtime["visible_scope_score"],
            "access_score": runtime["access_score"],
            "privacy_risk": runtime["privacy_risk"],
            "hidden_removed_count": runtime["hidden_removed_count"],
            "visible_object_ids": runtime["visible_object_ids"],
            "hidden_filter_guard": runtime["hidden_filter_guard"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
            "filtered_information_policy": "no_hidden_canon_or_private_belief_dump",
        }
    if kind == MechanismKind.AGENT_ACTIVATION:
        return common | {
            "goal_component": runtime["goal_component"],
            "pressure_component": runtime["pressure_component"],
            "relationship_component": runtime["relationship_component"],
            "memory_component": runtime["memory_component"],
            "causal_component": runtime["causal_component"],
            "risk_penalty": runtime["risk_penalty"],
            "background_status": runtime["background_status"],
            "threshold_passed": runtime["threshold_passed"],
        }
    if kind == MechanismKind.COGNITION_RETRIEVAL:
        return common | {
            "selected_memory_ids": runtime["selected_memory_ids"],
            "selected_belief_ids": runtime["selected_belief_ids"],
            "available_memory_ids": runtime["available_memory_ids"],
            "suppressed_memory_ids": runtime["suppressed_memory_ids"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
            "salience_component": runtime["salience_component"],
            "recency_component": runtime["recency_component"],
            "belief_confidence_component": runtime["belief_confidence_component"],
            "suppression_penalty": runtime["suppression_penalty"],
        }
    if kind == MechanismKind.ACTION_PROPOSAL:
        return common | {
            "goal_utility": runtime["goal_utility"],
            "pressure_alignment": runtime["pressure_alignment"],
            "feasibility_score": runtime["feasibility_score"],
            "governance_score": runtime["governance_score"],
            "risk_penalty": runtime["risk_penalty"],
            "proposal_confidence": runtime["proposal_confidence"],
            "proposal_behavior_score": runtime["proposal_behavior_score"],
            "proposal_behavior_decision": runtime["proposal_behavior_decision"],
            "proposal_behavior_risk_flags": runtime["proposal_behavior_risk_flags"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
        }
    if kind == MechanismKind.EVENT_CANDIDATE:
        return common | {
            "event_type_mapping_score": runtime["event_type_mapping_score"],
            "actor_score": runtime["actor_score"],
            "target_score": runtime["target_score"],
            "precondition_score": runtime["precondition_score"],
            "predicted_state_diff_score": runtime["predicted_state_diff_score"],
            "verification_requirement_score": runtime["verification_requirement_score"],
            "trace_completeness_score": runtime["trace_completeness_score"],
            "candidate_route": runtime["candidate_route"],
            "candidate_status": runtime["candidate_status"],
            "candidate_behavior_score": runtime["candidate_behavior_score"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
        }
    if kind == MechanismKind.EVENT_VERIFICATION:
        return common | {
            "hard_gate": runtime["hard_gate"],
            "check_pass_rate": runtime["check_pass_rate"],
            "risk_penalty": runtime["risk_penalty"],
            "posterior": runtime["posterior"],
            "decision_score": runtime["decision_score"],
            "verification_decision": runtime["verification_decision"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
            "rejected_claim_count": runtime["rejected_claim_count"],
        }
    if kind == MechanismKind.BELIEF_CONFIDENCE:
        return common | {
            "source_belief_id": runtime["source_belief_id"],
            "prior_confidence": runtime["prior_confidence"],
            "support_count": runtime["support_count"],
            "contradiction_count": runtime["contradiction_count"],
            "source_reliability": runtime["source_reliability"],
            "posterior_confidence": runtime["posterior_confidence"],
            "confidence_after": runtime["confidence_after"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
        }
    if kind == MechanismKind.MEMORY_DECAY:
        return common | {
            "memory_id": runtime["memory_id"],
            "decay_delta": runtime["decay_delta"],
            "reinforcement_delta": runtime["reinforcement_delta"],
            "suppression_penalty": runtime["suppression_penalty"],
            "retained_strength": runtime["retained_strength"],
            "updated_strength": runtime["updated_strength"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
        }
    if kind == MechanismKind.RELATIONSHIP_UPDATE:
        return common | {
            "relationship_id": runtime["relationship_id"],
            "event_delta": runtime["event_delta"],
            "cooperation_score": runtime["cooperation_score"],
            "conflict_score": runtime["conflict_score"],
            "trust_before": runtime["trust_before"],
            "trust_delta": runtime["trust_delta"],
            "trust_after": runtime["trust_after"],
            "asymmetric_direction": runtime["asymmetric_direction"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
        }
    if kind == MechanismKind.WORLD_PRESSURE:
        return common | {
            "pressure_type": runtime["pressure_type"],
            "before_level": runtime["before_level"],
            "event_impulse": runtime["event_impulse"],
            "ewma_component": runtime["ewma_component"],
            "after_level": runtime["after_level"],
            "threshold_triggered": runtime["threshold_triggered"],
            "pressure_saturation": runtime["pressure_saturation"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
        }
    if kind == MechanismKind.EVOLUTION_OPPORTUNITY:
        return common | {
            "opportunity_score": runtime["opportunity_score"],
            "pressure_component": runtime["pressure_component"],
            "safety_gate": runtime["safety_gate"],
            "causal_open_thread_component": runtime["causal_open_thread_component"],
            "goal_intersection_component": runtime["goal_intersection_component"],
            "defer_decision": runtime["defer_decision"],
            "opportunity_route": runtime["opportunity_route"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
        }
    if kind == MechanismKind.CAUSAL_GRAPH:
        return common | {
            "edge_id": runtime["edge_id"],
            "edge_type": runtime["edge_type"],
            "source_event_id": runtime["source_event_id"],
            "target_state_diff_id": runtime["target_state_diff_id"],
            "temporal_adjacency": runtime["temporal_adjacency"],
            "entity_overlap": runtime["entity_overlap"],
            "pressure_linkage": runtime["pressure_linkage"],
            "edge_confidence": runtime["edge_confidence"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
        }
    if kind == MechanismKind.PLAYER_INPUT:
        return common | {
            "player_input_id": runtime["player_input_id"],
            "input_type_score": runtime["input_type_score"],
            "permission_gate": runtime["permission_gate"],
            "claim_risk": runtime["claim_risk"],
            "canon_contamination_risk": runtime["canon_contamination_risk"],
            "verification_requirement": runtime["verification_requirement"],
            "assimilation_decision": runtime["assimilation_decision"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
        }
    if kind == MechanismKind.EVALUATION_SCORING:
        return common | {
            "state_consistency": runtime["state_consistency"],
            "canon_safety": runtime["canon_safety"],
            "knowledge_boundary": runtime["knowledge_boundary"],
            "event_validity": runtime["event_validity"],
            "causal_coherence": runtime["causal_coherence"],
            "world_pressure_alignment": runtime["world_pressure_alignment"],
            "player_impact": runtime["player_impact"],
            "trace_completeness": runtime["trace_completeness"],
            "penalty_total": runtime["penalty_total"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
        }
    if kind == MechanismKind.MODEL_COMBINATION:
        return common | {
            "selected_model_families": summary.model_families,
            "variant_ids": runtime["variant_ids"],
            "governance_score": runtime["governance_score"],
            "evidence_fit": runtime["evidence_fit"],
            "metric_gain": runtime["metric_gain"],
            "complexity_penalty": runtime["complexity_penalty"],
            "risk_penalty": runtime["risk_penalty"],
            "trace_completeness": runtime["trace_completeness"],
            "selection_score": runtime["selection_score"],
            "pareto_rank": runtime["pareto_rank"],
            "coverage_score": runtime["coverage_score"],
            "behavior_source": runtime["behavior_source"],
            "alignment_status": runtime["alignment_status"],
            "pareto_or_weighted_decision": "weighted_product05_runtime_selection",
        }
    return common


def _common_runtime_features(
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    output_ids = tuple(
        item
        for item in (
            result.action_proposal.id if result.action_proposal is not None else None,
            result.event_candidate.id if result.event_candidate is not None else None,
            result.verification_result.id if result.verification_result is not None else None,
            result.committed_event.id if result.committed_event is not None else None,
            result.committed_event.state_diff.id if result.committed_event is not None else None,
            result.player_claim.claim_id if result.player_claim is not None else None,
            (
                result.player_claim.belief_candidate.id
                if result.player_claim is not None
                and result.player_claim.belief_candidate is not None
                else None
            ),
        )
        if item is not None
    )
    return {
        "agent_id": result.agent_id,
        "scenario_id": result.scenario_id,
        "provider_called": result.provider_called,
        "fallback_used": result.fallback_used,
        "state_diff_applied": result.state_diff_applied,
        "verification_decision": (
            result.verification_result.decision.value
            if result.verification_result is not None
            else None
        ),
        "source_ids": _source_ids(result, context),
        "output_object_ids": output_ids,
        "risk_flags": (
            tuple(result.verification_result.risk_flags)
            if result.verification_result is not None
            else ()
        ),
    }


def _mechanism_features(
    kind: MechanismKind,
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    if kind == MechanismKind.AGENT_ACTIVATION:
        return {
            "goal_ids": _goal_ids(context),
            "relationship_ids": _relationship_ids(context),
            "memory_ids": _memory_ids(context),
            "pressure_ids": _pressure_ids(context),
        }
    if kind == MechanismKind.OBSERVATION_SCOPE:
        return {
            "visible_object_ids": _visible_object_ids(context),
            "owned_belief_ids": _belief_ids(context),
            "hidden_canon_count": _hidden_canon_count(context.bundle),
        }
    if kind in {MechanismKind.COGNITION_RETRIEVAL, MechanismKind.MEMORY_DECAY}:
        summary = result.retrieval_summary or {}
        return {
            "selected_memory_ids": _tuple(summary.get("selected_memory_ids")) or _memory_ids(context),
            "selected_belief_ids": _tuple(summary.get("selected_belief_ids")) or _belief_ids(context),
            "suppressed_memory_ids": _tuple(summary.get("suppressed_memory_ids")),
        }
    if kind == MechanismKind.RELATIONSHIP_UPDATE:
        return {"relationship_ids": _relationship_ids(context)}
    if kind == MechanismKind.CAUSAL_GRAPH:
        return {"causal_edge_ids": _causal_edge_ids(result)}
    if kind == MechanismKind.PLAYER_INPUT:
        return {"player_input_id": _player_input_id(result)}
    return {}


def _runtime_components(
    kind: MechanismKind,
    result: SingleStepResult,
    context: RuntimeModelingContext,
    trace: dict[str, object],
) -> dict[str, object]:
    if kind == MechanismKind.AGENT_ACTIVATION:
        return _activation_components(result, context)
    if kind == MechanismKind.OBSERVATION_SCOPE:
        return _observation_components(result, context)
    if kind == MechanismKind.COGNITION_RETRIEVAL:
        return _retrieval_components(result, context)
    if kind == MechanismKind.ACTION_PROPOSAL:
        return _proposal_components(result, context)
    if kind == MechanismKind.EVENT_CANDIDATE:
        return _candidate_components(result)
    if kind == MechanismKind.EVENT_VERIFICATION:
        return _verification_components(result)
    if kind == MechanismKind.BELIEF_CONFIDENCE:
        return _belief_components(result, context)
    if kind == MechanismKind.MEMORY_DECAY:
        return _memory_components(result, context)
    if kind == MechanismKind.RELATIONSHIP_UPDATE:
        return _relationship_components(result, context)
    if kind == MechanismKind.WORLD_PRESSURE:
        return _pressure_components(result, context)
    if kind == MechanismKind.EVOLUTION_OPPORTUNITY:
        return _opportunity_components(result, context)
    if kind == MechanismKind.CAUSAL_GRAPH:
        return _causal_components(result, context)
    if kind == MechanismKind.PLAYER_INPUT:
        return _player_input_components(result)
    if kind == MechanismKind.EVALUATION_SCORING:
        return _evaluation_components(result)
    if kind == MechanismKind.MODEL_COMBINATION:
        return _model_combination_components(result, trace)
    return {}


def _runtime_formula(kind: MechanismKind, default: str) -> str:
    formulas = {
        MechanismKind.AGENT_ACTIVATION: (
            "activation = weighted(goal, pressure, relationship, memory, causal) - risk"
        ),
        MechanismKind.OBSERVATION_SCOPE: "scope = visible_access * (1 - privacy_risk)",
        MechanismKind.COGNITION_RETRIEVAL: (
            "retrieval = salience + recency + belief_confidence - suppression"
        ),
        MechanismKind.ACTION_PROPOSAL: (
            "proposal = utility + feasibility + governance - risk"
        ),
        MechanismKind.EVENT_CANDIDATE: (
            "candidate = schema * precondition * state_diff_plausibility"
        ),
        MechanismKind.EVENT_VERIFICATION: "verification = hard_gate * weighted_checks - risk",
        MechanismKind.BELIEF_CONFIDENCE: (
            "belief = bayes(prior, source_reliability, contradiction)"
        ),
        MechanismKind.MEMORY_DECAY: "memory = exp_decay + reinforcement - suppression",
        MechanismKind.RELATIONSHIP_UPDATE: "trust' = clamp(trust + tanh(event_delta))",
        MechanismKind.WORLD_PRESSURE: "pressure' = ewma(previous, event_impulse)",
        MechanismKind.EVOLUTION_OPPORTUNITY: (
            "opportunity = safety_gate * weighted(pressure, causal, goal)"
        ),
        MechanismKind.CAUSAL_GRAPH: (
            "edge_confidence = temporal + entity_overlap + pressure_linkage"
        ),
        MechanismKind.PLAYER_INPUT: "assimilation = permission * (1 - claim_risk)",
        MechanismKind.EVALUATION_SCORING: "score = harmonic(governance, trace) - penalties",
        MechanismKind.MODEL_COMBINATION: (
            "selection = evidence_fit + metric_gain - complexity - risk"
        ),
    }
    return formulas.get(kind, default)


def _activation_components(
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    goal_component = _avg_goal_priority(context)
    pressure_component = _max_pressure(context)
    relationship_component = _relationship_score(context)
    memory_component = _ratio(len(_memory_ids(context)), 5)
    causal_component = 1.0 if result.committed_event is not None else 0.25
    risk_penalty = _risk_penalty(result)
    score = weighted_scheduler_score(
        goal=goal_component,
        pressure=pressure_component,
        relationship=relationship_component,
        memory=memory_component,
        causal=causal_component,
        risk=risk_penalty,
    )
    return {
        "goal_component": goal_component,
        "pressure_component": pressure_component,
        "relationship_component": relationship_component,
        "memory_component": memory_component,
        "causal_component": causal_component,
        "risk_penalty": risk_penalty,
        "background_status": "active" if score >= 0.55 else "background",
        "threshold_passed": score >= 0.55,
        "runtime_score": score,
    }


def _observation_components(
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    visible_ids = _observation_visible_ids(result.observation_summary) or _visible_object_ids(context)
    hidden_count = _hidden_canon_count(context.bundle)
    total_canon = max(len(context.bundle.world.canon_facts), 1)
    privacy_risk = _clamp01(hidden_count / total_canon)
    visible_scope_score = _clamp01(_ratio(len(visible_ids), 8) * (1 - privacy_risk))
    return {
        "visible_scope_score": visible_scope_score,
        "access_score": 1.0 if context.step_context is not None else 0.0,
        "privacy_risk": privacy_risk,
        "hidden_removed_count": hidden_count,
        "visible_object_ids": visible_ids,
        "hidden_filter_guard": 1.0 if hidden_count >= 0 else 0.0,
        "behavior_source": (
            "SingleStepResult.observation_summary"
            if result.observation_summary is not None
            else "diagnostic_context_fallback"
        ),
        "alignment_status": "exact_match" if result.observation_summary is not None else "alignment_limited",
        "runtime_score": visible_scope_score,
    }


def _retrieval_components(
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    summary = result.retrieval_summary or {}
    memory_ids = _tuple(summary.get("selected_memory_ids")) or _memory_ids(context)
    belief_ids = _tuple(summary.get("selected_belief_ids")) or _belief_ids(context)
    available_memory_ids = _tuple(summary.get("available_memory_ids")) or _memory_ids(context)
    suppressed_memory_ids = _tuple(summary.get("suppressed_memory_ids"))
    salience = _memory_salience(context)
    recency = 1.0 if memory_ids else 0.0
    confidence = _belief_confidence_score(context)
    suppression = _ratio(len(suppressed_memory_ids), max(len(available_memory_ids), 1))
    return {
        "selected_memory_ids": memory_ids,
        "selected_belief_ids": belief_ids,
        "available_memory_ids": available_memory_ids,
        "suppressed_memory_ids": suppressed_memory_ids,
        "behavior_source": "SingleStepResult.retrieval_summary",
        "alignment_status": "exact_match",
        "salience_component": salience,
        "recency_component": recency,
        "belief_confidence_component": confidence,
        "suppression_penalty": suppression,
        "runtime_score": retrieval_rank_score(
            salience=salience,
            recency=recency,
            belief_confidence=confidence,
            suppression=suppression,
        ),
    }


def _proposal_components(
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    proposal = result.action_proposal
    has_target = proposal is not None and (
        proposal.target_location_id is not None or bool(proposal.target_entity_ids)
    )
    goal_utility = _avg_goal_priority(context)
    pressure_alignment = _max_pressure(context)
    feasibility = 1.0 if has_target else 0.0
    governance = 1.0 if result.event_candidate is not None else 0.0
    risk = _risk_penalty(result)
    confidence = proposal_confidence_score(
        goal_utility=goal_utility,
        pressure_alignment=pressure_alignment,
        feasibility=feasibility,
        governance=governance,
        risk=risk,
    )
    behavior_score = (
        result.proposal_behavior_score
        if result.proposal_behavior_score is not None
        else confidence
    )
    return {
        "goal_utility": goal_utility,
        "pressure_alignment": pressure_alignment,
        "feasibility_score": feasibility,
        "governance_score": governance,
        "risk_penalty": risk,
        "proposal_confidence": confidence,
        "proposal_behavior_score": behavior_score,
        "proposal_behavior_decision": result.proposal_behavior_decision or "not_available",
        "proposal_behavior_risk_flags": result.proposal_behavior_risk_flags,
        "behavior_source": "SingleStepResult.proposal_behavior_*",
        "alignment_status": "exact_match" if result.proposal_behavior_score is not None else "alignment_limited",
        "runtime_score": behavior_score,
    }


def _candidate_components(result: SingleStepResult) -> dict[str, object]:
    candidate = result.event_candidate
    actor = 1.0 if candidate is not None and candidate.actor_agent_id else 0.0
    target = 1.0 if candidate is not None and candidate.involved_entity_ids else 0.5
    location = 1.0 if candidate is not None and candidate.involved_location_ids else 0.5
    state_diff = 1.0 if result.committed_event is not None else 0.4
    verification = 1.0 if result.verification_result is not None else 0.0
    score = (
        result.candidate_behavior_score
        if result.candidate_behavior_score is not None
        else _clamp01((actor + target + location + state_diff + verification) / 5)
    )
    return {
        "event_type_mapping_score": actor,
        "actor_score": actor,
        "target_score": target,
        "precondition_score": verification,
        "predicted_state_diff_score": state_diff,
        "verification_requirement_score": verification,
        "trace_completeness_score": score,
        "candidate_route": result.candidate_behavior_route or "not_available",
        "candidate_status": candidate.status.value if candidate is not None else "missing",
        "candidate_behavior_score": score,
        "behavior_source": "SingleStepResult.candidate_behavior_*",
        "alignment_status": "exact_match" if result.candidate_behavior_score is not None else "alignment_limited",
        "runtime_score": score,
    }


def _verification_components(result: SingleStepResult) -> dict[str, object]:
    verification = result.verification_result
    checks = verification.checks if verification is not None else ()
    pass_rate = _ratio(sum(1 for check in checks if check.passed), max(len(checks), 1))
    hard_gate = 1.0 if verification is not None and verification.decision.value == "commit" else 0.0
    risk = _ratio(len(verification.risk_flags) if verification is not None else 1, 5)
    posterior = _clamp01(0.65 * pass_rate + 0.35 * hard_gate - 0.25 * risk)
    return {
        "hard_gate": hard_gate,
        "check_pass_rate": pass_rate,
        "risk_penalty": risk,
        "posterior": posterior,
        "decision_score": posterior,
        "verification_decision": verification.decision.value if verification is not None else "missing",
        "behavior_source": "VerificationResult.decision",
        "alignment_status": "exact_match" if verification is not None else "alignment_limited",
        "rejected_claim_count": len(verification.rejected_claim_ids) if verification else 0,
        "runtime_score": posterior,
    }


def _belief_components(
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    update = _first_belief_update(context.evolution_update)
    if update is not None:
        prior = _confidence_band_score(update.confidence_before.value)
        posterior = _confidence_band_score(update.confidence_after.value)
        return {
            "source_belief_id": update.source_belief_id,
            "prior_confidence": prior,
            "support_count": 1 if update.source_event_id else 0,
            "contradiction_count": 0 if context.evolution_update.decision.value == "commit" else 1,
            "source_reliability": 1.0 - _risk_penalty(result),
            "posterior_confidence": posterior,
            "confidence_after": update.confidence_after.value,
            "behavior_source": "EvolutionUpdateSummary.belief_updates[0]",
            "alignment_status": "exact_match",
            "runtime_score": posterior,
        }
    belief_ids = _belief_ids(context)
    prior = _belief_confidence_score(context)
    support = 1 if result.committed_event is not None else 0
    contradiction = 1 if result.verification_result is not None and result.verification_result.decision.value == "reject" else 0
    reliability = 1.0 - _risk_penalty(result)
    posterior = _clamp01((prior + support + reliability) / (2 + contradiction + 1))
    return {
        "source_belief_id": belief_ids[0] if belief_ids else NOT_AVAILABLE,
        "prior_confidence": prior,
        "support_count": support,
        "contradiction_count": contradiction,
        "source_reliability": reliability,
        "posterior_confidence": posterior,
        "confidence_after": "not_available",
        "behavior_source": "diagnostic_context_fallback",
        "alignment_status": "alignment_limited",
        "runtime_score": posterior,
    }


def _memory_components(
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    update = _first_memory_update(context.evolution_update)
    if update is not None:
        retained = update.retained_strength if update.retained_strength is not None else 0.0
        reinforcement = update.reinforcement_delta if update.reinforcement_delta is not None else 0.0
        suppression = update.suppression_penalty if update.suppression_penalty is not None else 0.0
        updated = update.updated_strength if update.updated_strength is not None else retained
        return {
            "memory_id": update.id,
            "decay_delta": _clamp01(1 - retained),
            "reinforcement_delta": reinforcement,
            "suppression_penalty": suppression,
            "retained_strength": retained,
            "updated_strength": updated,
            "behavior_source": "EvolutionUpdateSummary.memory_updates[0]",
            "alignment_status": "exact_match",
            "runtime_score": updated,
        }
    memory_ids = _memory_ids(context)
    retained = _memory_salience(context)
    reinforcement = 0.35 if result.committed_event is not None else 0.0
    suppression = 0.20 if _risk_penalty(result) > 0 else 0.0
    updated = decay_reinforcement_score(
        retained=retained,
        reinforcement=reinforcement,
        suppression=suppression,
    )
    return {
        "memory_id": memory_ids[0] if memory_ids else NOT_AVAILABLE,
        "decay_delta": _clamp01(1 - retained),
        "reinforcement_delta": reinforcement,
        "suppression_penalty": suppression,
        "retained_strength": retained,
        "updated_strength": updated,
        "behavior_source": "diagnostic_context_fallback",
        "alignment_status": "alignment_limited",
        "runtime_score": updated,
    }


def _relationship_components(
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    update = _first_relationship_update(context.evolution_update)
    if update is None:
        return _alignment_limited_relationship()
    trust_before = update.trust_before
    trust_delta = update.trust_delta
    trust_after = update.trust_after
    event_delta = _clamp01((trust_delta + 5) / 10)
    return {
        "relationship_id": update.relationship_id,
        "event_delta": event_delta,
        "cooperation_score": max(trust_delta, 0),
        "conflict_score": abs(min(trust_delta, 0)),
        "trust_before": trust_before,
        "trust_delta": trust_delta,
        "trust_after": trust_after,
        "asymmetric_direction": f"{update.source_agent_id}->{update.target_agent_id}",
        "behavior_source": "EvolutionUpdateSummary.relationship_updates[0]",
        "alignment_status": "exact_match",
        "runtime_score": _clamp01((trust_after + 5) / 10),
    }


def _pressure_components(
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    update = _first_pressure_update(context.evolution_update)
    if update is None:
        return _alignment_limited_pressure()
    before = update.before_level
    after = update.after_level
    impulse = _clamp01((after - before + 10) / 20)
    ewma = _clamp01(after / 10)
    return {
        "pressure_type": update.pressure_type,
        "before_level": before,
        "event_impulse": impulse,
        "ewma_component": ewma,
        "after_level": after,
        "threshold_triggered": after >= 6,
        "pressure_saturation": after in {0, 10},
        "behavior_source": "EvolutionUpdateSummary.pressure_updates[0]",
        "alignment_status": "exact_match",
        "runtime_score": ewma,
    }


def _opportunity_components(
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    evolution = context.evolution_update
    pressure = _pressure_components(result, context)
    safety = 1.0 - _risk_penalty(result)
    causal = 1.0 if _first_evolution_edge(evolution) is not None else 0.25
    goal = _avg_goal_priority(context)
    opportunity = evolution.opportunity_score if evolution is not None else None
    if opportunity is None:
        opportunity = opportunity_score(
            pressure_component=float(pressure["ewma_component"]),
            safety=safety,
            causal=causal,
            goal=goal,
        )
    return {
        "opportunity_score": opportunity,
        "pressure_component": pressure["ewma_component"],
        "safety_gate": safety >= 0.5,
        "causal_open_thread_component": causal,
        "goal_intersection_component": goal,
        "defer_decision": opportunity < 0.55,
        "opportunity_route": (
            evolution.opportunity_route if evolution is not None else "diagnostic"
        ),
        "behavior_source": "EvolutionUpdateSummary.opportunity_*" if evolution is not None else "diagnostic_context_fallback",
        "alignment_status": "exact_match" if evolution is not None else "alignment_limited",
        "runtime_score": opportunity,
    }


def _causal_components(
    result: SingleStepResult,
    context: RuntimeModelingContext,
) -> dict[str, object]:
    edge = _first_evolution_edge(context.evolution_update)
    if edge is None:
        return _alignment_limited_causal()
    temporal = 1.0
    entity_overlap = 1.0 if result.event_candidate and result.event_candidate.involved_entity_ids else 0.0
    pressure_link = edge.confidence
    confidence = edge.confidence
    return {
        "edge_id": edge.id,
        "edge_type": edge.edge_type.value,
        "source_event_id": result.committed_event.id if result.committed_event else NOT_AVAILABLE,
        "target_state_diff_id": (
            result.committed_event.state_diff.id if result.committed_event else NOT_AVAILABLE
        ),
        "temporal_adjacency": temporal,
        "entity_overlap": entity_overlap,
        "pressure_linkage": pressure_link,
        "edge_confidence": confidence,
        "behavior_source": "EvolutionUpdateSummary.causal_graph.edges[0]",
        "alignment_status": "exact_match",
        "runtime_score": confidence,
    }


def _player_input_components(result: SingleStepResult) -> dict[str, object]:
    summary = result.player_input_summary or {}
    player_input_id = _player_input_id(result)
    claim_risk = 1.0 if summary.get("canon_updated") is True else 0.0
    if result.player_claim is not None:
        claim_risk = 0.85
    permission = 1.0 if summary.get("world_state_modified") is not True else 0.0
    verification = 1.0 if result.verification_result is not None else 0.0
    assimilation = player_assimilation_score(
        permission_gate=permission,
        claim_risk=claim_risk,
        verification_requirement=verification,
    )
    return {
        "player_input_id": player_input_id or NOT_AVAILABLE,
        "input_type_score": 1.0 if player_input_id else 0.0,
        "permission_gate": permission,
        "claim_risk": claim_risk,
        "canon_contamination_risk": claim_risk,
        "verification_requirement": verification,
        "assimilation_decision": (
            summary.get("route") or result.verification_result.decision.value
            if result.verification_result is not None
            else "not_applicable"
        ),
        "behavior_source": "SingleStepResult.player_input_summary",
        "alignment_status": "exact_match" if summary else "alignment_limited",
        "runtime_score": assimilation if player_input_id else 0.5,
    }


def _evaluation_components(result: SingleStepResult) -> dict[str, object]:
    state_consistency = 1.0 if result.committed_event is None or result.state_diff_applied else 0.0
    canon_safety = 0.0 if result.player_claim is not None and result.player_claim.canon_updated else 1.0
    knowledge = 1.0 if (result.retrieval_summary or {}).get("hidden_context_used") is not True else 0.0
    event_validity = 1.0 if result.event_candidate is not None else 0.0
    causal = 1.0 if result.committed_event is None or result.committed_event.state_diff is not None else 0.0
    pressure = 1.0 if result.verification_result is not None else 0.0
    player = 1.0 if result.player_input_summary is None or result.player_input_summary.get("canon_updated") is False else 0.0
    trace = _clamp01(
        sum(
            1
            for item in (
                result.action_proposal,
                result.event_candidate,
                result.verification_result,
                result.committed_event,
            )
            if item is not None
        )
        / 4
    )
    penalty = _clamp01((1 - canon_safety) + (1 - knowledge) + _risk_penalty(result))
    score = harmonic_governance_score(
        state_consistency=state_consistency,
        canon_safety=canon_safety,
        event_validity=event_validity,
        trace_completeness=trace,
        penalty=penalty,
    )
    return {
        "state_consistency": state_consistency,
        "canon_safety": canon_safety,
        "knowledge_boundary": knowledge,
        "event_validity": event_validity,
        "causal_coherence": causal,
        "world_pressure_alignment": pressure,
        "player_impact": player,
        "trace_completeness": trace,
        "penalty_total": penalty,
        "behavior_source": "diagnostic_single_step_metrics",
        "alignment_status": "diagnostic_match",
        "runtime_score": score,
    }


def _model_combination_components(
    result: SingleStepResult,
    trace: dict[str, object],
) -> dict[str, object]:
    governance = 1.0 - _risk_penalty(result)
    evidence_fit = _clamp01(float(trace.get("evidence_fit", governance) or governance))
    metric_gain = 1.0 if result.state_diff_applied else 0.35
    complexity = _clamp01(float(trace.get("complexity_penalty", 0.15) or 0.15))
    risk = _risk_penalty(result)
    trace_score = _evaluation_components(result)["trace_completeness"]
    selection = model_selection_score(
        governance=governance,
        evidence_fit=evidence_fit,
        metric_gain=metric_gain,
        trace_completeness=float(trace_score),
        complexity=complexity,
        risk=risk,
    )
    return {
        "variant_ids": ("v11_product05_runtime",),
        "governance_score": governance,
        "evidence_fit": evidence_fit,
        "metric_gain": metric_gain,
        "complexity_penalty": complexity,
        "risk_penalty": risk,
        "trace_completeness": trace_score,
        "selection_score": selection,
        "pareto_rank": 1,
        "coverage_score": 1.0,
        "behavior_source": "diagnostic_runtime_selector",
        "alignment_status": "diagnostic_match",
        "runtime_score": selection,
    }


def _first_belief_update(evolution: EvolutionUpdateSummary | None):
    return evolution.belief_updates[0] if evolution is not None and evolution.belief_updates else None


def _first_memory_update(evolution: EvolutionUpdateSummary | None):
    return evolution.memory_updates[0] if evolution is not None and evolution.memory_updates else None


def _first_relationship_update(evolution: EvolutionUpdateSummary | None):
    return (
        evolution.relationship_updates[0]
        if evolution is not None and evolution.relationship_updates
        else None
    )


def _first_pressure_update(evolution: EvolutionUpdateSummary | None):
    return evolution.pressure_updates[0] if evolution is not None and evolution.pressure_updates else None


def _first_evolution_edge(evolution: EvolutionUpdateSummary | None):
    if evolution is None or evolution.causal_graph is None or not evolution.causal_graph.edges:
        return None
    return evolution.causal_graph.edges[0]


def _alignment_limited_relationship() -> dict[str, object]:
    return {
        "relationship_id": NOT_AVAILABLE,
        "event_delta": 0.0,
        "cooperation_score": 0.0,
        "conflict_score": 0.0,
        "trust_before": 0,
        "trust_delta": 0,
        "trust_after": 0,
        "asymmetric_direction": NOT_AVAILABLE,
        "behavior_source": "EvolutionUpdateSummary.relationship_updates",
        "alignment_status": "alignment_limited",
        "runtime_score": 0.5,
    }


def _alignment_limited_pressure() -> dict[str, object]:
    return {
        "pressure_type": NOT_AVAILABLE,
        "before_level": 0,
        "event_impulse": 0.0,
        "ewma_component": 0.0,
        "after_level": 0,
        "threshold_triggered": False,
        "pressure_saturation": False,
        "behavior_source": "EvolutionUpdateSummary.pressure_updates",
        "alignment_status": "alignment_limited",
        "runtime_score": 0.0,
    }


def _alignment_limited_causal() -> dict[str, object]:
    return {
        "edge_id": NOT_AVAILABLE,
        "edge_type": NOT_AVAILABLE,
        "source_event_id": NOT_AVAILABLE,
        "target_state_diff_id": NOT_AVAILABLE,
        "temporal_adjacency": 0.0,
        "entity_overlap": 0.0,
        "pressure_linkage": 0.0,
        "edge_confidence": 0.0,
        "behavior_source": "EvolutionUpdateSummary.causal_graph.edges",
        "alignment_status": "alignment_limited",
        "runtime_score": 0.0,
    }


def _score_from_breakdown(
    summary: AlgorithmMechanismSummary,
    score_breakdown: dict[str, object],
) -> float:
    runtime_score = score_breakdown.get("runtime_score")
    if isinstance(runtime_score, int | float):
        return _clamp01(float(runtime_score))
    return summary.complex_score


def _mechanism_risk_flags(
    kind: MechanismKind,
    result: SingleStepResult,
    runtime: dict[str, object],
) -> tuple[str, ...]:
    flags = list(result.verification_result.risk_flags) if result.verification_result else []
    if kind == MechanismKind.OBSERVATION_SCOPE and runtime.get("hidden_removed_count", 0):
        flags.append("hidden_canon_filtered")
    if kind == MechanismKind.EVENT_VERIFICATION and result.verification_result is None:
        flags.append("verification_result_not_available")
    if kind == MechanismKind.PLAYER_INPUT:
        if result.player_claim is not None:
            flags.append("player_claim_not_canon")
        if runtime.get("permission_gate") == 0.0:
            flags.append("player_input_permission_gate_closed")
    if kind == MechanismKind.MODEL_COMBINATION and runtime.get("complexity_penalty", 0) > 0:
        flags.append("complexity_penalty_applied")
    if kind == MechanismKind.ACTION_PROPOSAL and runtime.get("risk_penalty", 0) > 0:
        flags.append("proposal_risk_penalty_applied")
    if kind == MechanismKind.EVALUATION_SCORING and runtime.get("penalty_total", 0) > 0:
        flags.append("evaluation_penalty_applied")
    return tuple(dict.fromkeys(flags))


def _source_ids(result: SingleStepResult, context: RuntimeModelingContext) -> tuple[str, ...]:
    ids: list[str] = [result.scenario_id, result.agent_id]
    ids.extend(_goal_ids(context))
    ids.extend(_belief_ids(context))
    ids.extend(_memory_ids(context))
    ids.extend(_relationship_ids(context))
    ids.extend(_pressure_ids(context))
    ids.extend(_causal_edge_ids(result))
    player_input_id = _player_input_id(result)
    if player_input_id:
        ids.append(player_input_id)
    return tuple(dict.fromkeys(ids))


def _goal_ids(context: RuntimeModelingContext) -> tuple[str, ...]:
    cognition = context.step_context.cognition if context.step_context is not None else None
    if cognition is None:
        return ()
    return tuple(goal.id for goal in cognition.agent.cognitive_state.goals)


def _belief_ids(context: RuntimeModelingContext) -> tuple[str, ...]:
    cognition = context.step_context.cognition if context.step_context is not None else None
    if cognition is None:
        return ()
    return tuple(belief.id for belief in cognition.owned_beliefs)


def _memory_ids(context: RuntimeModelingContext) -> tuple[str, ...]:
    cognition = context.step_context.cognition if context.step_context is not None else None
    if cognition is None:
        return ()
    return tuple(memory.id for memory in cognition.owned_memories)


def _relationship_ids(context: RuntimeModelingContext) -> tuple[str, ...]:
    return tuple(relationship.id for relationship in _relationships(context))


def _relationships(context: RuntimeModelingContext):
    agent_id = (
        context.step_context.cognition.agent.id if context.step_context is not None else None
    )
    if agent_id is None:
        return ()
    return tuple(
        relationship
        for relationship in context.bundle.agents.relationships
        if agent_id in {relationship.source_agent_id, relationship.target_agent_id}
    )


def _pressure_ids(context: RuntimeModelingContext) -> tuple[str, ...]:
    if context.bundle.metadata is None:
        return ()
    return tuple(pressure.id for pressure in context.bundle.metadata.pressure_seeds)


def _visible_object_ids(context: RuntimeModelingContext) -> tuple[str, ...]:
    if context.step_context is None:
        return ()
    observation = context.step_context.observation
    return tuple(
        dict.fromkeys(
            [
                observation.location.id,
                *(entity.id for entity in observation.visible_entities),
                *(resource.id for resource in observation.visible_resources),
                *observation.visible_agent_ids,
                *(fact.id for fact in observation.visible_public_facts),
                *(rumor.id for rumor in observation.visible_rumors),
            ]
        )
    )


def _observation_visible_ids(summary: dict[str, object] | None) -> tuple[str, ...]:
    if not summary:
        return ()
    ids: list[str] = []
    location_id = summary.get("location_id")
    if location_id:
        ids.append(str(location_id))
    for key in (
        "visible_entity_ids",
        "visible_resource_ids",
        "visible_agent_ids",
        "visible_public_fact_ids",
        "visible_rumor_ids",
    ):
        ids.extend(_tuple(summary.get(key)))
    return tuple(dict.fromkeys(ids))


def _causal_edge_ids(result: SingleStepResult) -> tuple[str, ...]:
    if result.committed_event is None:
        return ()
    state_diff_id = result.committed_event.state_diff.id
    return (
        f"causal:edge:{result.verification_result.id}:verified:{result.committed_event.id}"
        if result.verification_result is not None
        else f"causal:edge:{result.committed_event.id}:verified",
        f"causal:edge:{result.committed_event.id}:caused:{state_diff_id}",
    )


def _player_input_id(result: SingleStepResult) -> str | None:
    if result.player_claim is not None:
        return result.player_claim.claim_id
    summary = result.player_input_summary or {}
    value = summary.get("input_id") or summary.get("belief_candidate_id")
    return str(value) if value is not None else None


def _avg_goal_priority(context: RuntimeModelingContext) -> float:
    cognition = context.step_context.cognition if context.step_context is not None else None
    if cognition is None:
        return 0.0
    priorities = [goal.priority / 5 for goal in cognition.agent.cognitive_state.goals]
    return _avg(priorities)


def _max_pressure(context: RuntimeModelingContext) -> float:
    if context.bundle.metadata is None:
        return 0.0
    return _clamp01(max((pressure.level for pressure in context.bundle.metadata.pressure_seeds), default=0) / 10)


def _relationship_score(context: RuntimeModelingContext) -> float:
    relationships = _relationships(context)
    if not relationships:
        return 0.5
    return _clamp01(_avg([(relationship.trust + 5) / 10 for relationship in relationships]))


def _memory_salience(context: RuntimeModelingContext) -> float:
    cognition = context.step_context.cognition if context.step_context is not None else None
    if cognition is None or not cognition.owned_memories:
        return 0.0
    return _avg([memory.salience for memory in cognition.owned_memories])


def _belief_confidence_score(context: RuntimeModelingContext) -> float:
    cognition = context.step_context.cognition if context.step_context is not None else None
    if cognition is None or not cognition.owned_beliefs:
        return 0.5
    mapping = {"low": 0.25, "medium": 0.55, "high": 0.85}
    return _avg([mapping.get(belief.confidence.value, 0.5) for belief in cognition.owned_beliefs])


def _hidden_canon_count(bundle: SeedBundle) -> int:
    return sum(1 for fact in bundle.world.canon_facts if fact.visibility == "hidden_canon")


def _pressure_seed(context: RuntimeModelingContext):
    if context.bundle.metadata is None:
        return None
    return next(iter(context.bundle.metadata.pressure_seeds), None)


def _risk_penalty(result: SingleStepResult) -> float:
    if result.verification_result is None:
        return 1.0 if result.failure_code is not None else 0.0
    return _clamp01(len(result.verification_result.risk_flags) / 5)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(value: int, denominator: int) -> float:
    return _clamp01(value / max(denominator, 1))


def _tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _confidence_band_score(value: str) -> float:
    return {"low": 0.25, "medium": 0.55, "high": 0.85}.get(value, 0.5)


def _harmonic(*values: float) -> float:
    clean = tuple(max(_clamp01(value), 1e-9) for value in values)
    return len(clean) / sum(1 / value for value in clean)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
