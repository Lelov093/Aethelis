from __future__ import annotations


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def weighted_scheduler_score(
    *,
    goal: float,
    pressure: float,
    relationship: float,
    memory: float,
    causal: float,
    risk: float,
) -> float:
    return clamp01(
        0.28 * goal
        + 0.22 * pressure
        + 0.18 * relationship
        + 0.16 * memory
        + 0.16 * causal
        - 0.20 * risk
    )


def observation_context_score(
    *,
    visibility: float,
    access: float,
    public_relevance: float,
    agent_relevance: float,
    privacy_risk: float,
) -> float:
    return clamp01(
        0.35 * visibility
        + 0.25 * access
        + 0.20 * public_relevance
        + 0.10 * agent_relevance
        - 0.10 * privacy_risk
    )


def retrieval_rank_score(
    *,
    salience: float,
    recency: float,
    belief_confidence: float,
    suppression: float,
) -> float:
    return clamp01(0.4 * salience + 0.2 * recency + 0.4 * belief_confidence - suppression)


def action_proposal_behavior_score(
    *,
    utility: float,
    feasibility: float,
    governance: float,
    risk: float,
) -> float:
    return clamp01(0.30 * utility + 0.25 * feasibility + 0.25 * governance - 0.20 * risk)


def proposal_confidence_score(
    *,
    goal_utility: float,
    pressure_alignment: float,
    feasibility: float,
    governance: float,
    risk: float,
) -> float:
    return clamp01(
        0.3 * goal_utility
        + 0.2 * pressure_alignment
        + 0.3 * feasibility
        + 0.2 * governance
        - 0.2 * risk
    )


def candidate_quality_score(
    *,
    schema_completeness: float,
    precondition_fit: float,
    state_diff_plausibility: float,
    boundary_risk: float,
) -> float:
    return clamp01(
        0.35 * schema_completeness
        + 0.25 * precondition_fit
        + 0.25 * state_diff_plausibility
        - 0.15 * boundary_risk
    )


def verification_behavior_score(
    *,
    hard_gate: float,
    check_pass_rate: float,
    evidence_support: float,
    causal_coherence: float,
    state_safety: float,
    contradiction_risk: float,
) -> float:
    return clamp01(
        0.35 * hard_gate
        + 0.20 * check_pass_rate
        + 0.18 * evidence_support
        + 0.12 * causal_coherence
        + 0.10 * state_safety
        - 0.15 * contradiction_risk
    )


def belief_confidence_score(
    *,
    prior: float,
    source_reliability: float,
    verification_support: float,
    contradiction_risk: float,
    causal_support: float,
) -> float:
    return clamp01(
        0.20 * prior
        + 0.25 * source_reliability
        + 0.30 * verification_support
        + 0.15 * causal_support
        - 0.20 * contradiction_risk
    )


def decay_reinforcement_score(
    *,
    retained: float,
    reinforcement: float,
    suppression: float,
) -> float:
    return clamp01(retained + reinforcement - suppression)


def bounded_trust_update(*, trust_before: int, event_delta: float) -> tuple[int, int]:
    trust_delta = max(-2, min(2, round(2 * event_delta)))
    return trust_delta, max(-5, min(5, trust_before + trust_delta))


def ewma_pressure_update(
    *,
    before_level: int,
    event_impulse: float,
    alpha: float = 0.48,
) -> tuple[float, int]:
    component = clamp01(alpha * event_impulse + (1 - alpha) * (before_level / 10))
    return component, round(10 * component)


def opportunity_score(
    *,
    pressure_component: float,
    safety: float,
    causal: float,
    goal: float,
) -> float:
    return clamp01(0.35 * pressure_component + 0.25 * safety + 0.20 * causal + 0.20 * goal)


def causal_edge_confidence(
    *,
    temporal: float,
    entity_overlap: float,
    pressure_linkage: float,
) -> float:
    return clamp01(0.45 * temporal + 0.35 * entity_overlap + 0.20 * pressure_linkage)


def player_assimilation_score(
    *,
    permission_gate: float,
    claim_risk: float,
    verification_requirement: float,
) -> float:
    return clamp01(permission_gate * (1 - claim_risk) * verification_requirement)


def harmonic_governance_score(
    *,
    state_consistency: float,
    canon_safety: float,
    event_validity: float,
    trace_completeness: float,
    penalty: float,
) -> float:
    return clamp01(
        harmonic_mean(state_consistency, canon_safety, event_validity, trace_completeness)
        - penalty / 4
    )


def model_selection_score(
    *,
    governance: float,
    evidence_fit: float,
    metric_gain: float,
    trace_completeness: float,
    complexity: float,
    risk: float,
) -> float:
    return clamp01(
        0.35 * governance
        + 0.30 * evidence_fit
        + 0.20 * metric_gain
        + 0.15 * trace_completeness
        - 0.25 * complexity
        - 0.20 * risk
    )


def harmonic_mean(*values: float) -> float:
    clean = tuple(max(clamp01(value), 1e-9) for value in values)
    return len(clean) / sum(1 / value for value in clean)
