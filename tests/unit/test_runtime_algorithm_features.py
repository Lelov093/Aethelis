import pytest

from aethelis.algorithms.runtime_features import (
    bounded_trust_update,
    causal_edge_confidence,
    decay_reinforcement_score,
    ewma_pressure_update,
    harmonic_governance_score,
    model_selection_score,
    opportunity_score,
    player_assimilation_score,
    proposal_confidence_score,
    retrieval_rank_score,
    weighted_scheduler_score,
)


def test_runtime_feature_formulas_cover_p0_mechanism_math() -> None:
    assert weighted_scheduler_score(
        goal=0.8,
        pressure=0.6,
        relationship=0.7,
        memory=0.4,
        causal=1.0,
        risk=0.2,
    ) == 0.666
    assert retrieval_rank_score(
        salience=0.7,
        recency=1.0,
        belief_confidence=0.85,
        suppression=0.1,
    ) == pytest.approx(0.72)
    assert decay_reinforcement_score(
        retained=0.5,
        reinforcement=0.35,
        suppression=0.2,
    ) == pytest.approx(0.65)
    assert bounded_trust_update(trust_before=4, event_delta=1.0) == (2, 5)
    assert ewma_pressure_update(before_level=8, event_impulse=1.0) == (0.896, 9)
    assert opportunity_score(
        pressure_component=0.8,
        safety=1.0,
        causal=0.5,
        goal=0.75,
    ) == pytest.approx(0.78)
    assert causal_edge_confidence(
        temporal=1.0,
        entity_overlap=1.0,
        pressure_linkage=0.5,
    ) == pytest.approx(0.9)
    assert player_assimilation_score(
        permission_gate=1.0,
        claim_risk=0.85,
        verification_requirement=1.0,
    ) == pytest.approx(0.15)
    assert model_selection_score(
        governance=1.0,
        evidence_fit=0.8,
        metric_gain=0.5,
        trace_completeness=1.0,
        complexity=0.2,
        risk=0.0,
    ) == pytest.approx(0.79)


def test_runtime_feature_formulas_cover_p1_boundaries() -> None:
    assert proposal_confidence_score(
        goal_utility=1.0,
        pressure_alignment=1.0,
        feasibility=1.0,
        governance=1.0,
        risk=2.0,
    ) == 0.6
    assert harmonic_governance_score(
        state_consistency=1.0,
        canon_safety=0.0,
        event_validity=1.0,
        trace_completeness=1.0,
        penalty=1.0,
    ) == 0.0
    assert bounded_trust_update(trust_before=-4, event_delta=-1.0) == (-2, -5)
