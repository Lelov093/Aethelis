from pathlib import Path

from aethelis.agents.action_proposal import ProposalSourceMode
from aethelis.algorithms.mechanisms import MechanismKind
from aethelis.algorithms.runtime_wiring import (
    RUNTIME_MECHANISMS,
    build_runtime_algorithm_decisions,
)
from aethelis.evolution import DeterministicEvolutionBuilder
from aethelis.runtime.single_step import run_single_step
from aethelis.seeds.loader import SeedLoader

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_SCORE_FIELDS = {
    "agent_activation": {
        "goal_component",
        "pressure_component",
        "relationship_component",
        "memory_component",
        "causal_component",
        "risk_penalty",
        "background_status",
        "threshold_passed",
    },
    "observation_scope": {
        "visible_scope_score",
        "access_score",
        "privacy_risk",
        "hidden_removed_count",
        "visible_object_ids",
        "hidden_filter_guard",
        "behavior_source",
        "alignment_status",
    },
    "cognition_retrieval": {
        "selected_memory_ids",
        "selected_belief_ids",
        "available_memory_ids",
        "suppressed_memory_ids",
        "behavior_source",
        "alignment_status",
        "salience_component",
        "recency_component",
        "belief_confidence_component",
        "suppression_penalty",
    },
    "action_proposal": {
        "goal_utility",
        "pressure_alignment",
        "feasibility_score",
        "governance_score",
        "risk_penalty",
        "proposal_confidence",
        "proposal_behavior_score",
        "proposal_behavior_decision",
        "proposal_behavior_risk_flags",
        "behavior_source",
        "alignment_status",
    },
    "event_candidate": {
        "event_type_mapping_score",
        "actor_score",
        "target_score",
        "precondition_score",
        "predicted_state_diff_score",
        "verification_requirement_score",
        "trace_completeness_score",
        "candidate_route",
        "candidate_status",
        "candidate_behavior_score",
        "behavior_source",
        "alignment_status",
    },
    "event_verification": {
        "hard_gate",
        "check_pass_rate",
        "risk_penalty",
        "posterior",
        "decision_score",
        "verification_decision",
        "behavior_source",
        "alignment_status",
        "rejected_claim_count",
    },
    "belief_confidence": {
        "source_belief_id",
        "prior_confidence",
        "support_count",
        "contradiction_count",
        "source_reliability",
        "posterior_confidence",
        "confidence_after",
        "behavior_source",
        "alignment_status",
    },
    "memory_decay": {
        "memory_id",
        "decay_delta",
        "reinforcement_delta",
        "suppression_penalty",
        "retained_strength",
        "updated_strength",
        "behavior_source",
        "alignment_status",
    },
    "relationship_update": {
        "relationship_id",
        "event_delta",
        "cooperation_score",
        "conflict_score",
        "trust_before",
        "trust_delta",
        "trust_after",
        "asymmetric_direction",
        "behavior_source",
        "alignment_status",
    },
    "world_pressure": {
        "pressure_type",
        "before_level",
        "event_impulse",
        "ewma_component",
        "after_level",
        "threshold_triggered",
        "pressure_saturation",
        "behavior_source",
        "alignment_status",
    },
    "evolution_opportunity": {
        "opportunity_score",
        "pressure_component",
        "safety_gate",
        "causal_open_thread_component",
        "goal_intersection_component",
        "defer_decision",
        "opportunity_route",
        "behavior_source",
        "alignment_status",
    },
    "causal_graph": {
        "edge_id",
        "edge_type",
        "source_event_id",
        "target_state_diff_id",
        "temporal_adjacency",
        "entity_overlap",
        "pressure_linkage",
        "edge_confidence",
        "behavior_source",
        "alignment_status",
    },
    "player_input": {
        "player_input_id",
        "input_type_score",
        "permission_gate",
        "claim_risk",
        "canon_contamination_risk",
        "verification_requirement",
        "assimilation_decision",
        "behavior_source",
        "alignment_status",
    },
    "evaluation_scoring": {
        "state_consistency",
        "canon_safety",
        "knowledge_boundary",
        "event_validity",
        "causal_coherence",
        "world_pressure_alignment",
        "player_impact",
        "trace_completeness",
        "penalty_total",
        "behavior_source",
        "alignment_status",
    },
    "model_combination": {
        "variant_ids",
        "governance_score",
        "evidence_fit",
        "metric_gain",
        "complexity_penalty",
        "risk_penalty",
        "trace_completeness",
        "selection_score",
        "pareto_rank",
        "coverage_score",
        "pareto_or_weighted_decision",
        "behavior_source",
        "alignment_status",
    },
}


def test_product05_runtime_wiring_covers_all_mechanisms() -> None:
    assert len(RUNTIME_MECHANISMS) == 15
    assert set(RUNTIME_MECHANISMS) == set(MechanismKind)
    assert len(set(RUNTIME_MECHANISMS)) == len(RUNTIME_MECHANISMS)


def test_batch22_mechanisms_have_runtime_decisions() -> None:
    result = run_single_step(
        seed_path=ROOT / "seeds" / "mistgate_v01",
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        proposal_source=ProposalSourceMode.DETERMINISTIC,
        apply=True,
    )
    evolution_update = _evolution_update(result)

    decisions = build_runtime_algorithm_decisions(
        seed_path=ROOT / "seeds" / "mistgate_v01",
        result=result,
        config_path=ROOT / "configs" / "v11_algorithm_mechanism_completion.yaml",
        evolution_update=evolution_update,
    )
    by_id = {decision.mechanism_id: decision for decision in decisions}

    assert set(by_id) == {kind.value for kind in MechanismKind}
    assert by_id["observation_scope"].runtime_object_type == "observation_scope"
    assert by_id["event_candidate"].runtime_object_type == "event_candidate"
    assert by_id["evolution_opportunity"].runtime_object_type == "evolution_update"
    assert by_id["model_combination"].runtime_object_type == "algorithm_mode"

    for mechanism_id, expected_fields in REQUIRED_SCORE_FIELDS.items():
        decision = by_id[mechanism_id]
        assert expected_fields <= set(decision.score_breakdown)
        assert "runtime_score" in decision.score_breakdown
        assert "source_ids" in decision.score_breakdown
        assert "output_object_ids" in decision.score_breakdown
        assert isinstance(decision.score, float)
        assert 0.0 <= decision.score <= 1.0

    assert by_id["cognition_retrieval"].score_breakdown["selected_memory_ids"]
    assert by_id["relationship_update"].score_breakdown["relationship_id"]
    assert by_id["causal_graph"].score_breakdown["edge_id"]
    assert by_id["model_combination"].score_breakdown["pareto_rank"] == 1
    assert "weighted(goal" in by_id["agent_activation"].formula
    assert "ewma" in by_id["world_pressure"].formula
    assert by_id["model_combination"].score_breakdown["risk_flags"] == (
        "complexity_penalty_applied",
    )
    assert by_id["cognition_retrieval"].score_breakdown["selected_memory_ids"] == tuple(
        result.retrieval_summary["selected_memory_ids"]
    )
    observation = result.observation_summary
    assert observation is not None
    expected_visible = tuple(
        dict.fromkeys(
            [
                observation["location_id"],
                *observation["visible_entity_ids"],
                *observation["visible_resource_ids"],
                *observation["visible_agent_ids"],
                *observation["visible_public_fact_ids"],
                *observation["visible_rumor_ids"],
            ]
        )
    )
    assert by_id["observation_scope"].score_breakdown["visible_object_ids"] == expected_visible
    assert by_id["cognition_retrieval"].score_breakdown["suppressed_memory_ids"] == tuple(
        result.retrieval_summary["suppressed_memory_ids"]
    )
    assert by_id["action_proposal"].score_breakdown["proposal_behavior_score"] == (
        result.proposal_behavior_score
    )
    assert by_id["action_proposal"].score_breakdown["proposal_behavior_decision"] == (
        result.proposal_behavior_decision
    )
    assert by_id["event_candidate"].score_breakdown["candidate_route"] == (
        result.candidate_behavior_route
    )
    assert by_id["event_candidate"].score_breakdown["candidate_status"] == (
        result.event_candidate.status.value
    )
    assert by_id["event_verification"].score_breakdown["verification_decision"] == (
        result.verification_result.decision.value
    )
    relationship = evolution_update.relationship_updates[0]
    assert by_id["relationship_update"].score_breakdown["relationship_id"] == (
        relationship.relationship_id
    )
    assert by_id["relationship_update"].score_breakdown["trust_delta"] == relationship.trust_delta
    pressure = evolution_update.pressure_updates[0]
    assert by_id["world_pressure"].score_breakdown["pressure_type"] == pressure.pressure_type
    assert by_id["world_pressure"].score_breakdown["after_level"] == pressure.after_level
    assert by_id["evolution_opportunity"].score_breakdown["opportunity_route"] == (
        evolution_update.opportunity_route
    )
    assert by_id["causal_graph"].score_breakdown["edge_confidence"] == (
        evolution_update.causal_graph.edges[0].confidence
    )


def test_batch45_player_input_score_breakdown_uses_guarded_input_object() -> None:
    result = run_single_step(
        seed_path=ROOT / "seeds" / "mistgate_v01",
        agent_id="player",
        scenario_id="player_claim_key_in_hand",
        proposal_source=ProposalSourceMode.DETERMINISTIC,
        apply=True,
    )

    decisions = build_runtime_algorithm_decisions(
        seed_path=ROOT / "seeds" / "mistgate_v01",
        result=result,
        config_path=ROOT / "configs" / "v11_algorithm_mechanism_completion.yaml",
    )
    player_decision = {
        decision.mechanism_id: decision for decision in decisions
    }["player_input"]

    breakdown = player_decision.score_breakdown
    assert breakdown["player_input_id"] == "claim_player_key_in_hand"
    assert breakdown["claim_risk"] == 0.85
    assert breakdown["canon_contamination_risk"] == 0.85
    assert breakdown["assimilation_decision"] == "rejected_claim"
    assert "player_claim_not_canon" in breakdown["risk_flags"]
    assert "claim_player_key_in_hand" in breakdown["source_ids"]
    assert "belief_candidate_claim_player_key_in_hand" in breakdown["output_object_ids"]


def _evolution_update(result):
    load_result = SeedLoader().load(ROOT / "seeds" / "mistgate_v01")
    assert load_result.bundle is not None
    assert result.verification_result is not None
    update = DeterministicEvolutionBuilder().build_for_step(
        bundle=load_result.bundle,
        step_id="test_step_1",
        scenario_id=result.scenario_id,
        agent_id=result.agent_id,
        decision=result.verification_result.decision,
        committed_event_id=result.committed_event.id if result.committed_event else None,
        state_diff_id=result.committed_event.state_diff.id if result.committed_event else None,
        verification_result_id=result.verification_result.id,
        event_candidate_id=result.event_candidate.id if result.event_candidate else None,
        state_diff_applied=result.state_diff_applied,
        verification_result=result.verification_result,
        event_candidate=result.event_candidate,
    )
    assert update is not None
    return update
