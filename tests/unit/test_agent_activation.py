from __future__ import annotations

from pathlib import Path

from aethelis.agents.activation import (
    AgentActivationBuilder,
    build_public_observation_for_activation,
)
from aethelis.schemas.activation import (
    ActivationCandidate,
    ActivationReason,
    ActivationResult,
    ActivationStatus,
    AgentActivationConfig,
)
from aethelis.schemas.run import RunStepPlanItem
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"


def load_valid_bundle():
    load_result = SeedLoader().load(VALID_SEED)
    report = SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )
    assert report.success
    assert load_result.bundle is not None
    return load_result.bundle


def test_activation_schema_contracts() -> None:
    reason = ActivationReason(
        reason_type="scenario_relevance",
        score=3,
        evidence_ids=("mira_search_archive_wrong_key",),
        message="Scenario matrix selected this deterministic static step.",
        visibility_scope="scenario_metadata",
    )
    candidate = ActivationCandidate(
        candidate_id="activation_candidate_test",
        run_id="run_test",
        step_id="step_test",
        agent_id="mira",
        actor_type="agent",
        scenario_id="mira_search_archive_wrong_key",
        status=ActivationStatus.SELECTED_STATIC_PLAN,
        score_total=3,
        reasons=(reason,),
    )
    result = ActivationResult(
        activation_result_id="activation_result_test",
        run_id="run_test",
        step_id="step_test",
        scenario_id="mira_search_archive_wrong_key",
        selected_candidate=candidate,
        candidate_count=1,
        candidates=(candidate,),
    )
    config = AgentActivationConfig()

    assert config.mode.value == "static_trace"
    assert config.scoring_version.value == "rule_based_v0"
    assert config.allow_real_llm is False
    assert config.allow_private_belief_scoring is False
    assert config.top_k == 1
    assert config.selection_threshold == 0
    assert result.provider_called is False
    assert result.scheduler_version == "deterministic_scheduler_v0"
    assert result.world_state_modified is False
    assert result.action_proposal_generated is False
    assert result.hidden_context_used is False


def test_static_step_produces_selected_candidate_with_structured_reasons() -> None:
    bundle = load_valid_bundle()
    step = RunStepPlanItem(
        step_id="step_mira_wrong_key_reject",
        agent_id="mira",
        actor_type="agent",
        scenario_id="mira_search_archive_wrong_key",
    )

    result = AgentActivationBuilder().build_for_step(
        bundle=bundle,
        run_id="run_test",
        step=step,
        config=AgentActivationConfig(),
    )

    assert result.selected_candidate.status == ActivationStatus.SELECTED_STATIC_PLAN
    assert result.selected_candidate.selected_by == "weighted_scheduler_v1_top_k_threshold"
    assert result.selected_candidate.threshold_passed is True
    assert result.selected_candidate.top_k_rank == 1
    assert result.selected_candidate.agent_id == "mira"
    assert result.candidate_count == 6
    assert len(result.candidates) == 1
    assert result.provider_called is False
    assert result.action_proposal_generated is False
    assert result.world_state_modified is False
    assert result.hidden_context_used is False
    reason_types = {reason.reason_type for reason in result.selected_candidate.reasons}
    assert {
        "static_plan_alignment",
        "location_relevance",
        "scenario_relevance",
        "pressure_relevance",
        "action_metadata_relevance",
        "actor_role_relevance",
        "goal_relevance_from_static_profile",
        "relationship_relevance",
        "player_input_relevance",
        "recent_committed_event_relevance",
        "causal_open_thread_relevance",
    } <= reason_types


def test_activation_does_not_include_private_beliefs_memories_secrets_or_hidden_canon() -> None:
    bundle = load_valid_bundle()
    step = RunStepPlanItem(
        step_id="step_mira_wrong_key_reject",
        agent_id="mira",
        actor_type="agent",
        scenario_id="mira_search_archive_wrong_key",
    )

    result = AgentActivationBuilder().build_for_step(
        bundle=bundle,
        run_id="run_test",
        step=step,
        config=AgentActivationConfig(),
    )
    summary_text = str(result.safe_summary())

    forbidden_markers = (
        "belief_mira_key_in_archive",
        "belief_ivo_key_in_safe",
        "memory_",
        "secret_",
        "canon_key_in_workshop_safe",
        "canon_repair_requires_key_and_parts",
        "workshop safe contains the calibration key",
    )
    for marker in forbidden_markers:
        assert marker not in summary_text


def test_activation_uses_pressure_and_action_metadata_safely() -> None:
    bundle = load_valid_bundle()
    step = RunStepPlanItem(
        step_id="step_player_claim_reject",
        agent_id="player",
        actor_type="player",
        scenario_id="player_claim_key_in_hand",
    )

    result = AgentActivationBuilder().build_for_step(
        bundle=bundle,
        run_id="run_test",
        step=step,
        config=AgentActivationConfig(),
    )
    reasons = {reason.reason_type: reason for reason in result.selected_candidate.reasons}

    assert result.selected_candidate.actor_type == "player"
    assert result.selected_candidate.agent_id == "player"
    assert "pressure_civic_trust" in reasons["pressure_relevance"].evidence_ids
    assert "pressure_rumor_spread" in reasons["pressure_relevance"].evidence_ids
    assert "action_report_rumor" in reasons["action_metadata_relevance"].evidence_ids
    assert reasons["actor_role_relevance"].evidence_ids == ("player",)


def test_activation_score_is_deterministic() -> None:
    bundle = load_valid_bundle()
    step = RunStepPlanItem(
        step_id="step_rowan_force_safe_gate",
        agent_id="rowan",
        actor_type="agent",
        scenario_id="unsafe_force_open_safe",
    )
    builder = AgentActivationBuilder()

    first = builder.build_for_step(
        bundle=bundle,
        run_id="run_test",
        step=step,
        config=AgentActivationConfig(),
    )
    second = builder.build_for_step(
        bundle=bundle,
        run_id="run_test",
        step=step,
        config=AgentActivationConfig(),
    )

    assert first.safe_summary() == second.safe_summary()
    assert first.selected_candidate.score_total > 0
    assert first.selected_candidate.tie_break_key == ("false", "rowan")
    assert first.selected_candidate.agent_id == "rowan"
    assert first.candidate_count == 6


def test_activation_weighted_scheduler_changes_with_runtime_features() -> None:
    bundle = load_valid_bundle()
    step = RunStepPlanItem(
        step_id="step_player_claim_reject",
        agent_id="player",
        actor_type="player",
        scenario_id="player_claim_key_in_hand",
    )
    builder = AgentActivationBuilder()

    quiet = builder.build_for_step(
        bundle=bundle,
        run_id="run_test",
        step=step,
        config=AgentActivationConfig(),
    )
    pressured = builder.build_for_step(
        bundle=bundle,
        run_id="run_test",
        step=step,
        config=AgentActivationConfig(),
        evolution_context={
            "latest_pressure_levels": [
                {"pressure_type": "rumor_spread", "after_level": 10},
            ],
        },
    )

    assert pressured.selected_candidate.score_total > quiet.selected_candidate.score_total


def test_activation_can_project_top_k_candidate_selection_without_scheduling() -> None:
    bundle = load_valid_bundle()
    step = RunStepPlanItem(
        step_id="step_mira_wrong_key_reject",
        agent_id="mira",
        actor_type="agent",
        scenario_id="mira_search_archive_wrong_key",
    )

    result = AgentActivationBuilder().build_for_step(
        bundle=bundle,
        run_id="run_test",
        step=step,
        config=AgentActivationConfig(top_k=2, selection_threshold=1),
    )

    assert result.candidate_count == 6
    assert len(result.candidates) == 2
    assert result.selected_candidate.agent_id == "mira"
    assert result.candidates[0].status == ActivationStatus.SELECTED_STATIC_PLAN
    assert result.candidates[1].status == ActivationStatus.SELECTED_STATIC_PLAN
    assert result.candidates[0].top_k_rank == 1
    assert result.candidates[1].top_k_rank == 2


def test_activation_marks_non_selected_threshold_passes_as_background() -> None:
    bundle = load_valid_bundle()
    step = RunStepPlanItem(
        step_id="step_mira_wrong_key_reject",
        agent_id="mira",
        actor_type="agent",
        scenario_id="mira_search_archive_wrong_key",
    )

    result = AgentActivationBuilder().build_for_step(
        bundle=bundle,
        run_id="run_test",
        step=step,
        config=AgentActivationConfig(
            include_non_selected_candidates=True,
            top_k=1,
            selection_threshold=1,
        ),
    )

    statuses = {candidate.status for candidate in result.candidates}
    assert result.selected_candidate.status == ActivationStatus.SELECTED_STATIC_PLAN
    assert ActivationStatus.BACKGROUND in statuses
    assert all(
        candidate.status != ActivationStatus.BACKGROUND or candidate.threshold_passed
        for candidate in result.candidates
    )


def test_activation_threshold_can_mark_candidate_skipped_without_scheduling() -> None:
    bundle = load_valid_bundle()
    step = RunStepPlanItem(
        step_id="step_mira_wrong_key_reject",
        agent_id="mira",
        actor_type="agent",
        scenario_id="mira_search_archive_wrong_key",
    )

    result = AgentActivationBuilder().build_for_step(
        bundle=bundle,
        run_id="run_test",
        step=step,
        config=AgentActivationConfig(selection_threshold=99, top_k=1),
    )

    assert result.selected_candidate.status == ActivationStatus.SKIPPED
    assert result.selected_candidate.threshold_passed is False
    assert result.provider_called is False
    assert result.action_proposal_generated is False


def test_activation_uses_safe_evolution_context_for_recent_and_causal_reasons() -> None:
    bundle = load_valid_bundle()
    step = RunStepPlanItem(
        step_id="step_mira_wrong_key_reject",
        agent_id="mira",
        actor_type="agent",
        scenario_id="mira_search_archive_wrong_key",
    )
    evolution_context = {
        "causal_node_count": 4,
        "causal_runtime_summary": {
            "latest_committed_event_ids": ["committed_previous"],
        },
        "latest_pressure_levels": [
            {
                "pressure_type": "rumor_spread",
                "after_level": 8,
                "source_step_id": "step_previous",
            }
        ],
    }

    result = AgentActivationBuilder().build_for_step(
        bundle=bundle,
        run_id="run_test",
        step=step,
        config=AgentActivationConfig(),
        evolution_context=evolution_context,
    )
    reasons = {reason.reason_type: reason for reason in result.selected_candidate.reasons}

    assert reasons["recent_committed_event_relevance"].score == 1
    assert reasons["recent_committed_event_relevance"].evidence_ids == ("committed_previous",)
    assert reasons["causal_open_thread_relevance"].score == 1
    assert result.provider_called is False
    assert "belief_ivo_key_in_safe" not in str(result.safe_summary())


def test_public_observation_for_activation_does_not_return_cognition_context() -> None:
    bundle = load_valid_bundle()

    observation = build_public_observation_for_activation(
        bundle,
        actor_id="mira",
        actor_type="agent",
        scenario_id="mira_search_archive_wrong_key",
    )

    assert observation.agent_id == "mira"
    assert observation.location.id == "central_archive"
    assert not hasattr(observation, "owned_beliefs")
    assert not hasattr(observation, "owned_memories")
