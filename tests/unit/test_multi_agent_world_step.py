from __future__ import annotations

from pathlib import Path

import pytest

from aethelis.agents.action_proposal import (
    ActionProposalGenerationResult,
    ActionProposalSource,
    ProposalBehaviorDecision,
)
from aethelis.runtime.multi_agent_step import run_multi_agent_world_step
from aethelis.schemas.events import ActionIntent, ActionProposal, VerificationDecision

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"


def test_multi_agent_world_step_produces_governed_runtime_objects() -> None:
    result = run_multi_agent_world_step(
        seed_path=VALID_SEED,
        step_id="r5_b4_single_governed_step",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo",),
        apply=False,
    )

    assert result.context.active_agent_ids == ("ivo",)
    assert result.context_frames[0].agent_id == "ivo"
    assert result.action_proposals[0].proposer_agent_id == "ivo"
    assert result.proposal_bundle.frames[0].proposal_id == result.action_proposals[0].id
    assert result.dynamics_summary.bundle_id == result.proposal_bundle.id
    assert result.arbitration_recommendation.bundle_id == result.proposal_bundle.id
    assert result.routed_proposal_ids == (result.action_proposals[0].id,)
    assert len(result.event_candidates) == 1
    assert len(result.verification_results) == 1
    assert len(result.committed_events) == 1
    assert len(result.state_diffs) == 1
    assert result.event_candidates[0].source_action_proposal_id == result.action_proposals[0].id
    assert result.committed_events[0].event_candidate_id == result.event_candidates[0].id
    assert result.state_diffs[0].source_action_proposal_id is None
    assert result.provider_called is False
    assert result.db_written is False


def test_multi_agent_step_preserves_per_agent_context_source_attribution() -> None:
    result = run_multi_agent_world_step(
        seed_path=VALID_SEED,
        step_id="r5_b4_context_isolation",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo", "mira"),
        proposal_results={
            "ivo": _generation(
                _proposal(
                    "ivo",
                    "proposal_ivo_safe",
                    ActionIntent.INVESTIGATE,
                    target_location_id="workshop_lane",
                    target_entity_ids=("workshop_safe",),
                )
            ),
            "mira": _generation(
                _proposal(
                    "mira",
                    "proposal_mira_archive",
                    ActionIntent.OBSERVE,
                    target_location_id="central_archive",
                    target_entity_ids=("harmonic_tuner",),
                ),
                behavior_score=0.55,
            ),
        },
    )

    ivo_context = result.context.frame_for("ivo")
    mira_context = result.context.frame_for("mira")
    ivo_frame = result.proposal_bundle.frame_for_proposal("proposal_ivo_safe:ivo")
    mira_frame = result.proposal_bundle.frame_for_proposal("proposal_mira_archive:mira")

    assert result.context.no_shared_omniscient_context is True
    assert "belief_ivo_key_in_safe" in ivo_frame.selected_belief_ids
    assert "belief_mira_key_in_archive" not in ivo_frame.selected_belief_ids
    assert "belief_mira_key_in_archive" in mira_frame.selected_belief_ids
    assert "mem_mira_archive_ledger" not in ivo_frame.selected_memory_ids
    assert ivo_frame.context_source_ids == ivo_context.packed_source_ids
    assert mira_frame.context_source_ids == mira_context.packed_source_ids
    assert set(ivo_frame.context_source_ids).isdisjoint(mira_context.retrieval.filtered_belief_ids)
    for boundary in result.verifier_retrieval_boundaries:
        context_frame = result.context.frame_for(boundary.agent_id)
        proposal_frame = result.proposal_bundle.frame_for_proposal(
            next(
                frame.proposal_id
                for frame in result.proposal_bundle.frames
                if frame.agent_id == boundary.agent_id
            )
        )
        assert boundary.context_source_ids == context_frame.packed_source_ids
        assert boundary.verifier_selected_belief_ids == proposal_frame.selected_belief_ids
        assert boundary.verifier_selected_memory_ids == proposal_frame.selected_memory_ids
        assert set(boundary.verifier_selected_belief_ids).isdisjoint(
            boundary.verifier_filtered_belief_ids
        )
        assert set(boundary.verifier_selected_memory_ids).isdisjoint(
            boundary.verifier_suppressed_memory_ids
        )


def test_proposal_owner_mismatch_fails_before_governance_objects() -> None:
    with pytest.raises(ValueError, match="Proposal owner mismatch"):
        run_multi_agent_world_step(
            seed_path=VALID_SEED,
            step_id="r5_b4_owner_mismatch",
            scenario_id="ivo_inspect_workshop_safe_fixture",
            active_agent_ids=("ivo",),
            proposal_results={
                "ivo": _generation(
                    _proposal(
                        "mira",
                        "proposal_wrong_owner",
                        ActionIntent.INVESTIGATE,
                        target_location_id="workshop_lane",
                        target_entity_ids=("workshop_safe",),
                    )
                ),
            },
        )


def test_conflict_routing_blocks_lower_priority_proposal_before_governance() -> None:
    result = run_multi_agent_world_step(
        seed_path=VALID_SEED,
        step_id="r5_b4_conflict_route",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("rowan", "ivo"),
        proposal_results={
            "rowan": _generation(
                _proposal(
                    "rowan",
                    "proposal_rowan_guard_key",
                    ActionIntent.GUARD,
                    target_location_id="workshop_lane",
                    target_entity_ids=("calibration_key",),
                ),
                behavior_score=0.9,
            ),
            "ivo": _generation(
                _proposal(
                    "ivo",
                    "proposal_ivo_move_key",
                    ActionIntent.MOVE,
                    target_location_id="workshop_lane",
                    target_entity_ids=("calibration_key",),
                ),
                behavior_score=0.4,
            ),
        },
    )

    blocked_id = "proposal_ivo_move_key:ivo"
    assert result.dynamics_summary.conflicts
    assert result.blocked_proposal_ids == (blocked_id,)
    assert blocked_id not in result.routed_proposal_ids
    assert all(
        candidate.source_action_proposal_id != blocked_id
        for candidate in result.event_candidates
    )
    blocked_route = _route_for(result, blocked_id)
    assert blocked_route.route == "blocked_by_hard_conflict"
    assert blocked_route.event_candidate_id is None
    assert blocked_route.verification_result_id is None
    assert blocked_route.committed_event_id is None
    assert blocked_route.state_diff_id is None


def test_cooperation_joint_candidate_is_evidence_not_direct_commit() -> None:
    result = run_multi_agent_world_step(
        seed_path=VALID_SEED,
        step_id="r5_b4_joint_boundary",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo", "taren"),
        proposal_results={
            "ivo": _generation(
                _proposal(
                    "ivo",
                    "proposal_ivo_investigate_lens",
                    ActionIntent.INVESTIGATE,
                    target_location_id="old_aqueduct",
                    target_entity_ids=("gate_lens",),
                ),
                behavior_score=0.65,
            ),
            "taren": _generation(
                _proposal(
                    "taren",
                    "proposal_taren_repair_lens",
                    ActionIntent.REPAIR,
                    target_location_id="old_aqueduct",
                    target_entity_ids=("gate_lens",),
                ),
                behavior_score=0.7,
            ),
        },
    )

    joint = result.joint_intent_candidates[0]

    assert result.dynamics_summary.cooperations
    assert joint.proposal_ids == (
        "proposal_ivo_investigate_lens:ivo",
        "proposal_taren_repair_lens:taren",
    )
    assert result.routed_proposal_ids == ()
    assert result.event_candidates == ()
    assert result.committed_events == ()
    assert result.state_diffs == ()
    assert joint.can_modify_world_state is False
    assert joint.can_mutate_canon is False
    assert result.joint_intent_direct_commit is False


def test_requires_revision_route_stops_before_governance_objects() -> None:
    result = run_multi_agent_world_step(
        seed_path=VALID_SEED,
        step_id="r5_b4_revision_route",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo", "mira"),
        proposal_results={
            "ivo": _generation(
                _proposal(
                    "ivo",
                    "proposal_ivo_inspect_safe",
                    ActionIntent.INVESTIGATE,
                    target_location_id="workshop_lane",
                    target_entity_ids=("workshop_safe",),
                ),
                behavior_score=0.8,
            ),
            "mira": _generation(
                _proposal(
                    "mira",
                    "proposal_mira_observe_safe",
                    ActionIntent.OBSERVE,
                    target_location_id="workshop_lane",
                    target_entity_ids=("workshop_safe",),
                ),
                behavior_score=0.4,
            ),
        },
    )

    revision_id = "proposal_mira_observe_safe:mira"

    assert result.dynamics_summary.contentions
    assert result.dynamics_summary.conflicts == ()
    assert result.dynamics_summary.cooperations == ()
    assert result.revision_required_proposal_ids == (revision_id,)
    assert revision_id not in result.routed_proposal_ids
    assert all(
        candidate.source_action_proposal_id != revision_id
        for candidate in result.event_candidates
    )
    revision_route = _route_for(result, revision_id)
    assert revision_route.route == "requires_revision"
    assert revision_route.event_candidate_id is None
    assert revision_route.verification_result_id is None
    assert revision_route.committed_event_id is None
    assert revision_route.state_diff_id is None


def test_independent_proposals_route_through_existing_candidate_and_verifier_chain() -> None:
    result = run_multi_agent_world_step(
        seed_path=VALID_SEED,
        step_id="r5_b4_independent_route",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo", "mira"),
        proposal_results={
            "ivo": _generation(
                _proposal(
                    "ivo",
                    "proposal_ivo_safe",
                    ActionIntent.INVESTIGATE,
                    target_location_id="workshop_lane",
                    target_entity_ids=("workshop_safe",),
                ),
                behavior_score=0.8,
            ),
            "mira": _generation(
                _proposal(
                    "mira",
                    "proposal_mira_archive",
                    ActionIntent.OBSERVE,
                    target_location_id="central_archive",
                    target_entity_ids=("harmonic_tuner",),
                ),
                behavior_score=0.5,
            ),
        },
    )

    assert result.arbitration_recommendation.primary_proposal_id == "proposal_ivo_safe:ivo"
    assert result.independent_proposal_ids == ("proposal_mira_archive:mira",)
    assert set(result.routed_proposal_ids) == {
        "proposal_ivo_safe:ivo",
        "proposal_mira_archive:mira",
    }
    assert {
        candidate.source_action_proposal_id for candidate in result.event_candidates
    } == set(result.routed_proposal_ids)
    assert len(result.verification_results) == 2
    assert any(
        verification.decision == VerificationDecision.COMMIT
        for verification in result.verification_results
    )
    assert result.arbitration_direct_commit is False
    assert result.dynamics_direct_commit is False


def test_rejected_verification_produces_no_committed_event_or_state_diff() -> None:
    result = run_multi_agent_world_step(
        seed_path=VALID_SEED,
        step_id="r5_b4_reject_boundary",
        scenario_id="mira_search_archive_wrong_key",
        active_agent_ids=("mira",),
    )

    assert len(result.event_candidates) == 1
    assert result.verification_results[0].decision == VerificationDecision.REJECT
    assert result.committed_events == ()
    assert result.state_diffs == ()
    assert result.state_diff_applied is False


def test_apply_false_does_not_mutate_world_state_and_apply_true_uses_applier() -> None:
    dry_run = run_multi_agent_world_step(
        seed_path=VALID_SEED,
        step_id="r5_b4_apply_false",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo",),
        apply=False,
    )
    applied = run_multi_agent_world_step(
        seed_path=VALID_SEED,
        step_id="r5_b4_apply_true",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo",),
        apply=True,
    )

    assert dry_run.committed_events
    assert dry_run.state_diff_applied is False
    assert dry_run.apply_reports == ()
    assert dry_run.applied_world_state is None
    assert applied.state_diff_applied is True
    assert applied.apply_reports[0].applied is True
    assert applied.applied_world_state is not None
    calibration_key = next(
        resource
        for resource in applied.applied_world_state.resources
        if resource.id == "calibration_key"
    )
    assert calibration_key.discovery_state.discovered_by_agent_ids == ("ivo",)


def test_filtered_context_does_not_leak_into_proposal_source_attribution() -> None:
    result = run_multi_agent_world_step(
        seed_path=VALID_SEED,
        step_id="r5_b4_source_boundary",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo", "mira"),
        proposal_results={
            "ivo": _generation(
                _proposal(
                    "ivo",
                    "proposal_ivo_safe",
                    ActionIntent.INVESTIGATE,
                    target_location_id="workshop_lane",
                    target_entity_ids=("workshop_safe",),
                )
            ),
            "mira": _generation(
                _proposal(
                    "mira",
                    "proposal_mira_archive",
                    ActionIntent.OBSERVE,
                    target_location_id="central_archive",
                    target_entity_ids=("harmonic_tuner",),
                )
            ),
        },
    )

    for frame in result.proposal_bundle.frames:
        context_frame = result.context.frame_for(frame.agent_id)
        filtered_or_suppressed = {
            *context_frame.retrieval.filtered_belief_ids,
            *context_frame.retrieval.suppressed_memory_ids,
            *context_frame.suppressed_source_ids,
        }
        assert set(frame.context_source_ids).isdisjoint(filtered_or_suppressed)
        assert set(frame.selected_belief_ids).isdisjoint(
            context_frame.retrieval.filtered_belief_ids
        )
        assert set(frame.selected_memory_ids).isdisjoint(
            context_frame.retrieval.suppressed_memory_ids
        )


def test_multi_agent_world_step_governance_boundary_flags_are_false() -> None:
    result = run_multi_agent_world_step(
        seed_path=VALID_SEED,
        step_id="r5_b4_boundary_flags",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo",),
    )

    summary = result.safe_summary()

    assert result.provider_called is False
    assert result.db_written is False
    assert result.arbitration_direct_commit is False
    assert result.dynamics_direct_commit is False
    assert result.joint_intent_direct_commit is False
    assert result.can_mutate_canon is False
    assert result.direct_world_state_mutation is False
    assert summary["db_written"] is False
    assert result.event_candidates[0].source_action_proposal_id == result.action_proposals[0].id
    assert result.committed_events[0].state_diff.committed_event_id == result.committed_events[0].id


def _proposal(
    agent_id: str,
    proposal_id: str,
    intent: ActionIntent,
    *,
    target_location_id: str,
    target_entity_ids: tuple[str, ...],
) -> ActionProposal:
    return ActionProposal(
        id=proposal_id,
        proposer_agent_id=agent_id,
        intent=intent,
        rationale=f"{agent_id} proposes {intent.value} as pre-governance evidence.",
        target_location_id=target_location_id,
        target_entity_ids=target_entity_ids,
        expected_outcome=f"{agent_id} proposal must pass governed verification.",
    )


def _generation(
    proposal: ActionProposal,
    *,
    behavior_score: float = 0.7,
    behavior_decision: ProposalBehaviorDecision = ProposalBehaviorDecision.ACCEPT,
) -> ActionProposalGenerationResult:
    return ActionProposalGenerationResult(
        proposal=proposal,
        source=ActionProposalSource.TEST_STUB,
        provider_called=False,
        behavior_score=behavior_score,
        behavior_decision=behavior_decision,
        behavior_risk_flags=(),
    )


def _route_for(result, proposal_id: str):
    return next(route for route in result.routes if route.proposal_id == proposal_id)
