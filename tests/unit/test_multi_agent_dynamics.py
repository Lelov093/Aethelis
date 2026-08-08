from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aethelis.agents.dynamics import (
    AgentProposalFrame,
    JointIntentCandidate,
    MultiAgentDynamicsSummary,
    MultiAgentProposalBundle,
    ProposalArbitrationRecommendation,
    ProposalBundleSummary,
    analyze_multi_agent_dynamics,
    recommend_arbitration,
)
from aethelis.agents.retrieval import build_multi_agent_step_context
from aethelis.schemas.events import (
    ActionIntent,
    ActionProposal,
    CommittedEvent,
    EventCandidate,
    StateDiff,
    VerificationResult,
)
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"


def test_multi_agent_proposal_bundle_holds_separate_agent_frames() -> None:
    bundle = _proposal_bundle(
        _frame("ivo", "proposal_ivo_inspect_safe", ActionIntent.INVESTIGATE),
        _frame("mira", "proposal_mira_search_records", ActionIntent.OBSERVE),
    )

    summary = bundle.summary()

    assert summary.active_agent_ids == ("ivo", "mira")
    assert summary.proposal_ids == ("proposal_ivo_inspect_safe", "proposal_mira_search_records")
    assert bundle.frames[0].context_source_ids == ("ctx_ivo",)
    assert bundle.frames[1].context_source_ids == ("ctx_mira",)
    assert bundle.can_modify_world_state is False
    assert bundle.can_mutate_canon is False


def test_same_resource_and_location_proposals_produce_contention() -> None:
    seed = _load_valid_bundle()
    bundle = _proposal_bundle(
        _frame(
            "ivo",
            "proposal_ivo_repair_lens",
            ActionIntent.REPAIR,
            target_location_id="old_aqueduct",
            target_resource_ids=("gate_lens",),
        ),
        _frame(
            "taren",
            "proposal_taren_observe_lens",
            ActionIntent.OBSERVE,
            target_location_id="old_aqueduct",
            target_resource_ids=("gate_lens",),
        ),
    )

    summary = analyze_multi_agent_dynamics(bundle=seed, proposal_bundle=bundle)

    assert {item.contention_type for item in summary.contentions} >= {
        "same_target",
        "resource_contention",
        "location_contention",
    }
    assert summary.pressure_aligned_proposal_ids == (
        "proposal_ivo_repair_lens",
        "proposal_taren_observe_lens",
    )


def test_opposing_intents_over_same_target_produce_conflict() -> None:
    seed = _load_valid_bundle()
    bundle = _proposal_bundle(
        _frame(
            "rowan",
            "proposal_rowan_guard_key",
            ActionIntent.GUARD,
            target_location_id="workshop_lane",
            target_entity_ids=("calibration_key",),
            priority_score=0.8,
        ),
        _frame(
            "ivo",
            "proposal_ivo_move_key",
            ActionIntent.MOVE,
            target_location_id="workshop_lane",
            target_entity_ids=("calibration_key",),
            priority_score=0.4,
        ),
    )

    summary = analyze_multi_agent_dynamics(bundle=seed, proposal_bundle=bundle)
    recommendation = recommend_arbitration(bundle=seed, proposal_bundle=bundle)

    assert summary.conflicts[0].conflict_type == "opposing_intents"
    assert recommendation.primary_proposal_id == "proposal_rowan_guard_key"
    assert recommendation.proposals_blocked_by_hard_conflict == ("proposal_ivo_move_key",)


def test_complementary_intents_produce_cooperation_and_joint_candidate() -> None:
    seed = _load_valid_bundle()
    bundle = _proposal_bundle(
        _frame(
            "ivo",
            "proposal_ivo_investigate_lens",
            ActionIntent.INVESTIGATE,
            target_location_id="old_aqueduct",
            target_resource_ids=("gate_lens",),
        ),
        _frame(
            "taren",
            "proposal_taren_repair_lens",
            ActionIntent.REPAIR,
            target_location_id="old_aqueduct",
            target_resource_ids=("gate_lens",),
        ),
    )

    summary = analyze_multi_agent_dynamics(bundle=seed, proposal_bundle=bundle)
    recommendation = recommend_arbitration(bundle=seed, proposal_bundle=bundle)

    assert summary.cooperations[0].cooperation_type == "mutually_supportive_intents"
    assert summary.dependencies[0].prerequisite_proposal_id == "proposal_ivo_investigate_lens"
    assert recommendation.joint_intent_candidates[0].proposal_ids == (
        "proposal_ivo_investigate_lens",
        "proposal_taren_repair_lens",
    )


def test_relationship_faction_and_pressure_signals_order_recommendation() -> None:
    seed = _load_valid_bundle()
    bundle = _proposal_bundle(
        _frame(
            "ivo",
            "proposal_ivo_pressure_lens",
            ActionIntent.OBSERVE,
            target_location_id="old_aqueduct",
            target_resource_ids=("gate_lens",),
            confidence=0.5,
            priority_score=0.55,
        ),
        _frame(
            "mira",
            "proposal_mira_archive_note",
            ActionIntent.OBSERVE,
            target_location_id="central_archive",
            target_entity_ids=("harmonic_tuner",),
            confidence=0.5,
            priority_score=0.5,
        ),
        _frame(
            "taren",
            "proposal_taren_pressure_lens",
            ActionIntent.GUARD,
            target_location_id="old_aqueduct",
            target_resource_ids=("gate_lens",),
            confidence=0.5,
            priority_score=0.5,
        ),
    )

    summary = analyze_multi_agent_dynamics(bundle=seed, proposal_bundle=bundle)
    recommendation = recommend_arbitration(bundle=seed, proposal_bundle=bundle)
    scores = recommendation.score_components

    assert summary.relationship_signals[0].signal_type == "relationship_support"
    assert summary.faction_tension_signals
    assert "proposal_ivo_pressure_lens" in summary.pressure_aligned_proposal_ids
    assert scores["proposal_ivo_pressure_lens"]["pressure_alignment"] > 0
    assert scores["proposal_mira_archive_note"]["pressure_alignment"] == 0
    assert recommendation.primary_proposal_id == "proposal_ivo_pressure_lens"


def test_belief_divergence_uses_counts_without_private_belief_leakage() -> None:
    seed = _load_valid_bundle()
    context = build_multi_agent_step_context(
        seed,
        step_id="r5_b3_divergence",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo", "mira"),
    )
    bundle = _proposal_bundle(
        AgentProposalFrame.from_proposal(
            _proposal(
                "ivo",
                "proposal_ivo_key",
                ActionIntent.INVESTIGATE,
                target_entity_ids=("calibration_key",),
            ),
            context_frame=context.frame_for("ivo"),
        ),
        AgentProposalFrame.from_proposal(
            _proposal(
                "mira",
                "proposal_mira_key",
                ActionIntent.INVESTIGATE,
                target_entity_ids=("calibration_key",),
            ),
            context_frame=context.frame_for("mira"),
        ),
    )

    summary = analyze_multi_agent_dynamics(bundle=seed, proposal_bundle=bundle)
    divergence_payload = summary.belief_divergences[0].model_dump(mode="json")

    assert summary.belief_divergences[0].selected_belief_counts == (1, 2)
    assert "belief_ivo_key_in_safe" not in str(divergence_payload)
    assert "belief_mira_key_in_archive" not in str(divergence_payload)


def test_arbitration_recommendation_does_not_create_governance_objects() -> None:
    seed = _load_valid_bundle()
    bundle = _proposal_bundle(
        _frame("ivo", "proposal_ivo_observe", ActionIntent.OBSERVE),
        _frame("mira", "proposal_mira_observe", ActionIntent.OBSERVE),
    )

    recommendation = recommend_arbitration(bundle=seed, proposal_bundle=bundle)

    assert not isinstance(
        recommendation,
        EventCandidate | VerificationResult | CommittedEvent | StateDiff,
    )
    assert recommendation.creates_event_candidate is False
    assert recommendation.creates_verification_result is False
    assert recommendation.creates_committed_event is False
    assert recommendation.creates_state_diff is False
    assert recommendation.can_modify_world_state is False
    assert recommendation.can_mutate_canon is False


def test_governance_boundary_flags_are_schema_enforced_literal_false() -> None:
    frame = _frame("ivo", "proposal_ivo_observe", ActionIntent.OBSERVE)
    bundle = _proposal_bundle(frame)
    summary = bundle.summary()
    dynamics = MultiAgentDynamicsSummary(bundle_id=bundle.id)
    joint_candidate = JointIntentCandidate(
        id="joint_intent:proposal_ivo_observe",
        proposal_ids=("proposal_ivo_observe",),
        agent_ids=("ivo",),
        target_ids=("workshop_safe",),
        intent_labels=("observe",),
        reason_labels=("test_joint_candidate",),
    )
    recommendation = ProposalArbitrationRecommendation(bundle_id=bundle.id)

    _assert_rejects_flags(
        AgentProposalFrame,
        frame.model_dump(),
        "can_modify_world_state",
        "can_mutate_canon",
    )
    _assert_rejects_flags(
        ProposalBundleSummary,
        summary.model_dump(),
        "can_modify_world_state",
        "can_mutate_canon",
    )
    _assert_rejects_flags(
        MultiAgentProposalBundle,
        bundle.model_dump(),
        "can_modify_world_state",
        "can_mutate_canon",
        "emits_event_candidate",
        "emits_verification_result",
        "emits_committed_event",
        "emits_state_diff",
    )
    _assert_rejects_flags(
        MultiAgentDynamicsSummary,
        dynamics.model_dump(),
        "can_modify_world_state",
        "can_mutate_canon",
        "emits_event_candidate",
        "emits_verification_result",
        "emits_committed_event",
        "emits_state_diff",
    )
    _assert_rejects_flags(
        JointIntentCandidate,
        joint_candidate.model_dump(),
        "can_modify_world_state",
        "can_mutate_canon",
    )
    _assert_rejects_flags(
        ProposalArbitrationRecommendation,
        recommendation.model_dump(),
        "can_modify_world_state",
        "can_mutate_canon",
        "creates_event_candidate",
        "creates_verification_result",
        "creates_committed_event",
        "creates_state_diff",
    )


def _load_valid_bundle():
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


def _proposal_bundle(*frames: AgentProposalFrame) -> MultiAgentProposalBundle:
    return MultiAgentProposalBundle(
        id="bundle_r5_b3",
        step_id="step_r5_b3",
        scenario_id="scenario_r5_b3",
        active_agent_ids=tuple(frame.agent_id for frame in frames),
        frames=frames,
    )


def _frame(
    agent_id: str,
    proposal_id: str,
    intent: ActionIntent,
    *,
    target_location_id: str = "workshop_lane",
    target_entity_ids: tuple[str, ...] = ("workshop_safe",),
    target_resource_ids: tuple[str, ...] = (),
    confidence: float = 0.6,
    priority_score: float = 0.6,
) -> AgentProposalFrame:
    return AgentProposalFrame.from_proposal(
        _proposal(
            agent_id,
            proposal_id,
            intent,
            target_location_id=target_location_id,
            target_entity_ids=target_entity_ids,
        ),
        target_resource_ids=target_resource_ids,
        confidence=confidence,
        priority_score=priority_score,
        utility_score=0.6,
    ).model_copy(update={"context_source_ids": (f"ctx_{agent_id}",)})


def _proposal(
    agent_id: str,
    proposal_id: str,
    intent: ActionIntent,
    *,
    target_location_id: str = "workshop_lane",
    target_entity_ids: tuple[str, ...] = ("workshop_safe",),
) -> ActionProposal:
    return ActionProposal(
        id=proposal_id,
        proposer_agent_id=agent_id,
        intent=intent,
        rationale=f"{agent_id} proposes {intent.value}.",
        target_location_id=target_location_id,
        target_entity_ids=target_entity_ids,
        expected_outcome=f"{agent_id} action remains pre-governance evidence.",
    )


def _assert_rejects_flags(model_cls, payload: dict[str, object], *flag_names: str) -> None:
    for flag_name in flag_names:
        with pytest.raises(ValidationError):
            model_cls(**(payload | {flag_name: True}))
