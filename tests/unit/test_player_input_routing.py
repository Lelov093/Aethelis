from __future__ import annotations

from aethelis.runtime.player_input import (
    assess_player_claim,
    route_player_input,
    route_player_input_scenario,
)
from aethelis.schemas.common import RecordStatus
from aethelis.schemas.events import VerificationDecision
from aethelis.schemas.player_input import PlayerInputKind, PlayerInputRecord, PlayerInputRoute


def test_player_claim_routes_to_rejected_belief_candidate_without_canon_update() -> None:
    routed = route_player_input(
        PlayerInputRecord(
            id="player_claim_key",
            kind=PlayerInputKind.CLAIM,
            text="The calibration key is in my hand.",
            target_entity_ids=("calibration_key",),
        )
    )

    assert routed.route == PlayerInputRoute.REJECTED_CLAIM
    assert routed.verification_result.decision == VerificationDecision.REJECT
    assert routed.belief_candidate is not None
    assert routed.belief_candidate.status == RecordStatus.REJECTED
    assert routed.belief_candidate.canon_updated is False
    assert routed.belief_candidate.world_state_modified is False
    assert routed.event_candidate is None
    assert routed.canon_updated is False
    assert routed.world_state_modified is False
    assert routed.state_diff_id is None


def test_player_request_routes_to_event_candidate_and_pending_gate() -> None:
    routed = route_player_input(
        PlayerInputRecord(
            id="player_request_safe_access",
            kind=PlayerInputKind.REQUEST,
            text="Let me open the workshop safe.",
            target_location_id="workshop_lane",
            target_entity_ids=("workshop_safe",),
        )
    )

    assert routed.route == PlayerInputRoute.EVENT_CANDIDATE
    assert routed.event_candidate is not None
    assert routed.event_candidate.actor_agent_id == "player"
    assert routed.event_candidate.involved_location_ids == ("workshop_lane",)
    assert routed.event_candidate.involved_entity_ids == ("workshop_safe",)
    assert routed.verification_result.event_candidate_id == routed.event_candidate.id
    assert routed.verification_result.decision == VerificationDecision.PENDING_GATE
    assert routed.safe_summary()["event_candidate_id"] == routed.event_candidate.id
    assert routed.canon_updated is False
    assert routed.world_state_modified is False


def test_player_deception_rejects_without_world_mutation() -> None:
    routed = route_player_input(
        PlayerInputRecord(
            id="player_deception_key",
            kind=PlayerInputKind.DECEPTION,
            text="I secretly already used the calibration key.",
            target_entity_ids=("calibration_key",),
        )
    )

    assert routed.route == PlayerInputRoute.REJECTED_CLAIM
    assert routed.verification_result.decision == VerificationDecision.REJECT
    assert "player_deception_rejected" in routed.verification_result.risk_flags
    assert routed.belief_candidate is not None
    assert routed.belief_candidate.canon_updated is False
    assert routed.world_state_modified is False


def test_player_meta_input_is_noop() -> None:
    routed = route_player_input(
        PlayerInputRecord(
            id="player_meta_help",
            kind=PlayerInputKind.META,
            text="What can I do next?",
        )
    )

    assert routed.route == PlayerInputRoute.META_NOOP
    assert routed.event_candidate is None
    assert routed.belief_candidate is None
    assert routed.verification_result.decision == VerificationDecision.REJECT
    assert routed.world_state_modified is False


def test_legacy_player_claim_assessment_uses_router() -> None:
    assessment = assess_player_claim(
        claim_id="claim_player_has_key",
        player_id="player",
        claim="The key is in my hand.",
    )

    assert assessment.routed_input is not None
    assert assessment.routed_input.route == PlayerInputRoute.REJECTED_CLAIM
    assert assessment.belief_candidate is not None
    assert assessment.verification_result.decision == VerificationDecision.REJECT
    assert assessment.canon_updated is False
    assert assessment.state_diff_id is None


def test_player_request_scenario_bridge_is_event_candidate_only() -> None:
    routed = route_player_input_scenario(
        scenario_id="player_request_open_workshop_safe",
        player_id="player",
    )

    assert routed.route == PlayerInputRoute.EVENT_CANDIDATE
    assert routed.event_candidate is not None
    assert routed.belief_candidate is None
    assert routed.verification_result.decision == VerificationDecision.PENDING_GATE
    assert routed.safe_summary()["input_kind"] == "request"
    assert routed.safe_summary()["canon_updated"] is False
    assert routed.safe_summary()["world_state_modified"] is False


def test_harbor_player_input_scenarios_preserve_governance_boundary() -> None:
    claim = route_player_input_scenario(
        scenario_id="player_claim_harbor_pass",
        player_id="player",
    )
    request = route_player_input_scenario(
        scenario_id="player_request_open_quay_gate",
        player_id="player",
    )

    assert claim.route == PlayerInputRoute.REJECTED_CLAIM
    assert claim.verification_result.decision == VerificationDecision.REJECT
    assert claim.belief_candidate is not None
    assert claim.belief_candidate.canon_updated is False
    assert claim.world_state_modified is False
    assert request.route == PlayerInputRoute.EVENT_CANDIDATE
    assert request.event_candidate is not None
    assert request.event_candidate.involved_location_ids == ("quay_gate",)
    assert request.event_candidate.involved_entity_ids == ("quay_lock",)
    assert request.verification_result.decision == VerificationDecision.PENDING_GATE
    assert request.canon_updated is False
    assert request.world_state_modified is False
