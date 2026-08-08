from __future__ import annotations

from pydantic import Field

from aethelis.algorithms.runtime_features import player_assimilation_score
from aethelis.runtime.scenario_matrix import get_player_input_fixture_contract
from aethelis.schemas.common import AethelisModel, ConfidenceBand, Identifier, RecordStatus
from aethelis.schemas.events import (
    EventCandidate,
    EventCandidateStatus,
    VerificationDecision,
    VerificationResult,
)
from aethelis.schemas.ledger import BeliefCandidate
from aethelis.schemas.player_input import (
    PlayerInputKind,
    PlayerInputRecord,
    PlayerInputRoute,
    RoutedPlayerInput,
)


class PlayerClaimAssessment(AethelisModel):
    claim_id: Identifier
    player_id: Identifier
    claim: str = Field(min_length=1)
    verification_result: VerificationResult
    routed_input: RoutedPlayerInput | None = None
    belief_candidate: BeliefCandidate | None = None
    canon_updated: bool = False
    state_diff_id: Identifier | None = None


def route_player_input(record: PlayerInputRecord) -> RoutedPlayerInput:
    """Route player input without mutating Canon or WorldState."""

    risk = _player_input_risk(record)
    risk_flag = _risk_flag(risk)
    if record.kind == PlayerInputKind.CLAIM:
        belief_candidate = _belief_candidate(record, status=RecordStatus.REJECTED)
        verification = _verification(
            record,
            decision=VerificationDecision.REJECT,
            reasons=("Player claim requires assimilation and verification before canon mutation.",),
            risk_flags=("unverified_player_claim",),
            rejected_claim_ids=(record.id,),
        )
        return RoutedPlayerInput(
            input_id=record.id,
            player_id=record.player_id,
            input_kind=record.kind,
            route=PlayerInputRoute.REJECTED_CLAIM,
            belief_candidate=belief_candidate,
            verification_result=verification,
            safety_flags=("player_claim_not_canon", "belief_candidate_not_canon", risk_flag),
        )
    if record.kind in {PlayerInputKind.REQUEST, PlayerInputKind.ACTION}:
        candidate = _event_candidate(record)
        verification = _verification(
            record,
            event_candidate_id=candidate.id,
            decision=VerificationDecision.PENDING_GATE,
            reasons=(
                "World-changing player input must enter EventCandidate verification before commit.",
            ),
            risk_flags=("player_input_requires_gate",),
        )
        return RoutedPlayerInput(
            input_id=record.id,
            player_id=record.player_id,
            input_kind=record.kind,
            route=PlayerInputRoute.EVENT_CANDIDATE,
            event_candidate=candidate,
            verification_result=verification,
            safety_flags=("player_input_event_candidate_only", risk_flag),
        )
    if record.kind == PlayerInputKind.DECEPTION:
        belief_candidate = _belief_candidate(record, status=RecordStatus.REJECTED)
        verification = _verification(
            record,
            decision=VerificationDecision.REJECT,
            reasons=("Contradictory or deceptive player input is rejected before canon mutation.",),
            risk_flags=("player_deception_rejected",),
            rejected_claim_ids=(record.id,),
        )
        return RoutedPlayerInput(
            input_id=record.id,
            player_id=record.player_id,
            input_kind=record.kind,
            route=PlayerInputRoute.REJECTED_CLAIM,
            belief_candidate=belief_candidate,
            verification_result=verification,
            safety_flags=("player_deception_not_canon", risk_flag),
        )

    verification = _verification(
        record,
        decision=VerificationDecision.REJECT,
        reasons=("Meta player input is not a world-changing event.",),
        risk_flags=("meta_input_no_world_mutation",),
    )
    return RoutedPlayerInput(
        input_id=record.id,
        player_id=record.player_id,
        input_kind=record.kind,
        route=PlayerInputRoute.META_NOOP,
        verification_result=verification,
        safety_flags=("meta_input_noop", risk_flag),
    )


def assess_player_claim(
    *,
    claim_id: str,
    player_id: str,
    claim: str,
) -> PlayerClaimAssessment:
    """Govern a player claim without writing it into canon."""

    record = PlayerInputRecord(
        id=claim_id,
        player_id=player_id,
        kind=PlayerInputKind.CLAIM,
        text=claim,
    )
    routed = route_player_input(record)
    return PlayerClaimAssessment(
        claim_id=claim_id,
        player_id=player_id,
        claim=claim,
        verification_result=routed.verification_result,
        routed_input=routed,
        belief_candidate=routed.belief_candidate,
        canon_updated=routed.canon_updated,
        state_diff_id=routed.state_diff_id,
    )


def route_player_input_scenario(*, scenario_id: str, player_id: str) -> RoutedPlayerInput:
    """Route a deterministic player-input scenario without applying world changes."""
    contract = get_player_input_fixture_contract(scenario_id)
    return route_player_input(
        PlayerInputRecord(
            id=contract.input_id,
            player_id=player_id,
            kind=contract.kind,
            text=contract.text,
            target_location_id=contract.target_location_id,
            target_entity_ids=contract.target_entity_ids,
            source_scenario_id=scenario_id,
        )
    )


def _belief_candidate(record: PlayerInputRecord, *, status: RecordStatus) -> BeliefCandidate:
    return BeliefCandidate(
        id=f"belief_candidate_{record.id}",
        source_type=f"player_input:{record.kind.value}",
        source_id=record.id,
        claim=record.text,
        confidence=ConfidenceBand.LOW,
        status=status,
        subject_ids=record.target_entity_ids,
        trace_reference_id=record.source_scenario_id,
        canon_updated=False,
        world_state_modified=False,
    )


def _event_candidate(record: PlayerInputRecord) -> EventCandidate:
    return EventCandidate(
        id=f"candidate_{record.id}",
        source_action_proposal_id=f"player_input:{record.id}",
        actor_agent_id=record.player_id,
        summary=f"Player input routed for verification: {_summarize(record.text)}",
        status=EventCandidateStatus.UNDER_REVIEW,
        involved_location_ids=(
            (record.target_location_id,) if record.target_location_id is not None else ()
        ),
        involved_entity_ids=record.target_entity_ids,
    )


def _verification(
    record: PlayerInputRecord,
    *,
    decision: VerificationDecision,
    reasons: tuple[str, ...],
    risk_flags: tuple[str, ...],
    event_candidate_id: str | None = None,
    rejected_claim_ids: tuple[str, ...] = (),
) -> VerificationResult:
    return VerificationResult(
        id=f"verification_{record.id}",
        event_candidate_id=event_candidate_id or record.id,
        decision=decision,
        verifier="player_input_router_v0",
        checks=(),
        reasons=reasons,
        risk_flags=risk_flags,
        rejected_claim_ids=rejected_claim_ids,
    )


def _summarize(value: str, limit: int = 120) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _player_input_risk(record: PlayerInputRecord) -> float:
    claim_risk = 1.0 if record.kind in {PlayerInputKind.CLAIM, PlayerInputKind.DECEPTION} else 0.2
    permission = 0.0 if record.kind == PlayerInputKind.DECEPTION else 1.0
    verification = 1.0 if record.kind in {PlayerInputKind.REQUEST, PlayerInputKind.ACTION} else 0.6
    assimilation = player_assimilation_score(
        permission_gate=permission,
        claim_risk=claim_risk,
        verification_requirement=verification,
    )
    return 1.0 - assimilation


def _risk_flag(risk: float) -> str:
    return f"player_input_risk_{round(risk * 100):03d}"
