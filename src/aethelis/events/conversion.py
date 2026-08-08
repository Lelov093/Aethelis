from __future__ import annotations

from aethelis.algorithms.runtime_features import candidate_quality_score
from aethelis.schemas.events import (
    ActionProposal,
    EventCandidate,
    EventCandidateStatus,
    EventCandidateSummary,
    VerificationCheck,
    VerificationDecision,
    VerificationResult,
)


class CandidateBehaviorRoute:
    UNDER_REVIEW = "under_review"
    HOLD = "hold"
    REVISE = "revise"
    REJECTED = "rejected"


def action_proposal_to_event_candidate(
    proposal: ActionProposal,
    *,
    scenario_id: str,
    candidate_kind: str | None = None,
) -> EventCandidate:
    route, quality = candidate_behavior_route(proposal)
    status = (
        EventCandidateStatus.REJECTED
        if route == CandidateBehaviorRoute.REJECTED
        else EventCandidateStatus.UNDER_REVIEW
    )
    return EventCandidate(
        id=f"candidate_{scenario_id}_{proposal.proposer_agent_id}",
        source_action_proposal_id=proposal.id,
        actor_agent_id=proposal.proposer_agent_id,
        summary=(
            f"{proposal.proposer_agent_id} proposes to {proposal.intent.value}: "
            f"{proposal.expected_outcome} "
            f"[candidate_quality={quality:.2f}; route={route}]"
        ),
        status=status,
        involved_location_ids=(
            (proposal.target_location_id,) if proposal.target_location_id is not None else ()
        ),
        involved_entity_ids=proposal.target_entity_ids,
    )


def event_candidate_summary(
    candidate: EventCandidate,
    *,
    candidate_kind: str | None = None,
) -> EventCandidateSummary:
    return EventCandidateSummary.from_candidate(
        candidate,
        candidate_kind=candidate_kind,
    )


def candidate_behavior_route(proposal: ActionProposal) -> tuple[str, float]:
    quality = _candidate_quality(proposal)
    risk = _boundary_risk(proposal)
    target_present = bool(proposal.target_location_id or proposal.target_entity_ids)
    if risk >= 1.0:
        return CandidateBehaviorRoute.REJECTED, quality
    if quality < 0.45:
        return CandidateBehaviorRoute.HOLD, quality
    if not target_present or quality < 0.65:
        return CandidateBehaviorRoute.REVISE, quality
    return CandidateBehaviorRoute.UNDER_REVIEW, quality


def candidate_gate_verification_result(
    candidate: EventCandidate,
    *,
    route: str,
    quality: float,
) -> VerificationResult:
    decision = (
        VerificationDecision.REJECT
        if route == CandidateBehaviorRoute.REJECTED
        else VerificationDecision.PENDING_GATE
        if route == CandidateBehaviorRoute.HOLD
        else VerificationDecision.REVISE
    )
    return VerificationResult(
        id=f"verification_{candidate.id}_candidate_gate",
        event_candidate_id=candidate.id,
        decision=decision,
        verifier="candidate_behavior_quality_gate",
        checks=(
            VerificationCheck(
                name="candidate_quality_route",
                passed=False,
                message=f"Candidate quality route={route}; quality={quality:.2f}.",
            ),
        ),
        reasons=(f"Candidate behavior quality gate routed candidate to {route}.",),
        risk_flags=(f"candidate_route_{route}",),
    )


def _candidate_quality(proposal: ActionProposal) -> float:
    schema = 1.0 if proposal.id and proposal.proposer_agent_id and proposal.expected_outcome else 0.0
    precondition = 1.0 if proposal.target_location_id or proposal.target_entity_ids else 0.15
    state_diff = 0.85 if proposal.target_entity_ids else 0.40
    risk = _boundary_risk(proposal)
    return candidate_quality_score(
        schema_completeness=schema,
        precondition_fit=precondition,
        state_diff_plausibility=state_diff,
        boundary_risk=risk,
    )


def _boundary_risk(proposal: ActionProposal) -> float:
    text = f"{proposal.rationale} {proposal.expected_outcome}".lower()
    hard_markers = (
        "write a state diff",
        "force a statediff",
        "force a state diff",
        "direct state diff",
        "rewrite canon",
        "mutate canon",
    )
    return 1.0 if any(marker in text for marker in hard_markers) else 0.0