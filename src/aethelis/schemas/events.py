from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from aethelis.schemas.common import AethelisModel, Identifier


class ActionIntent(StrEnum):
    INVESTIGATE = "investigate"
    MOVE = "move"
    NEGOTIATE = "negotiate"
    TRADE = "trade"
    REPAIR = "repair"
    GUARD = "guard"
    OBSERVE = "observe"
    DIALOGUE = "dialogue"


class EventCandidateStatus(StrEnum):
    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    REJECTED = "rejected"
    VERIFIED = "verified"


class VerificationDecision(StrEnum):
    COMMIT = "commit"
    REJECT = "reject"
    REVISE = "revise"
    PENDING_GATE = "pending_gate"


class PatchOperation(StrEnum):
    ADD = "add"
    UPDATE = "update"
    REMOVE = "remove"
    INCREMENT = "increment"
    DECREMENT = "decrement"
    APPEND = "append"
    MARK_STATUS = "mark_status"


class PatchTargetType(StrEnum):
    WORLD = "world"
    LOCATION = "location"
    ENTITY = "entity"
    RESOURCE = "resource"
    CANON_FACT = "canon_fact"
    AGENT_STATE = "agent_state"
    RELATIONSHIP = "relationship"


class ActionProposal(AethelisModel):
    """Agent output. It never directly mutates WorldState."""

    id: Identifier
    proposer_agent_id: Identifier
    intent: ActionIntent
    rationale: str = Field(min_length=1, max_length=220)
    target_location_id: Identifier | None = None
    target_entity_ids: tuple[Identifier, ...] = ()
    expected_outcome: str = Field(min_length=1, max_length=220)


class ActionProposalSummary(AethelisModel):
    action_proposal_id: Identifier
    proposer_agent_id: Identifier
    intent: ActionIntent
    target_location_id: Identifier | None = None
    target_entity_ids: tuple[Identifier, ...] = ()
    expected_outcome_summary: str = Field(min_length=1)
    contains_state_diff: bool = False
    contains_canon_mutation: bool = False
    generated_by: Identifier = "deterministic_fixture"

    @classmethod
    def from_proposal(
        cls,
        proposal: ActionProposal,
        *,
        generated_by: str = "deterministic_fixture",
    ) -> ActionProposalSummary:
        return cls(
            action_proposal_id=proposal.id,
            proposer_agent_id=proposal.proposer_agent_id,
            intent=proposal.intent,
            target_location_id=proposal.target_location_id,
            target_entity_ids=proposal.target_entity_ids,
            expected_outcome_summary=_summarize_text(proposal.expected_outcome),
            contains_state_diff=False,
            contains_canon_mutation=False,
            generated_by=generated_by,
        )


class EventCandidate(AethelisModel):
    """Runtime candidate derived from an ActionProposal.

    It deliberately does not contain StateDiff. Verification must happen before
    world-state changes are represented as a committed event diff.
    """

    id: Identifier
    source_action_proposal_id: Identifier
    actor_agent_id: Identifier
    summary: str = Field(min_length=1)
    status: EventCandidateStatus = EventCandidateStatus.PROPOSED
    involved_location_ids: tuple[Identifier, ...] = ()
    involved_entity_ids: tuple[Identifier, ...] = ()


class EventCandidateSummary(AethelisModel):
    event_candidate_id: Identifier
    source_action_proposal_id: Identifier
    actor_agent_id: Identifier
    status: EventCandidateStatus
    involved_location_ids: tuple[Identifier, ...] = ()
    involved_entity_ids: tuple[Identifier, ...] = ()
    candidate_kind: Identifier | None = None
    can_modify_world_state: bool = False
    predicted_state_diff_id: Identifier | None = None

    @classmethod
    def from_candidate(
        cls,
        candidate: EventCandidate,
        *,
        candidate_kind: str | None = None,
    ) -> EventCandidateSummary:
        return cls(
            event_candidate_id=candidate.id,
            source_action_proposal_id=candidate.source_action_proposal_id,
            actor_agent_id=candidate.actor_agent_id,
            status=candidate.status,
            involved_location_ids=candidate.involved_location_ids,
            involved_entity_ids=candidate.involved_entity_ids,
            candidate_kind=candidate_kind,
            can_modify_world_state=False,
            predicted_state_diff_id=None,
        )


class VerificationResult(AethelisModel):
    id: Identifier
    event_candidate_id: Identifier
    decision: VerificationDecision
    verifier: str = Field(min_length=1)
    checks: tuple[VerificationCheck, ...] = ()
    reasons: tuple[str, ...]
    risk_flags: tuple[str, ...] = ()
    rejected_claim_ids: tuple[Identifier, ...] = ()


class VerificationCheck(AethelisModel):
    name: str = Field(min_length=1)
    passed: bool
    message: str = Field(min_length=1)


class StatePatch(AethelisModel):
    operation: PatchOperation
    target_type: PatchTargetType
    target_id: Identifier
    path: str = Field(min_length=1)
    before: Any = None
    after: Any = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_patch_shape(self) -> StatePatch:
        if self.operation == PatchOperation.ADD and self.before is not None:
            raise ValueError("add operation requires before to be null")
        if self.operation == PatchOperation.ADD and self.after is None:
            raise ValueError("add operation requires after to be non-null")
        if self.operation == PatchOperation.REMOVE and self.before is None:
            raise ValueError("remove operation requires before to be non-null")
        if self.operation == PatchOperation.REMOVE and self.after is not None:
            raise ValueError("remove operation requires after to be null")
        if self.operation in {PatchOperation.UPDATE, PatchOperation.MARK_STATUS} and (
            self.before is None or self.after is None
        ):
            raise ValueError(f"{self.operation.value} requires before and after")
        if self.operation in {PatchOperation.INCREMENT, PatchOperation.DECREMENT}:
            if not isinstance(self.before, int | float) or not isinstance(self.after, int | float):
                raise ValueError(f"{self.operation.value} requires numeric before and after")
            if self.operation == PatchOperation.INCREMENT and self.after <= self.before:
                raise ValueError("increment requires after to be greater than before")
            if self.operation == PatchOperation.DECREMENT and self.after >= self.before:
                raise ValueError("decrement requires after to be less than before")
        if self.operation == PatchOperation.APPEND and self.after is None:
            raise ValueError("append operation requires after to be non-null")
        return self


class StateDiff(AethelisModel):
    """Controlled patch contract.

    A StateDiff may reference its source event candidate, but it must not be
    sourced directly from an ActionProposal. The committed_event_id is optional
    while a diff is being prepared, but CommittedEvent binds it.
    """

    id: Identifier
    source_event_candidate_id: Identifier | None = None
    committed_event_id: Identifier | None = None
    source_action_proposal_id: Identifier | None = None
    patches: tuple[StatePatch, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_boundary(self) -> StateDiff:
        if self.source_action_proposal_id is not None:
            raise ValueError("StateDiff cannot be sourced directly from ActionProposal")
        if self.source_event_candidate_id is None and self.committed_event_id is None:
            raise ValueError("StateDiff requires source_event_candidate_id or committed_event_id")
        return self


class CommittedEvent(AethelisModel):
    id: Identifier
    event_candidate_id: Identifier
    verification_result_id: Identifier
    summary: str = Field(min_length=1)
    tags: tuple[Identifier, ...] = ()
    state_diff: StateDiff

    @model_validator(mode="after")
    def validate_state_diff_binding(self) -> CommittedEvent:
        if self.state_diff.committed_event_id not in {None, self.id}:
            raise ValueError("state_diff.committed_event_id must match committed event id")
        if self.state_diff.source_event_candidate_id not in {None, self.event_candidate_id}:
            raise ValueError("state_diff.source_event_candidate_id must match event candidate id")
        return self


def _summarize_text(value: str, limit: int = 160) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."
