from __future__ import annotations

import pytest
from pydantic import ValidationError

from aethelis.schemas.events import (
    ActionProposal,
    EventCandidate,
    PatchOperation,
    PatchTargetType,
    StateDiff,
    StatePatch,
)


def test_action_proposal_rejects_invalid_enum() -> None:
    with pytest.raises(ValidationError):
        ActionProposal.model_validate(
            {
                "id": "proposal_1",
                "proposer_agent_id": "mira",
                "intent": "rewrite_world",
                "rationale": "not allowed",
                "expected_outcome": "world changes directly",
            }
        )


def test_action_proposal_text_fields_are_bounded() -> None:
    with pytest.raises(ValidationError):
        ActionProposal(
            id="proposal_too_long",
            proposer_agent_id="ivo",
            intent="investigate",
            rationale="x" * 221,
            expected_outcome="Inspect the safe.",
        )
    with pytest.raises(ValidationError):
        ActionProposal(
            id="proposal_too_long_outcome",
            proposer_agent_id="ivo",
            intent="investigate",
            rationale="Inspect the safe.",
            expected_outcome="x" * 221,
        )


def test_event_candidate_has_no_state_diff_field() -> None:
    with pytest.raises(ValidationError):
        EventCandidate.model_validate(
            {
                "id": "candidate_1",
                "source_action_proposal_id": "proposal_1",
                "actor_agent_id": "mira",
                "summary": "Mira searches the archive.",
                "state_diff": {"id": "diff_1", "patches": []},
            }
        )


def test_state_diff_rejects_direct_action_proposal_source() -> None:
    patch = StatePatch(
        operation=PatchOperation.UPDATE,
        target_type=PatchTargetType.RESOURCE,
        target_id="stabilizer_parts",
        path="/quantity",
        before=3,
        after=2,
        reason="one part reserved",
    )

    with pytest.raises(ValidationError, match="ActionProposal"):
        StateDiff(
            id="diff_bad",
            source_action_proposal_id="proposal_1",
            patches=(patch,),
        )


def test_state_patch_requires_operation_shape() -> None:
    with pytest.raises(ValidationError, match="increment"):
        StatePatch(
            operation=PatchOperation.INCREMENT,
            target_type=PatchTargetType.RESOURCE,
            target_id="stabilizer_parts",
            path="/quantity",
            before=3,
            after=2,
            reason="invalid increase",
        )
