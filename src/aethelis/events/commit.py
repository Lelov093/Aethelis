from __future__ import annotations

from aethelis.runtime.scenario_matrix import StateDiffContract, get_state_diff_contract
from aethelis.schemas.events import (
    CommittedEvent,
    EventCandidate,
    PatchOperation,
    PatchTargetType,
    StateDiff,
    StatePatch,
    VerificationDecision,
    VerificationResult,
)
from aethelis.schemas.world import WorldState


def build_committed_event_from_verification(
    *,
    candidate: EventCandidate,
    verification: VerificationResult,
    scenario_id: str,
    world_state: WorldState | None = None,
) -> CommittedEvent | None:
    """Build CommittedEvent only after VerificationResult(commit).

    This is the StateDiff generation boundary. It accepts EventCandidate plus
    VerificationResult only; ActionProposal is intentionally not part of this
    API.
    """

    if verification.decision != VerificationDecision.COMMIT:
        return None
    contract = get_state_diff_contract(scenario_id)
    if contract is None:
        return None
    event_id = f"committed_{candidate.id}"
    state_diff = _state_diff_for_contract(
        event_id=event_id,
        candidate=candidate,
        contract=contract,
        world_state=world_state,
    )
    return CommittedEvent(
        id=event_id,
        event_candidate_id=candidate.id,
        verification_result_id=verification.id,
        summary=contract.summary,
        state_diff=state_diff,
    )


def _state_diff_for_contract(
    *,
    event_id: str,
    candidate: EventCandidate,
    contract: StateDiffContract,
    world_state: WorldState | None = None,
) -> StateDiff:
    before, after = _state_transition_values(contract=contract, world_state=world_state)
    return StateDiff(
        id=f"diff_{event_id}",
        source_event_candidate_id=candidate.id,
        committed_event_id=event_id,
        patches=(
            StatePatch(
                operation=contract.operation,
                target_type=PatchTargetType.RESOURCE,
                target_id=contract.target_id,
                path=contract.path,
                before=before,
                after=after,
                reason=contract.reason,
            ),
        ),
    )


def _state_transition_values(
    *,
    contract: StateDiffContract,
    world_state: WorldState | None,
) -> tuple[object, object]:
    if world_state is None:
        return contract.before, contract.after

    resource = next(
        (item for item in world_state.resources if item.id == contract.target_id),
        None,
    )
    if resource is None:
        return contract.before, contract.after

    if contract.operation == PatchOperation.APPEND:
        current = list(resource.discovery_state.discovered_by_agent_ids)
        append_values = contract.after if isinstance(contract.after, list) else []
        after = list(dict.fromkeys([*current, *append_values]))
        return current, after

    if contract.operation in {PatchOperation.INCREMENT, PatchOperation.DECREMENT}:
        if not isinstance(contract.before, int) or not isinstance(contract.after, int):
            return contract.before, contract.after
        delta = contract.after - contract.before
        return resource.quantity, resource.quantity + delta

    return contract.before, contract.after
