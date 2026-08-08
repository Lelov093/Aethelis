from __future__ import annotations

from pathlib import Path

import pytest

from aethelis.agents.context import build_agent_context
from aethelis.events.conversion import action_proposal_to_event_candidate
from aethelis.runtime.single_step import build_committed_event
from aethelis.runtime.state_apply import ControlledStateDiffApplier
from aethelis.runtime.state_replay import GovernedStateReplayer
from aethelis.runtime.state_store import RuntimeStateStore
from aethelis.schemas.common import RecordStatus
from aethelis.schemas.events import (
    ActionIntent,
    ActionProposal,
    EventCandidate,
    PatchOperation,
    PatchTargetType,
    StateDiff,
    StatePatch,
    VerificationDecision,
)
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator
from aethelis.verification.deterministic import DeterministicVerifier

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


def ivo_proposal() -> ActionProposal:
    return ActionProposal(
        id="proposal_inspect_workshop_safe_ivo",
        proposer_agent_id="ivo",
        intent=ActionIntent.INVESTIGATE,
        rationale="Ivo has a lawful private reason to inspect his own workshop safe.",
        target_location_id="workshop_lane",
        target_entity_ids=("workshop_safe",),
        expected_outcome="Inspect the workshop safe for the calibration key.",
    )


def committed_event_and_verification():
    bundle = load_valid_bundle()
    observation, cognition = build_agent_context(
        bundle,
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
    )
    proposal = ivo_proposal()
    candidate = action_proposal_to_event_candidate(
        proposal,
        scenario_id="inspect_workshop_safe",
    )
    verification = DeterministicVerifier().verify(
        bundle=bundle,
        observation=observation,
        cognition=cognition,
        proposal=proposal,
        candidate=candidate,
        scenario_id="inspect_workshop_safe",
    )
    committed_event = build_committed_event(
        candidate=candidate,
        verification=verification,
        scenario_id="inspect_workshop_safe",
    )
    assert committed_event is not None
    return bundle, committed_event, verification


def committed_event_with_patches(committed_event, patches):
    state_diff = StateDiff(
        id=f"{committed_event.state_diff.id}_test",
        source_event_candidate_id=committed_event.event_candidate_id,
        committed_event_id=committed_event.id,
        patches=tuple(patches),
    )
    return committed_event.model_copy(update={"state_diff": state_diff})


def test_apply_returns_world_state_copy_without_mutating_seed() -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    original_key = next(
        resource for resource in bundle.world.resources if resource.id == "calibration_key"
    )

    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=verification,
    )
    applied_key = next(
        resource for resource in applied_world.resources if resource.id == "calibration_key"
    )

    assert report.applied
    assert report.applied_patch_count == 1
    assert report.patch_results[0].before == []
    assert report.patch_results[0].after == ["ivo"]
    assert applied_world is not bundle.world
    assert original_key.discovery_state.discovered_by_agent_ids == ()
    assert applied_key.discovery_state.discovered_by_agent_ids == ("ivo",)


@pytest.mark.parametrize(
    ("operation", "before", "after"),
    [
        (PatchOperation.INCREMENT, 3, 4),
        (PatchOperation.DECREMENT, 3, 2),
    ],
)
def test_resource_quantity_delta_patch_applies(
    operation: PatchOperation,
    before: int,
    after: int,
) -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    committed_event = committed_event_with_patches(
        committed_event,
        (
            StatePatch(
                operation=operation,
                target_type=PatchTargetType.RESOURCE,
                target_id="stabilizer_parts",
                path="/resource/stabilizer_parts/quantity",
                before=before,
                after=after,
                reason="test quantity delta",
            ),
        ),
    )

    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=verification,
    )
    resource = next(
        resource for resource in applied_world.resources if resource.id == "stabilizer_parts"
    )

    assert report.applied
    assert report.applied_patch_count == 1
    assert resource.quantity == after


def test_entity_status_mark_patch_applies() -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    committed_event = committed_event_with_patches(
        committed_event,
        (
            StatePatch(
                operation=PatchOperation.MARK_STATUS,
                target_type=PatchTargetType.ENTITY,
                target_id="workshop_safe",
                path="/entity/workshop_safe/status",
                before="active",
                after="inactive",
                reason="test entity status mark",
            ),
        ),
    )

    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=verification,
    )
    entity = next(entity for entity in applied_world.entities if entity.id == "workshop_safe")
    original = next(entity for entity in bundle.world.entities if entity.id == "workshop_safe")

    assert report.applied
    assert report.applied_patch_count == 1
    assert report.patch_results[0].before == "active"
    assert report.patch_results[0].after == "inactive"
    assert original.status == RecordStatus.ACTIVE
    assert entity.status == RecordStatus.INACTIVE


def test_invalid_entity_status_patch_rejects_safely() -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    committed_event = committed_event_with_patches(
        committed_event,
        (
            StatePatch(
                operation=PatchOperation.MARK_STATUS,
                target_type=PatchTargetType.ENTITY,
                target_id="workshop_safe",
                path="/entity/workshop_safe/status",
                before="active",
                after="missing_status",
                reason="test invalid entity status",
            ),
        ),
    )

    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=verification,
    )
    entity = next(entity for entity in applied_world.entities if entity.id == "workshop_safe")

    assert applied_world is bundle.world
    assert not report.applied
    assert report.applied_patch_count == 0
    assert report.errors == ("invalid_patch_after: entity status must be a valid RecordStatus",)
    assert entity.status == RecordStatus.ACTIVE


def test_missing_entity_status_patch_rejects_safely() -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    committed_event = committed_event_with_patches(
        committed_event,
        (
            StatePatch(
                operation=PatchOperation.MARK_STATUS,
                target_type=PatchTargetType.ENTITY,
                target_id="missing_entity",
                path="/entity/missing_entity/status",
                before="active",
                after="inactive",
                reason="test missing entity",
            ),
        ),
    )

    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=verification,
    )

    assert applied_world is bundle.world
    assert not report.applied
    assert report.errors == ("entity_target_not_found",)


def test_entity_status_before_mismatch_aborts_safely() -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    committed_event = committed_event_with_patches(
        committed_event,
        (
            StatePatch(
                operation=PatchOperation.MARK_STATUS,
                target_type=PatchTargetType.ENTITY,
                target_id="workshop_safe",
                path="/entity/workshop_safe/status",
                before="inactive",
                after="active",
                reason="test status before mismatch",
            ),
        ),
    )

    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=verification,
    )
    entity = next(entity for entity in applied_world.entities if entity.id == "workshop_safe")

    assert applied_world is bundle.world
    assert not report.applied
    assert "patch_before_mismatch" in report.errors[0]
    assert entity.status == RecordStatus.ACTIVE


def test_unsupported_patch_aborts_all_staged_changes() -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    committed_event = committed_event_with_patches(
        committed_event,
        (
            StatePatch(
                operation=PatchOperation.INCREMENT,
                target_type=PatchTargetType.RESOURCE,
                target_id="stabilizer_parts",
                path="/resource/stabilizer_parts/quantity",
                before=3,
                after=4,
                reason="test staged quantity delta",
            ),
            StatePatch(
                operation=PatchOperation.MARK_STATUS,
                target_type=PatchTargetType.RESOURCE,
                target_id="stabilizer_parts",
                path="/resource/stabilizer_parts/status",
                before="active",
                after="inactive",
                reason="unsupported in V0.2 first batch",
            ),
        ),
    )

    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=verification,
    )
    original = next(
        resource for resource in bundle.world.resources if resource.id == "stabilizer_parts"
    )
    after_apply = next(
        resource for resource in applied_world.resources if resource.id == "stabilizer_parts"
    )

    assert applied_world is bundle.world
    assert not report.applied
    assert report.applied_patch_count == 0
    assert report.skipped_patch_count == 2
    assert "unsupported_patch_operation" in report.errors[0]
    assert report.patch_results[0].error == "aborted_due_to_patch_error"
    assert original.quantity == 3
    assert after_apply.quantity == 3


def test_entity_status_and_failing_patch_abort_all_staged_changes() -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    committed_event = committed_event_with_patches(
        committed_event,
        (
            StatePatch(
                operation=PatchOperation.MARK_STATUS,
                target_type=PatchTargetType.ENTITY,
                target_id="workshop_safe",
                path="/entity/workshop_safe/status",
                before="active",
                after="inactive",
                reason="test entity status mark",
            ),
            StatePatch(
                operation=PatchOperation.INCREMENT,
                target_type=PatchTargetType.RESOURCE,
                target_id="stabilizer_parts",
                path="/resource/stabilizer_parts/quantity",
                before=999,
                after=1000,
                reason="test failing quantity patch",
            ),
        ),
    )

    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=verification,
    )
    entity = next(entity for entity in applied_world.entities if entity.id == "workshop_safe")
    resource = next(
        resource for resource in applied_world.resources if resource.id == "stabilizer_parts"
    )

    assert applied_world is bundle.world
    assert not report.applied
    assert report.applied_patch_count == 0
    assert report.skipped_patch_count == 2
    assert report.patch_results[0].error == "aborted_due_to_patch_error"
    assert "patch_before_mismatch" in report.errors[0]
    assert entity.status == RecordStatus.ACTIVE
    assert resource.quantity == 3


def test_state_diff_cannot_be_applied_directly() -> None:
    _, committed_event, _ = committed_event_and_verification()

    with pytest.raises(TypeError, match="StateDiff cannot be applied"):
        ControlledStateDiffApplier().apply_state_diff(committed_event.state_diff)


def test_action_proposal_and_event_candidate_cannot_trigger_apply() -> None:
    proposal = ActionProposal(
        id="proposal_test",
        proposer_agent_id="ivo",
        intent=ActionIntent.INVESTIGATE,
        rationale="test",
        target_location_id="workshop_lane",
        target_entity_ids=("workshop_safe",),
        expected_outcome="test",
    )
    candidate = EventCandidate(
        id="candidate_test",
        source_action_proposal_id=proposal.id,
        actor_agent_id="ivo",
        summary="test",
    )

    with pytest.raises(TypeError, match="ActionProposal cannot trigger"):
        ControlledStateDiffApplier().apply_action_proposal(proposal)
    with pytest.raises(TypeError, match="EventCandidate cannot trigger"):
        ControlledStateDiffApplier().apply_event_candidate(candidate)


@pytest.mark.parametrize(
    "decision",
    [
        VerificationDecision.REJECT,
        VerificationDecision.REVISE,
        VerificationDecision.PENDING_GATE,
    ],
)
def test_non_commit_decisions_do_not_apply(decision: VerificationDecision) -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    non_commit_verification = verification.model_copy(update={"decision": decision})

    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=non_commit_verification,
    )

    assert applied_world is bundle.world
    assert not report.applied
    assert decision.value in report.errors[0]


def test_non_commit_decision_does_not_apply_entity_status_mark() -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    committed_event = committed_event_with_patches(
        committed_event,
        (
            StatePatch(
                operation=PatchOperation.MARK_STATUS,
                target_type=PatchTargetType.ENTITY,
                target_id="workshop_safe",
                path="/entity/workshop_safe/status",
                before="active",
                after="inactive",
                reason="test non-commit entity status",
            ),
        ),
    )
    non_commit_verification = verification.model_copy(
        update={"decision": VerificationDecision.PENDING_GATE}
    )

    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=non_commit_verification,
    )
    entity = next(entity for entity in applied_world.entities if entity.id == "workshop_safe")

    assert applied_world is bundle.world
    assert not report.applied
    assert "pending_gate" in report.errors[0]
    assert entity.status == RecordStatus.ACTIVE


def test_mismatched_verification_id_does_not_apply() -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    mismatched_verification = verification.model_copy(update={"id": "verification_other"})

    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=mismatched_verification,
    )

    assert applied_world is bundle.world
    assert not report.applied
    assert "does not match" in report.errors[0]


def test_runtime_state_store_records_apply_journal() -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=committed_event,
        verification_result=verification,
    )

    store, journaled_report = RuntimeStateStore(world_state=bundle.world).record_apply_result(
        world_state=applied_world,
        report=report,
        verification_result_id=verification.id,
    )

    assert journaled_report.journal_entry_id is not None
    assert len(store.apply_journal) == 1
    assert store.apply_journal[0].applied
    assert store.apply_journal[0].committed_event_id == committed_event.id


def test_replay_committed_event_is_deterministic_and_journaled() -> None:
    bundle, committed_event, verification = committed_event_and_verification()

    first_store, first_report = GovernedStateReplayer().replay(
        store=RuntimeStateStore(world_state=bundle.world),
        committed_event=committed_event,
        verification_result=verification,
    )
    second_store, second_report = GovernedStateReplayer().replay(
        store=RuntimeStateStore(world_state=bundle.world),
        committed_event=committed_event,
        verification_result=verification,
    )

    assert first_report.replayed
    assert second_report.replayed
    assert first_report.safe_dict() == second_report.safe_dict()
    assert first_store.world_state == second_store.world_state
    assert len(first_store.replay_journal) == 1


def test_replay_reject_decision_does_not_apply() -> None:
    bundle, committed_event, verification = committed_event_and_verification()
    reject_verification = verification.model_copy(update={"decision": VerificationDecision.REJECT})

    store, report = GovernedStateReplayer().replay(
        store=RuntimeStateStore(world_state=bundle.world),
        committed_event=committed_event,
        verification_result=reject_verification,
    )

    assert not report.replayed
    assert "not commit" in report.errors[0]
    assert store.world_state is bundle.world
    assert len(store.replay_journal) == 1


def test_direct_replay_inputs_are_rejected() -> None:
    _, committed_event, _ = committed_event_and_verification()
    proposal = ActionProposal(
        id="proposal_replay_test",
        proposer_agent_id="ivo",
        intent=ActionIntent.INVESTIGATE,
        rationale="test",
        target_location_id="workshop_lane",
        target_entity_ids=("workshop_safe",),
        expected_outcome="test",
    )
    candidate = EventCandidate(
        id="candidate_replay_test",
        source_action_proposal_id=proposal.id,
        actor_agent_id="ivo",
        summary="test",
    )
    replayer = GovernedStateReplayer()

    with pytest.raises(TypeError, match="StateDiff cannot be replayed"):
        replayer.replay_state_diff(committed_event.state_diff)
    with pytest.raises(TypeError, match="ActionProposal cannot trigger"):
        replayer.replay_action_proposal(proposal)
    with pytest.raises(TypeError, match="EventCandidate cannot trigger"):
        replayer.replay_event_candidate(candidate)


def test_rollback_is_explicitly_unsupported() -> None:
    bundle, committed_event, verification = committed_event_and_verification()

    store, report = GovernedStateReplayer().rollback(
        store=RuntimeStateStore(world_state=bundle.world),
        committed_event=committed_event,
        verification_result=verification,
    )

    assert store.world_state is bundle.world
    assert not report.rollback_supported
    assert report.errors == ("rollback_unsupported",)
