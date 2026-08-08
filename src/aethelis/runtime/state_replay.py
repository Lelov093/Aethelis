from __future__ import annotations

from aethelis.runtime.state_apply import ControlledStateDiffApplier, StateApplyReport
from aethelis.runtime.state_store import RuntimeStateStore
from aethelis.schemas.common import AethelisModel
from aethelis.schemas.events import (
    ActionProposal,
    CommittedEvent,
    EventCandidate,
    StateDiff,
    VerificationResult,
)


class StateReplayReport(AethelisModel):
    replayed: bool = False
    replay_journal_entry_id: str | None = None
    apply_report: StateApplyReport | None = None
    rollback_supported: bool = False
    errors: tuple[str, ...] = ()

    def safe_dict(self) -> dict[str, object]:
        return {
            "replayed": self.replayed,
            "replay_journal_entry_id": self.replay_journal_entry_id,
            "apply_report": (
                self.apply_report.safe_dict() if self.apply_report is not None else None
            ),
            "rollback_supported": self.rollback_supported,
            "errors": list(self.errors),
        }


class GovernedStateReplayer:
    """Replay committed StateDiffs through the same commit-only applier."""

    def replay(
        self,
        *,
        store: RuntimeStateStore,
        committed_event: CommittedEvent,
        verification_result: VerificationResult,
    ) -> tuple[RuntimeStateStore, StateReplayReport]:
        updated_world, apply_report = ControlledStateDiffApplier().apply(
            world_state=store.world_state,
            committed_event=committed_event,
            verification_result=verification_result,
        )
        next_store, journaled_report = store.record_apply_result(
            world_state=updated_world,
            report=apply_report,
            verification_result_id=verification_result.id,
            entry_type="replay",
        )
        return (
            next_store,
            StateReplayReport(
                replayed=journaled_report.applied,
                replay_journal_entry_id=journaled_report.journal_entry_id,
                apply_report=journaled_report,
                errors=journaled_report.errors,
            ),
        )

    def replay_state_diff(self, state_diff: StateDiff) -> None:
        raise TypeError("StateDiff cannot be replayed without a committed event")

    def replay_action_proposal(self, action_proposal: ActionProposal) -> None:
        raise TypeError("ActionProposal cannot trigger StateDiff replay")

    def replay_event_candidate(self, event_candidate: EventCandidate) -> None:
        raise TypeError("EventCandidate cannot trigger StateDiff replay")

    def rollback(
        self,
        *,
        store: RuntimeStateStore,
        committed_event: CommittedEvent,
        verification_result: VerificationResult,
    ) -> tuple[RuntimeStateStore, StateReplayReport]:
        return (
            store,
            StateReplayReport(
                replayed=False,
                rollback_supported=False,
                errors=("rollback_unsupported",),
            ),
        )
