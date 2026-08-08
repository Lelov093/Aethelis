from __future__ import annotations

from typing import Literal

from pydantic import Field

from aethelis.runtime.state_apply import StateApplyReport
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.evolution import EvolutionRuntimeState
from aethelis.schemas.world import WorldState

JournalEntryType = Literal["apply", "replay"]


class StateJournalEntry(AethelisModel):
    entry_id: Identifier
    entry_type: JournalEntryType
    sequence_index: int = Field(ge=0)
    committed_event_id: Identifier | None = None
    state_diff_id: Identifier | None = None
    verification_result_id: Identifier | None = None
    applied: bool = False
    applied_patch_count: int = Field(default=0, ge=0)
    skipped_patch_count: int = Field(default=0, ge=0)
    errors: tuple[str, ...] = ()

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class RuntimeStateStore(AethelisModel):
    """In-memory runtime boundary for governed state evolution."""

    world_state: WorldState
    evolution_state: EvolutionRuntimeState = Field(default_factory=EvolutionRuntimeState)
    apply_journal: tuple[StateJournalEntry, ...] = ()
    replay_journal: tuple[StateJournalEntry, ...] = ()

    def record_apply_result(
        self,
        *,
        world_state: WorldState,
        report: StateApplyReport,
        verification_result_id: str | None,
        entry_type: JournalEntryType = "apply",
    ) -> tuple[RuntimeStateStore, StateApplyReport]:
        entry = _journal_entry(
            report=report,
            verification_result_id=verification_result_id,
            entry_type=entry_type,
            sequence_index=len(self.apply_journal) + len(self.replay_journal),
        )
        journaled_report = report.model_copy(update={"journal_entry_id": entry.entry_id})
        next_world = world_state if report.applied else self.world_state
        if entry_type == "replay":
            return (
                self.model_copy(
                    update={
                        "world_state": next_world,
                        "replay_journal": (*self.replay_journal, entry),
                    }
                ),
                journaled_report,
            )
        return (
            self.model_copy(
                update={
                    "world_state": next_world,
                    "apply_journal": (*self.apply_journal, entry),
                }
            ),
            journaled_report,
        )

    def with_evolution_state(self, evolution_state: EvolutionRuntimeState) -> RuntimeStateStore:
        return self.model_copy(update={"evolution_state": evolution_state})

    def safe_summary(self) -> dict[str, object]:
        return {
            "apply_journal_count": len(self.apply_journal),
            "replay_journal_count": len(self.replay_journal),
            "last_apply": self.apply_journal[-1].safe_dict() if self.apply_journal else None,
            "last_replay": (self.replay_journal[-1].safe_dict() if self.replay_journal else None),
        }


def _journal_entry(
    *,
    report: StateApplyReport,
    verification_result_id: str | None,
    entry_type: JournalEntryType,
    sequence_index: int,
) -> StateJournalEntry:
    state_diff_id = report.state_diff_id or "no_state_diff"
    return StateJournalEntry(
        entry_id=f"{entry_type}_journal:{sequence_index}:{state_diff_id}",
        entry_type=entry_type,
        sequence_index=sequence_index,
        committed_event_id=report.committed_event_id,
        state_diff_id=report.state_diff_id,
        verification_result_id=verification_result_id,
        applied=report.applied,
        applied_patch_count=report.applied_patch_count,
        skipped_patch_count=report.skipped_patch_count,
        errors=report.errors,
    )
