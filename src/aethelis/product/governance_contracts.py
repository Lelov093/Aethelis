from __future__ import annotations

from dataclasses import dataclass

from aethelis.runtime.state_apply import StateApplyReport
from aethelis.schemas.events import (
    ActionProposal,
    CommittedEvent,
    EventCandidate,
    VerificationResult,
)
from aethelis.schemas.world import WorldState


@dataclass(frozen=True)
class GovernedWorldOutcome:
    proposal: ActionProposal
    candidate: EventCandidate
    verification: VerificationResult
    committed_event: CommittedEvent | None
    resulting_world_state: WorldState | None
    apply_report: StateApplyReport | None
    player_message: str
    consequences: tuple[str, ...]

    @property
    def committed(self) -> bool:
        return self.committed_event is not None and self.resulting_world_state is not None
