from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.events import EventCandidate, VerificationResult
from aethelis.schemas.ledger import BeliefCandidate


class PlayerInputKind(StrEnum):
    CLAIM = "claim"
    REQUEST = "request"
    ACTION = "action"
    DECEPTION = "deception"
    META = "meta"


class PlayerInputRoute(StrEnum):
    BELIEF_CANDIDATE = "belief_candidate"
    EVENT_CANDIDATE = "event_candidate"
    REJECTED_CLAIM = "rejected_claim"
    PENDING_GATE = "pending_gate"
    META_NOOP = "meta_noop"


class PlayerInputRecord(AethelisModel):
    id: Identifier
    player_id: Identifier = "player"
    kind: PlayerInputKind
    text: str = Field(min_length=1, max_length=500)
    target_location_id: Identifier | None = None
    target_entity_ids: tuple[Identifier, ...] = ()
    source_scenario_id: Identifier | None = None


class RoutedPlayerInput(AethelisModel):
    input_id: Identifier
    player_id: Identifier
    input_kind: PlayerInputKind
    route: PlayerInputRoute
    belief_candidate: BeliefCandidate | None = None
    event_candidate: EventCandidate | None = None
    verification_result: VerificationResult
    canon_updated: bool = False
    world_state_modified: bool = False
    state_diff_id: Identifier | None = None
    safety_flags: tuple[Identifier, ...] = ()

    def safe_summary(self) -> dict[str, object]:
        return {
            "input_id": self.input_id,
            "player_id": self.player_id,
            "input_kind": self.input_kind.value,
            "route": self.route.value,
            "belief_candidate_id": (
                self.belief_candidate.id if self.belief_candidate is not None else None
            ),
            "belief_candidate_status": (
                self.belief_candidate.status.value if self.belief_candidate is not None else None
            ),
            "belief_candidate_confidence": (
                self.belief_candidate.confidence.value
                if self.belief_candidate is not None
                else None
            ),
            "belief_candidate_trace_reference_id": (
                self.belief_candidate.trace_reference_id
                if self.belief_candidate is not None
                else None
            ),
            "event_candidate_id": (
                self.event_candidate.id if self.event_candidate is not None else None
            ),
            "verification_decision": self.verification_result.decision.value,
            "canon_updated": self.canon_updated,
            "world_state_modified": self.world_state_modified,
            "state_diff_id": self.state_diff_id,
            "safety_flags": list(self.safety_flags),
        }
