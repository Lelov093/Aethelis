from __future__ import annotations

from pydantic import Field

from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.events import VerificationDecision
from aethelis.schemas.trace import FormalTraceEnvelope


class EvaluationInput(AethelisModel):
    case_id: Identifier
    trace_id: Identifier
    scenario_id: Identifier
    agent_id: Identifier
    verification_decision: VerificationDecision
    committed_event_present: bool
    state_diff_present: bool
    state_diff_applied: bool
    state_transition_present: bool = False
    causal_projection_present: bool = False
    evolution_update_present: bool = False
    applied_evolution_update_count: int = 0
    applied_patch_count: int = 0
    canon_updated: bool | None = None
    player_input_route: Identifier | None = None
    player_input_canon_updated: bool | None = None
    player_input_world_state_modified: bool | None = None
    activation_required: bool = False
    activation_present: bool = False
    retrieval_present: bool = False
    proposal_summary_present: bool = False
    candidate_summary_present: bool = False
    proposal_source: Identifier | None = None
    verification_check_count: int = 0
    verification_risk_flags: tuple[Identifier, ...] = ()
    safety_flags: tuple[str, ...] = ()
    notes: tuple[str, ...] = Field(default_factory=tuple)


def formal_trace_to_evaluation_inputs(
    trace: FormalTraceEnvelope,
    *,
    case_id_prefix: str = "case",
) -> tuple[EvaluationInput, ...]:
    return tuple(
        EvaluationInput(
            case_id=f"{case_id_prefix}_{index}_{record.scenario_id}",
            trace_id=trace.trace_id,
            scenario_id=record.scenario_id,
            agent_id=record.agent_id,
            verification_decision=record.verification_decision,
            committed_event_present=record.committed_event_id is not None,
            state_diff_present=record.state_diff_id is not None,
            state_diff_applied=record.state_diff_applied,
            state_transition_present=record.state_transition is not None,
            causal_projection_present=record.causal_projection is not None,
            evolution_update_present=record.evolution_update is not None,
            applied_evolution_update_count=_applied_evolution_update_count(record.evolution_update),
            applied_patch_count=_applied_patch_count(record.state_transition),
            canon_updated=_canon_updated(record.player_claim, record.player_input_summary),
            player_input_route=_player_input_route(record.player_input_summary),
            player_input_canon_updated=_player_input_bool(
                record.player_input_summary,
                "canon_updated",
            ),
            player_input_world_state_modified=_player_input_bool(
                record.player_input_summary,
                "world_state_modified",
            ),
            activation_required=bool(trace.metadata.get("activation_trace_included", False)),
            activation_present=record.activation_summary is not None,
            retrieval_present=record.retrieval_summary is not None,
            proposal_summary_present=record.proposal_summary is not None,
            candidate_summary_present=record.candidate_summary is not None,
            proposal_source=record.proposal_source,
            verification_check_count=len(record.verification_checks),
            verification_risk_flags=record.verification_risk_flags,
            safety_flags=record.safety_flags,
            notes=record.verification_reasons or record.notes,
        )
        for index, record in enumerate(trace.records)
    )


def _applied_patch_count(state_transition: dict[str, object] | None) -> int:
    if state_transition is None:
        return 0
    value = state_transition.get("applied_patch_count", 0)
    return value if isinstance(value, int) else 0


def _applied_evolution_update_count(evolution_update: dict[str, object] | None) -> int:
    if evolution_update is None:
        return 0
    value = evolution_update.get("applied_update_count", 0)
    return value if isinstance(value, int) else 0


def _canon_updated(player_claim, player_input_summary: dict[str, object] | None) -> bool | None:
    if player_input_summary is not None:
        value = player_input_summary.get("canon_updated")
        return value if isinstance(value, bool) else None
    if player_claim is not None:
        return player_claim.canon_updated
    return None


def _player_input_route(player_input_summary: dict[str, object] | None) -> Identifier | None:
    if player_input_summary is None:
        return None
    value = player_input_summary.get("route")
    return value if isinstance(value, str) else None


def _player_input_bool(
    player_input_summary: dict[str, object] | None,
    key: str,
) -> bool | None:
    if player_input_summary is None:
        return None
    value = player_input_summary.get(key)
    return value if isinstance(value, bool) else None
