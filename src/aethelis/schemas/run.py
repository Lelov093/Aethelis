from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from aethelis.schemas.activation import ActivationResult, AgentActivationConfig
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.events import (
    ActionProposalSummary,
    EventCandidateSummary,
    VerificationDecision,
)
from aethelis.schemas.evolution import EvolutionRuntimeState, EvolutionUpdateSummary
from aethelis.schemas.world import WorldState

ActorType = Literal["agent", "player"]


class RunMode(StrEnum):
    DETERMINISTIC_PREVIEW = "deterministic_preview"
    REAL_PROVIDER_PREVIEW = "real_provider_preview"


class RunStepPlanItem(AethelisModel):
    step_id: Identifier
    agent_id: Identifier
    actor_type: ActorType = "agent"
    scenario_id: Identifier
    allow_real_llm: bool = False
    apply: bool = False


class RunConfig(AethelisModel):
    run_id: Identifier
    mode: RunMode = RunMode.REAL_PROVIDER_PREVIEW
    formal_experiment_result: Literal[False] = False
    allow_real_llm: bool = True
    dry_run: Literal[True] = True
    apply: Literal[False] = False
    activation: AgentActivationConfig = Field(default_factory=AgentActivationConfig)
    step_plan: tuple[RunStepPlanItem, ...] = Field(min_length=1)


class WorldRunState(AethelisModel):
    run_id: Identifier
    seed_id: Identifier
    dry_run: bool = True
    current_step_index: int = Field(default=0, ge=0)
    world_state: WorldState | None = None
    state_diff_applied_count: int = Field(default=0, ge=0)
    committed_event_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    revise_count: int = Field(default=0, ge=0)
    pending_gate_count: int = Field(default=0, ge=0)
    provider_called: bool = False
    evolution_state: EvolutionRuntimeState = Field(default_factory=EvolutionRuntimeState)
    apply_journal_count: int = Field(default=0, ge=0)
    replay_journal_count: int = Field(default=0, ge=0)
    safety_flags: tuple[str, ...] = ()


class StateTransitionPatchSummary(AethelisModel):
    patch_index: int = Field(ge=0)
    applied: bool = False
    target_type: str
    target_id: Identifier
    path: str = Field(min_length=1)
    before_summary: object = None
    after_summary: object = None
    error: str | None = None


class StateTransitionSummary(AethelisModel):
    step_id: Identifier
    committed_event_id: Identifier
    state_diff_id: Identifier
    applied: bool = False
    applied_patch_count: int = Field(default=0, ge=0)
    skipped_patch_count: int = Field(default=0, ge=0)
    patches: tuple[StateTransitionPatchSummary, ...] = ()


class CausalTraceProjection(AethelisModel):
    committed_event_node_id: Identifier
    state_diff_node_id: Identifier
    verification_result_id: Identifier
    affected_target_node_ids: tuple[Identifier, ...] = ()
    caused_state_diff_edge_id: Identifier


class WorldStepResult(AethelisModel):
    step_id: Identifier
    step_index: int = Field(ge=0)
    agent_id: Identifier
    actor_type: ActorType
    scenario_id: Identifier
    decision: VerificationDecision
    action_proposal_id: Identifier | None = None
    proposal_summary: ActionProposalSummary | None = None
    event_candidate_id: Identifier | None = None
    candidate_summary: EventCandidateSummary | None = None
    verification_result_id: Identifier | None = None
    verification_checks: tuple[dict[str, object], ...] = ()
    verification_reasons: tuple[str, ...] = ()
    verification_risk_flags: tuple[Identifier, ...] = ()
    committed_event_id: Identifier | None = None
    state_diff_id: Identifier | None = None
    state_diff_applied: bool = False
    apply_report: dict[str, object] | None = None
    state_transition: StateTransitionSummary | None = None
    causal_projection: CausalTraceProjection | None = None
    evolution_update: EvolutionUpdateSummary | None = None
    player_input_summary: dict[str, object] | None = None
    retrieval_summary: dict[str, object] | None = None
    proposal_source: Identifier | None = None
    provider_mode: Identifier | None = None
    fallback_used: bool = False
    fallback_reason: Identifier | None = None
    evidence_class: Identifier | None = None
    provider_called: bool = False
    player_claim_id: Identifier | None = None
    player_claim_summary: str | None = None
    player_claim_canon_updated: bool = False
    player_claim_state_diff_id: Identifier | None = None
    player_claim_rejected_claim_ids: tuple[Identifier, ...] = ()
    activation_result: ActivationResult | None = None
    safety_flags: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def proposal_summary_dict(self) -> dict[str, object] | None:
        if self.proposal_summary is None:
            return None
        return self.proposal_summary.model_dump(mode="json")

    def candidate_summary_dict(self) -> dict[str, object] | None:
        if self.candidate_summary is None:
            return None
        return self.candidate_summary.model_dump(mode="json")

    def activation_summary_dict(self) -> dict[str, object] | None:
        if self.activation_result is None:
            return None
        return self.activation_result.safe_summary()

    def safe_summary(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "step_index": self.step_index,
            "agent_id": self.agent_id,
            "actor_type": self.actor_type,
            "scenario_id": self.scenario_id,
            "decision": self.decision.value,
            "action_proposal_id": self.action_proposal_id,
            "event_candidate_id": self.event_candidate_id,
            "verification_result_id": self.verification_result_id,
            "committed_event_id": self.committed_event_id,
            "state_diff_id": self.state_diff_id,
            "state_diff_applied": self.state_diff_applied,
            "state_transition": (
                self.state_transition.model_dump(mode="json")
                if self.state_transition is not None
                else None
            ),
            "causal_projection": (
                self.causal_projection.model_dump(mode="json")
                if self.causal_projection is not None
                else None
            ),
            "evolution_update": (
                self.evolution_update.safe_dict() if self.evolution_update is not None else None
            ),
            "player_input_summary": self.player_input_summary,
            "retrieval_summary": self.retrieval_summary,
            "proposal_source": self.proposal_source,
            "provider_mode": self.provider_mode,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "evidence_class": self.evidence_class,
            "provider_called": self.provider_called,
            "proposal_summary": self.proposal_summary_dict(),
            "candidate_summary": self.candidate_summary_dict(),
            "verification_check_count": len(self.verification_checks),
            "verification_risk_flags": list(self.verification_risk_flags),
            "activation_summary": self.activation_summary_dict(),
            "safety_flags": list(self.safety_flags),
        }


class WorldRunResult(AethelisModel):
    run_id: Identifier
    seed_id: Identifier
    mode: RunMode
    dry_run: bool = True
    apply_requested: bool = False
    formal_experiment_result: Literal[False] = False
    wrote_runs: Literal[False] = False
    wrote_reports: Literal[False] = False
    raw_text_saved: Literal[False] = False
    provider_called: bool = False
    step_count: int = Field(ge=0)
    decisions: tuple[VerificationDecision, ...] = ()
    steps: tuple[WorldStepResult, ...] = ()
    state_diff_applied: bool = False
    state_diff_applied_count: int = Field(default=0, ge=0)
    committed_event_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    revise_count: int = Field(default=0, ge=0)
    pending_gate_count: int = Field(default=0, ge=0)
    final_state_summary: dict[str, object] | None = None
    final_evolution_state_summary: dict[str, object] | None = None
    causal_runtime_summary: dict[str, object] | None = None
    pressure_runtime_summary: dict[str, object] | None = None
    cognitive_runtime_summary: dict[str, object] | None = None
    apply_journal_count: int = Field(default=0, ge=0)
    replay_journal_count: int = Field(default=0, ge=0)
    errors: tuple[str, ...] = ()

    def safe_summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "seed_id": self.seed_id,
            "mode": self.mode.value,
            "dry_run": self.dry_run,
            "apply_requested": self.apply_requested,
            "formal_experiment_result": self.formal_experiment_result,
            "wrote_runs": self.wrote_runs,
            "wrote_reports": self.wrote_reports,
            "raw_text_saved": self.raw_text_saved,
            "provider_called": self.provider_called,
            "step_count": self.step_count,
            "activation_trace_included": any(
                step.activation_result is not None for step in self.steps
            ),
            "activation_mode": (
                self.steps[0].activation_result.mode.value
                if self.steps and self.steps[0].activation_result is not None
                else None
            ),
            "activation_provider_called": any(
                step.activation_result.provider_called
                for step in self.steps
                if step.activation_result is not None
            ),
            "decisions": [decision.value for decision in self.decisions],
            "state_diff_applied": self.state_diff_applied,
            "state_diff_applied_count": self.state_diff_applied_count,
            "committed_event_count": self.committed_event_count,
            "rejected_count": self.rejected_count,
            "revise_count": self.revise_count,
            "pending_gate_count": self.pending_gate_count,
            "final_state_summary": self.final_state_summary,
            "final_evolution_state_summary": self.final_evolution_state_summary,
            "causal_runtime_summary": self.causal_runtime_summary,
            "pressure_runtime_summary": self.pressure_runtime_summary,
            "cognitive_runtime_summary": self.cognitive_runtime_summary,
            "apply_journal_count": self.apply_journal_count,
            "replay_journal_count": self.replay_journal_count,
            "steps": [step.safe_summary() for step in self.steps],
            "errors": list(self.errors),
        }
