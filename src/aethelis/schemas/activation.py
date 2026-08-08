from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from aethelis.schemas.common import AethelisModel, Identifier

ActorType = Literal["agent", "player"]


class ActivationMode(StrEnum):
    STATIC_TRACE = "static_trace"


class ActivationScoringVersion(StrEnum):
    RULE_BASED_V0 = "rule_based_v0"


class ActivationStatus(StrEnum):
    CANDIDATE = "candidate"
    SELECTED_STATIC_PLAN = "selected_static_plan"
    BACKGROUND = "background"
    NOT_SELECTED = "not_selected"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ActivationReason(AethelisModel):
    reason_type: Identifier
    score: int = Field(ge=0, le=3)
    evidence_ids: tuple[Identifier, ...] = ()
    message: str = Field(min_length=1)
    visibility_scope: Identifier


class AgentActivationConfig(AethelisModel):
    mode: ActivationMode = ActivationMode.STATIC_TRACE
    scoring_version: ActivationScoringVersion = ActivationScoringVersion.RULE_BASED_V0
    include_non_selected_candidates: bool = False
    max_candidates: int = Field(default=1, ge=1)
    top_k: int = Field(default=1, ge=1)
    selection_threshold: int = Field(default=0, ge=0)
    use_pressure_seeds: bool = True
    use_action_metadata: bool = True
    use_relationship_placeholder: bool = True
    allow_private_belief_scoring: Literal[False] = False
    allow_real_llm: Literal[False] = False


class ActivationCandidate(AethelisModel):
    candidate_id: Identifier
    run_id: Identifier
    step_id: Identifier
    agent_id: Identifier
    actor_type: ActorType
    scenario_id: Identifier
    status: ActivationStatus
    score_total: int = Field(ge=0)
    reasons: tuple[ActivationReason, ...]
    selected_by: Identifier = "static_step_plan"
    threshold_passed: bool = True
    top_k_rank: int = Field(default=1, ge=1)
    tie_break_key: tuple[Identifier, ...] = ()

    def safe_summary(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "agent_id": self.agent_id,
            "actor_type": self.actor_type,
            "scenario_id": self.scenario_id,
            "activation_score": self.score_total,
            "status": self.status.value,
            "score_total": self.score_total,
            "selected_by": self.selected_by,
            "threshold_passed": self.threshold_passed,
            "top_k_rank": self.top_k_rank,
            "tie_break_key": list(self.tie_break_key),
            "reasons": [
                {
                    "reason_type": reason.reason_type,
                    "score": reason.score,
                    "evidence_ids": list(reason.evidence_ids),
                    "visibility_scope": reason.visibility_scope,
                }
                for reason in self.reasons
            ],
        }


class ActivationResult(AethelisModel):
    activation_result_id: Identifier
    run_id: Identifier
    step_id: Identifier
    scenario_id: Identifier
    mode: ActivationMode = ActivationMode.STATIC_TRACE
    scoring_version: ActivationScoringVersion = ActivationScoringVersion.RULE_BASED_V0
    scheduler_version: Identifier = "deterministic_scheduler_v0"
    selected_candidate: ActivationCandidate
    candidate_count: int = Field(ge=1)
    candidates: tuple[ActivationCandidate, ...]
    provider_called: Literal[False] = False
    world_state_modified: Literal[False] = False
    action_proposal_generated: Literal[False] = False
    hidden_context_used: Literal[False] = False

    def safe_summary(self) -> dict[str, object]:
        return {
            "activation_result_id": self.activation_result_id,
            "mode": self.mode.value,
            "scoring_version": self.scoring_version.value,
            "scheduler_version": self.scheduler_version,
            "selected_candidate": self.selected_candidate.safe_summary(),
            "candidate_count": self.candidate_count,
            "top_k": max(candidate.top_k_rank for candidate in self.candidates),
            "provider_called": self.provider_called,
            "world_state_modified": self.world_state_modified,
            "action_proposal_generated": self.action_proposal_generated,
            "hidden_context_used": self.hidden_context_used,
        }
