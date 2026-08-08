from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from aethelis.algorithms.runtime_features import action_proposal_behavior_score
from aethelis.events.fixtures import DeterministicActionProposalFactory
from aethelis.llm.base import LLMProvider, StructuredLLMResult
from aethelis.schemas.events import ActionProposal


class ActionProposalSource(StrEnum):
    DETERMINISTIC_FIXTURE = "deterministic_fixture"
    REAL_LLM_STRUCTURED_OUTPUT = "real_llm_structured_output"
    TEST_STUB = "test_stub"


class ProposalSourceMode(StrEnum):
    DETERMINISTIC = "deterministic"
    PROVIDER_STRUCTURED = "provider_structured"


class ProviderProposalFailureCode(StrEnum):
    DISABLED_BY_CONFIG = "disabled_by_config"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MALFORMED_OUTPUT = "malformed_output"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    UNSAFE_CONTENT_OR_RAW_TEXT_BLOCKED = "unsafe_content_or_raw_text_blocked"


class ProposalBehaviorDecision(StrEnum):
    ACCEPT = "accept"
    REPAIR = "repair"
    HOLD = "hold"
    REJECT = "reject"


@dataclass(frozen=True)
class ActionProposalGenerationResult:
    proposal: ActionProposal | None
    source: ActionProposalSource
    structured_output: StructuredLLMResult[ActionProposal] | None = None
    provider_called: bool = False
    error: str | None = None
    failure_code: ProviderProposalFailureCode | None = None
    behavior_score: float | None = None
    behavior_decision: ProposalBehaviorDecision | None = None
    behavior_risk_flags: tuple[str, ...] = ()


class ActionProposalEngine:
    """Generate ActionProposal through explicit deterministic or real-provider paths."""

    def generate_deterministic(
        self,
        *,
        agent_id: str,
        scenario_id: str,
    ) -> ActionProposalGenerationResult:
        proposal = DeterministicActionProposalFactory().build(
            agent_id=agent_id,
            scenario_id=scenario_id,
        )
        return ActionProposalGenerationResult(
            proposal=proposal,
            source=ActionProposalSource.DETERMINISTIC_FIXTURE,
            **_proposal_behavior_payload(proposal),
        )

    def generate_structured(
        self,
        *,
        provider: LLMProvider,
        prompt: str,
        max_tokens: int = 260,
        temperature: float = 0.0,
    ) -> ActionProposalGenerationResult:
        structured = None
        for _ in range(2):
            structured = provider.generate_structured(
                prompt,
                ActionProposal,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if structured.success or _failure_code(structured) not in {
                ProviderProposalFailureCode.MALFORMED_OUTPUT,
                ProviderProposalFailureCode.SCHEMA_VALIDATION_FAILED,
            }:
                break
        assert structured is not None
        return ActionProposalGenerationResult(
            proposal=structured.data,
            source=ActionProposalSource.REAL_LLM_STRUCTURED_OUTPUT,
            structured_output=structured,
            provider_called=True,
            error=None if structured.success else "Structured ActionProposal generation failed.",
            failure_code=_failure_code(structured),
            **_proposal_behavior_payload(structured.data),
        )


def repair_action_proposal(proposal: ActionProposal) -> ActionProposal:
    """Apply a bounded behavior-level repair without inventing new facts."""

    text = f"{proposal.rationale} {proposal.expected_outcome}".lower()
    if not any(marker in text for marker in ("force", "bypass")):
        return proposal
    return proposal.model_copy(
        update={
            "rationale": _safe_repair_text(proposal.rationale),
            "expected_outcome": _safe_repair_text(proposal.expected_outcome),
        }
    )


def _failure_code(
    structured: StructuredLLMResult[ActionProposal],
) -> ProviderProposalFailureCode | None:
    if structured.success:
        return None
    if structured.json_parse_error is not None:
        return ProviderProposalFailureCode.MALFORMED_OUTPUT
    if structured.validation_error is not None:
        return ProviderProposalFailureCode.SCHEMA_VALIDATION_FAILED
    return ProviderProposalFailureCode.UNSAFE_CONTENT_OR_RAW_TEXT_BLOCKED


def _proposal_behavior_payload(proposal: ActionProposal | None) -> dict[str, object]:
    if proposal is None:
        return {}
    target_present = bool(proposal.target_location_id or proposal.target_entity_ids)
    hard_risk = any(
        word in f"{proposal.rationale} {proposal.expected_outcome}".lower()
        for word in ("rewrite canon", "state diff")
    )
    soft_risk = any(
        word in f"{proposal.rationale} {proposal.expected_outcome}".lower()
        for word in ("force", "bypass")
    )
    risk = 1.0 if hard_risk else 0.70 if soft_risk else 0.0
    feasibility = 1.0 if target_present else 0.15
    governance = 0.0 if risk >= 1.0 else 1.0
    score = action_proposal_behavior_score(
        utility=0.8,
        feasibility=feasibility,
        governance=governance,
        risk=risk,
    )
    decision = (
        ProposalBehaviorDecision.REJECT
        if hard_risk
        else ProposalBehaviorDecision.HOLD
        if score < 0.40 or not target_present
        else ProposalBehaviorDecision.REPAIR
        if soft_risk or score < 0.65
        else ProposalBehaviorDecision.ACCEPT
    )
    return {
        "behavior_score": score,
        "behavior_decision": decision,
        "behavior_risk_flags": ("proposal_governance_risk",) if hard_risk else ("proposal_soft_risk",) if soft_risk else (),
    }


def _safe_repair_text(value: str) -> str:
    repaired = value
    for marker in ("force", "Force", "bypass", "Bypass"):
        repaired = repaired.replace(marker, "request verified")
    repaired = repaired.replace("StateDiff", "verified event")
    repaired = repaired.replace("state diff", "verified event")
    repaired = " ".join(repaired.split())
    if len(repaired) > 220:
        return repaired[:217] + "..."
    return repaired or "Request verified governance handling."
