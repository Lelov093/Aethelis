from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import Field

from aethelis.algorithms.runtime_features import verification_behavior_score
from aethelis.agents.context import CognitionContext, ObservationContext
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.events import (
    ActionProposal,
    EventCandidate,
    VerificationCheck,
    VerificationDecision,
    VerificationResult,
)
from aethelis.schemas.seed import SeedBundle


@dataclass(frozen=True)
class VerifierContext:
    bundle: SeedBundle
    scenario_id: str
    proposal: ActionProposal
    candidate: EventCandidate
    observation: ObservationContext | None = None
    cognition: CognitionContext | None = None


class VerificationRuleResult(AethelisModel):
    rule_id: Identifier
    passed: bool
    message: str = Field(min_length=1)
    suggested_decision: VerificationDecision | None = None
    risk_flags: tuple[Identifier, ...] = ()
    reason: str | None = None


VerificationRule = Callable[[VerifierContext], VerificationRuleResult]


class VerifierRegistry:
    """Minimal deterministic rule runner.

    It is intentionally not a plugin framework or policy engine. It only
    composes explicit rule results into the existing VerificationResult schema.
    """

    def __init__(self, *, verifier_name: str) -> None:
        self.verifier_name = verifier_name

    def verify(
        self,
        context: VerifierContext,
        rules: Sequence[VerificationRule],
    ) -> VerificationResult:
        results = tuple(rule(context) for rule in rules)
        decision = _compose_decision(results)
        risk_flags = _unique_flags(results)
        reasons = _reasons(results, decision, risk_flags)
        return VerificationResult(
            id=f"verification_{context.candidate.id}",
            event_candidate_id=context.candidate.id,
            decision=decision,
            verifier=self.verifier_name,
            checks=tuple(
                VerificationCheck(
                    name=result.rule_id,
                    passed=result.passed,
                    message=result.message,
                )
                for result in results
            ),
            reasons=reasons,
            risk_flags=risk_flags,
        )


def rule_result(
    rule_id: str,
    passed: bool,
    message: str,
    *,
    suggested_decision: VerificationDecision | None = None,
    risk_flags: tuple[str, ...] = (),
    reason: str | None = None,
) -> VerificationRuleResult:
    return VerificationRuleResult(
        rule_id=rule_id,
        passed=passed,
        message=message,
        suggested_decision=suggested_decision,
        risk_flags=risk_flags,
        reason=reason,
    )


def _compose_decision(results: tuple[VerificationRuleResult, ...]) -> VerificationDecision:
    suggested = {result.suggested_decision for result in results}
    if VerificationDecision.PENDING_GATE in suggested:
        return VerificationDecision.PENDING_GATE
    if VerificationDecision.REVISE in suggested:
        return VerificationDecision.REVISE
    if VerificationDecision.REJECT in suggested:
        return VerificationDecision.REJECT
    score = _verification_score(results)
    if score < 0.35:
        return VerificationDecision.REJECT
    if score < 0.65:
        return VerificationDecision.REVISE
    if all(result.passed for result in results):
        return VerificationDecision.COMMIT
    return VerificationDecision.REJECT


def _verification_score(results: tuple[VerificationRuleResult, ...]) -> float:
    passed = sum(1 for result in results if result.passed)
    total = max(len(results), 1)
    pass_rate = passed / total
    contradiction_risk = 0.0 if passed == total else (total - passed) / total
    return verification_behavior_score(
        hard_gate=1.0 if passed == total else 0.0,
        check_pass_rate=pass_rate,
        evidence_support=pass_rate,
        causal_coherence=pass_rate,
        state_safety=1.0 if not any(result.risk_flags for result in results) else 0.55,
        contradiction_risk=contradiction_risk,
    )


def _unique_flags(results: tuple[VerificationRuleResult, ...]) -> tuple[Identifier, ...]:
    flags: list[str] = []
    for result in results:
        for flag in result.risk_flags:
            if flag not in flags:
                flags.append(flag)
    return tuple(flags)


def _reasons(
    results: tuple[VerificationRuleResult, ...],
    decision: VerificationDecision,
    risk_flags: tuple[Identifier, ...],
) -> tuple[str, ...]:
    reasons = [
        result.reason or result.message
        for result in results
        if not result.passed or result.suggested_decision is not None
    ]
    if risk_flags:
        reasons.append(f"Risk flags: {', '.join(risk_flags)}")
    reasons.append(f"Behavior verifier score={_verification_score(results):.2f}.")
    if not reasons:
        reasons.append(f"Verification decision={decision.value}: all deterministic checks passed.")
    return tuple(reasons)
