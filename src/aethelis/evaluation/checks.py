from __future__ import annotations

from collections import Counter

from aethelis.evaluation.inputs import EvaluationInput, formal_trace_to_evaluation_inputs
from aethelis.evaluation.regression_cases import RegressionCase, default_regression_cases
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.events import VerificationDecision
from aethelis.schemas.trace import FormalTraceEnvelope


class CaseEvaluationResult(AethelisModel):
    case_id: Identifier
    scenario_id: Identifier
    expected_decision: VerificationDecision | None
    actual_decision: VerificationDecision
    passed: bool
    failure_reason: str | None = None
    safety_flags: tuple[str, ...] = ()
    trace_id: Identifier


class EvaluationCheckSummary(AethelisModel):
    trace_id: Identifier
    formal_experiment_result: bool
    case_count: int
    passed_count: int
    failed_count: int
    decisions: dict[str, int]
    results: tuple[CaseEvaluationResult, ...]

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def evaluate_formal_trace_preview(
    trace: FormalTraceEnvelope,
    *,
    regression_cases: tuple[RegressionCase, ...] | None = None,
) -> EvaluationCheckSummary:
    cases = regression_cases if regression_cases is not None else default_regression_cases()
    lookup = _case_lookup(cases)
    inputs = formal_trace_to_evaluation_inputs(trace, case_id_prefix="eval")
    results = tuple(_evaluate_input(input_, lookup.get(_case_key(input_))) for input_ in inputs)
    decisions = Counter(result.actual_decision.value for result in results)
    passed_count = sum(1 for result in results if result.passed)
    return EvaluationCheckSummary(
        trace_id=trace.trace_id,
        formal_experiment_result=trace.formal_experiment_result,
        case_count=len(results),
        passed_count=passed_count,
        failed_count=len(results) - passed_count,
        decisions=dict(sorted(decisions.items())),
        results=results,
    )


def _evaluate_input(
    input_: EvaluationInput,
    regression_case: RegressionCase | None,
) -> CaseEvaluationResult:
    if regression_case is None:
        return CaseEvaluationResult(
            case_id=input_.case_id,
            scenario_id=input_.scenario_id,
            expected_decision=None,
            actual_decision=input_.verification_decision,
            passed=False,
            failure_reason="unknown_regression_case",
            safety_flags=input_.safety_flags,
            trace_id=input_.trace_id,
        )

    failures: list[str] = []
    if input_.verification_decision != regression_case.expected_decision:
        failures.append(
            "decision mismatch: "
            f"expected {regression_case.expected_decision.value}, "
            f"got {input_.verification_decision.value}"
        )
    if input_.committed_event_present != regression_case.expects_committed_event:
        failures.append("committed_event presence mismatch")
    if input_.state_diff_present != regression_case.expects_state_diff:
        failures.append("state_diff presence mismatch")
    if input_.state_diff_applied != regression_case.expects_state_diff_applied:
        failures.append("state_diff_applied mismatch")
    if input_.verification_decision == VerificationDecision.COMMIT:
        if not input_.state_transition_present:
            failures.append("commit trace missing state_transition")
        if not input_.causal_projection_present:
            failures.append("commit trace missing causal_projection")
        if not input_.evolution_update_present:
            failures.append("commit trace missing evolution_update")
        if input_.applied_evolution_update_count <= 0:
            failures.append("commit trace missing applied evolution updates")
    else:
        if input_.state_transition_present:
            failures.append("non-commit trace has state_transition")
        if input_.causal_projection_present:
            failures.append("non-commit trace has causal_projection")
        if input_.applied_evolution_update_count != 0:
            failures.append("non-commit trace has applied evolution updates")
    if input_.state_diff_applied and input_.applied_patch_count <= 0:
        failures.append("state_diff_applied without applied patches")
    if not input_.state_diff_applied and input_.applied_patch_count != 0:
        failures.append("applied_patch_count mismatch")
    if (
        regression_case.expects_canon_updated is not None
        and input_.canon_updated != regression_case.expects_canon_updated
    ):
        failures.append("canon_updated mismatch")
    if input_.player_input_canon_updated is True:
        failures.append("player input updated canon directly")
    if input_.player_input_world_state_modified is True:
        failures.append("player input modified world state directly")
    if input_.agent_id != "player":
        if input_.activation_required and not input_.activation_present:
            failures.append("agent workflow missing activation summary")
        if not input_.retrieval_present:
            failures.append("agent workflow missing retrieval summary")
        if not input_.proposal_summary_present:
            failures.append("agent workflow missing proposal summary")
        if input_.proposal_source is None:
            failures.append("agent workflow missing proposal source")
        if not input_.candidate_summary_present:
            failures.append("agent workflow missing event candidate summary")

    return CaseEvaluationResult(
        case_id=regression_case.id,
        scenario_id=input_.scenario_id,
        expected_decision=regression_case.expected_decision,
        actual_decision=input_.verification_decision,
        passed=not failures,
        failure_reason="; ".join(failures) if failures else None,
        safety_flags=input_.safety_flags,
        trace_id=input_.trace_id,
    )


def _case_lookup(cases: tuple[RegressionCase, ...]) -> dict[tuple[str, str], RegressionCase]:
    return {(case.agent_id, case.scenario_id): case for case in cases}


def _case_key(input_: EvaluationInput) -> tuple[str, str]:
    return (input_.agent_id, input_.scenario_id)
