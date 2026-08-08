from __future__ import annotations

from pathlib import Path

from aethelis.agents.action_proposal import ProposalSourceMode
from aethelis.llm.base import LLMProvider, LLMResult
from aethelis.providers import ProviderAttempt
from aethelis.runtime.scenario_matrix import RUNTIME_SCENARIO_MATRIX
from aethelis.runtime.single_step import SingleStepResult, run_single_step
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.events import VerificationDecision

DEFAULT_REGRESSION_SCENARIO_IDS = frozenset(
    {
        "inspect_workshop_safe",
        "ivo_inspect_workshop_safe_fixture",
        "mira_search_archive_wrong_key",
        "selka_consume_stabilizer_part_fixture",
        "selka_restock_market_credit_fixture",
        "malformed_or_incomplete_action",
        "unsafe_force_open_safe",
        "player_claim_key_in_hand",
        "player_request_open_workshop_safe",
    }
)


class RegressionCase(AethelisModel):
    id: Identifier
    agent_id: Identifier
    scenario_id: Identifier
    expected_decision: VerificationDecision
    expects_committed_event: bool
    expects_state_diff: bool
    expects_state_diff_applied: bool = False
    expects_canon_updated: bool | None = None


class RegressionCaseResult(AethelisModel):
    case: RegressionCase
    passed: bool
    result: SingleStepResult
    failures: tuple[str, ...] = ()


def default_regression_cases() -> tuple[RegressionCase, ...]:
    return tuple(
        case
        for case in scenario_matrix_regression_cases()
        if case.scenario_id in DEFAULT_REGRESSION_SCENARIO_IDS
    )


def scenario_matrix_regression_cases() -> tuple[RegressionCase, ...]:
    return tuple(
        RegressionCase(
            id=scenario.regression_case_id,
            agent_id=scenario.actor_id,
            scenario_id=scenario.scenario_id,
            expected_decision=scenario.expected_decision,
            expects_committed_event=scenario.expects_committed_event,
            expects_state_diff=scenario.expects_state_diff,
            expects_state_diff_applied=False,
            expects_canon_updated=False if scenario.is_player_input else None,
        )
        for scenario in RUNTIME_SCENARIO_MATRIX
    )


def run_regression_case(seed_path: Path, case: RegressionCase) -> RegressionCaseResult:
    provider = (
        _FixtureActionProposalProvider() if case.scenario_id == "inspect_workshop_safe" else None
    )
    result = run_single_step(
        seed_path=seed_path,
        agent_id=case.agent_id,
        scenario_id=case.scenario_id,
        provider=provider,
        proposal_source=(
            ProposalSourceMode.PROVIDER_STRUCTURED
            if provider is not None
            else ProposalSourceMode.DETERMINISTIC
        ),
        provider_proposals_enabled=provider is not None,
        allow_real_provider=provider is not None,
        apply=False,
    )
    failures = _evaluate_result(case, result)
    return RegressionCaseResult(
        case=case,
        passed=not failures,
        result=result,
        failures=tuple(failures),
    )


def _evaluate_result(case: RegressionCase, result: SingleStepResult) -> list[str]:
    failures: list[str] = []
    decision = result.verification_result.decision if result.verification_result else None
    if decision != case.expected_decision:
        failures.append(f"expected decision {case.expected_decision.value}, got {decision}")
    if (result.committed_event is not None) != case.expects_committed_event:
        failures.append("committed_event presence mismatch")
    state_diff_present = (
        result.committed_event is not None and result.committed_event.state_diff is not None
    )
    if state_diff_present != case.expects_state_diff:
        failures.append("state_diff presence mismatch")
    if result.state_diff_applied != case.expects_state_diff_applied:
        failures.append("state_diff_applied mismatch")
    if case.expects_canon_updated is not None:
        canon_updated = _canon_updated(result)
        if canon_updated != case.expects_canon_updated:
            failures.append("canon_updated mismatch")
    return failures


def _canon_updated(result: SingleStepResult) -> bool | None:
    if result.player_input_summary is not None:
        value = result.player_input_summary.get("canon_updated")
        return value if isinstance(value, bool) else None
    if result.player_claim is not None:
        return result.player_claim.canon_updated
    return None


class _FixtureActionProposalProvider:
    provider_name = "fixture_test_provider"

    def generate(self, prompt: str, *, max_tokens: int = 32, temperature: float = 0.0) -> LLMResult:
        return LLMResult(
            content=(
                '{"id":"proposal_inspect_workshop_safe_ivo",'
                '"proposer_agent_id":"ivo",'
                '"intent":"investigate",'
                '"rationale":"Inspect the workshop safe using Ivo own lawful access.",'
                '"target_location_id":"workshop_lane",'
                '"target_entity_ids":["workshop_safe"],'
                '"expected_outcome":"Inspect the workshop safe for the calibration key."}'
            ),
            model="fixture-test-model",
            latency_ms=1,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            attempts=(ProviderAttempt(model="fixture-test-model", success=True, latency_ms=1),),
        )

    def generate_structured(self, prompt, schema_type, *, max_tokens=512, temperature=0.0):
        from aethelis.llm.structured import generate_structured

        return generate_structured(
            self,
            prompt,
            schema_type,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def _assert_protocol(_: LLMProvider) -> None:
    return None


_assert_protocol(_FixtureActionProposalProvider())
