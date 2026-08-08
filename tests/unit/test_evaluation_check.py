from __future__ import annotations

from pathlib import Path

from aethelis.evaluation import (
    CaseEvaluationResult,
    EvaluationCheckSummary,
    evaluate_formal_trace_preview,
)
from aethelis.evaluation.regression_cases import run_regression_case
from aethelis.schemas.events import VerificationDecision
from aethelis.schemas.trace import FormalTraceEnvelope, TraceStepTransaction
from aethelis.trace.formal import build_formal_trace_preview

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"


def test_case_evaluation_result_schema() -> None:
    result = CaseEvaluationResult(
        case_id="reg_reject_mira_wrong_key",
        scenario_id="mira_search_archive_wrong_key",
        expected_decision=VerificationDecision.REJECT,
        actual_decision=VerificationDecision.REJECT,
        passed=True,
        failure_reason=None,
        safety_flags=("state_diff_not_applied",),
        trace_id="trace_mira_search_archive_wrong_key_mira",
    )

    assert result.passed is True
    assert result.failure_reason is None


def test_evaluation_check_summary_schema() -> None:
    result = CaseEvaluationResult(
        case_id="reg_revise_incomplete_action",
        scenario_id="malformed_or_incomplete_action",
        expected_decision=VerificationDecision.REVISE,
        actual_decision=VerificationDecision.REVISE,
        passed=True,
        trace_id="trace_malformed_or_incomplete_action_ivo",
    )
    summary = EvaluationCheckSummary(
        trace_id="trace_malformed_or_incomplete_action_ivo",
        formal_experiment_result=False,
        case_count=1,
        passed_count=1,
        failed_count=0,
        decisions={"revise": 1},
        results=(result,),
    )

    assert summary.safe_dict()["formal_experiment_result"] is False
    assert summary.safe_dict()["passed_count"] == 1


def test_known_regression_case_passes() -> None:
    regression_result = run_regression_case(
        VALID_SEED,
        _case("mira_search_archive_wrong_key"),
    )
    trace = build_formal_trace_preview(
        regression_result.result,
        seed_id="mistgate_v01",
    )

    summary = evaluate_formal_trace_preview(trace)

    assert summary.case_count == 1
    assert summary.passed_count == 1
    assert summary.failed_count == 0
    assert summary.results[0].case_id == "reg_reject_mira_wrong_key"


def test_decision_mismatch_fails() -> None:
    trace = _single_record_trace(
        agent_id="mira",
        scenario_id="mira_search_archive_wrong_key",
        decision=VerificationDecision.COMMIT,
        committed_event_id="event_wrong_commit",
        state_diff_id="diff_wrong_commit",
    )

    summary = evaluate_formal_trace_preview(trace)

    assert summary.failed_count == 1
    assert summary.results[0].passed is False
    assert "decision mismatch" in (summary.results[0].failure_reason or "")


def test_commit_trace_requires_transition_causal_projection_and_evolution() -> None:
    trace = _single_record_trace(
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        decision=VerificationDecision.COMMIT,
        committed_event_id="committed_candidate_ivo_inspect_workshop_safe_fixture_ivo",
        state_diff_id="diff_committed_candidate_ivo_inspect_workshop_safe_fixture_ivo",
    )

    summary = evaluate_formal_trace_preview(trace)

    assert summary.failed_count == 1
    reason = summary.results[0].failure_reason or ""
    assert "commit trace missing state_transition" in reason
    assert "commit trace missing causal_projection" in reason
    assert "commit trace missing evolution_update" in reason


def test_non_commit_trace_rejects_transition_and_causal_projection() -> None:
    trace = _single_record_trace(
        agent_id="mira",
        scenario_id="mira_search_archive_wrong_key",
        decision=VerificationDecision.REJECT,
        state_transition={"applied_patch_count": 0},
        causal_projection={"committed_event_node_id": "event:invalid"},
    )

    summary = evaluate_formal_trace_preview(trace)

    assert summary.failed_count == 1
    reason = summary.results[0].failure_reason or ""
    assert "non-commit trace has state_transition" in reason
    assert "non-commit trace has causal_projection" in reason


def test_non_commit_trace_rejects_applied_evolution_updates() -> None:
    trace = _single_record_trace(
        agent_id="mira",
        scenario_id="mira_search_archive_wrong_key",
        decision=VerificationDecision.REJECT,
        evolution_update={"applied_update_count": 1},
    )

    summary = evaluate_formal_trace_preview(trace)

    assert summary.failed_count == 1
    assert "non-commit trace has applied evolution updates" in (
        summary.results[0].failure_reason or ""
    )


def test_unknown_scenario_fails() -> None:
    trace = _single_record_trace(
        agent_id="mira",
        scenario_id="unknown_scenario",
        decision=VerificationDecision.REJECT,
    )

    summary = evaluate_formal_trace_preview(trace)

    assert summary.failed_count == 1
    assert summary.results[0].passed is False
    assert summary.results[0].failure_reason == "unknown_regression_case"


def test_player_claim_requires_canon_updated_false() -> None:
    regression_result = run_regression_case(
        VALID_SEED,
        _case("player_claim_key_in_hand"),
    )
    trace = build_formal_trace_preview(
        regression_result.result,
        seed_id="mistgate_v01",
    )

    summary = evaluate_formal_trace_preview(trace)

    assert summary.failed_count == 0
    assert summary.results[0].case_id == "reg_player_claim_key_in_hand"
    assert summary.results[0].passed is True


def test_player_request_requires_no_direct_canon_or_world_mutation() -> None:
    regression_result = run_regression_case(
        VALID_SEED,
        _case("player_request_open_workshop_safe"),
    )
    trace = build_formal_trace_preview(
        regression_result.result,
        seed_id="mistgate_v01",
    )

    summary = evaluate_formal_trace_preview(trace)

    assert summary.failed_count == 0
    assert summary.results[0].case_id == "reg_player_request_open_workshop_safe"
    assert summary.results[0].passed is True


def test_player_input_direct_world_mutation_fails() -> None:
    trace = _single_record_trace(
        agent_id="player",
        scenario_id="player_request_open_workshop_safe",
        decision=VerificationDecision.PENDING_GATE,
        player_input_summary={
            "input_id": "player_request_open_workshop_safe",
            "player_id": "player",
            "input_kind": "request",
            "route": "event_candidate",
            "event_candidate_id": "candidate_player_request_open_workshop_safe",
            "verification_decision": "pending_gate",
            "canon_updated": False,
            "world_state_modified": True,
            "state_diff_id": None,
            "safety_flags": [],
        },
    )

    summary = evaluate_formal_trace_preview(trace)

    assert summary.failed_count == 1
    assert "player input modified world state directly" in (summary.results[0].failure_reason or "")


def test_agent_workflow_trace_requires_retrieval_proposal_candidate_and_source() -> None:
    trace = _single_record_trace(
        agent_id="mira",
        scenario_id="mira_search_archive_wrong_key",
        decision=VerificationDecision.REJECT,
        metadata={"activation_trace_included": True},
    )

    summary = evaluate_formal_trace_preview(trace)

    assert summary.failed_count == 1
    reason = summary.results[0].failure_reason or ""
    assert "agent workflow missing activation summary" in reason
    assert "agent workflow missing retrieval summary" in reason
    assert "agent workflow missing proposal summary" in reason
    assert "agent workflow missing proposal source" in reason
    assert "agent workflow missing event candidate summary" in reason


def test_run_step_agent_trace_projects_workflow_summaries() -> None:
    regression_result = run_regression_case(
        VALID_SEED,
        _case("mira_search_archive_wrong_key"),
    )
    trace = build_formal_trace_preview(
        regression_result.result,
        seed_id="mistgate_v01",
    )

    summary = evaluate_formal_trace_preview(trace)

    assert summary.failed_count == 0
    assert trace.records[0].retrieval_summary is not None
    assert trace.records[0].proposal_summary is not None
    assert trace.records[0].proposal_source == "deterministic_fixture"
    assert trace.records[0].candidate_summary is not None


def _case(scenario_id: str):
    from aethelis.evaluation.regression_cases import default_regression_cases

    return next(case for case in default_regression_cases() if case.scenario_id == scenario_id)


def _single_record_trace(
    *,
    agent_id: str,
    scenario_id: str,
    decision: VerificationDecision,
    committed_event_id: str | None = None,
    state_diff_id: str | None = None,
    state_transition: dict[str, object] | None = None,
    causal_projection: dict[str, object] | None = None,
    evolution_update: dict[str, object] | None = None,
    player_input_summary: dict[str, object] | None = None,
    activation_summary: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> FormalTraceEnvelope:
    return FormalTraceEnvelope(
        trace_id=f"trace_{scenario_id}_{agent_id}",
        formal_experiment_result=False,
        schema_version="0.1",
        seed_id="mistgate_v01",
        scenario_id=scenario_id,
        agent_id=agent_id,
        metadata=metadata or {},
        records=(
            TraceStepTransaction(
                step_id=f"step_{scenario_id}_{agent_id}",
                scenario_id=scenario_id,
                agent_id=agent_id,
                verification_decision=decision,
                committed_event_id=committed_event_id,
                state_diff_id=state_diff_id,
                state_diff_applied=False,
                state_transition=state_transition,
                causal_projection=causal_projection,
                evolution_update=evolution_update,
                player_input_summary=player_input_summary,
                activation_summary=activation_summary,
            ),
        ),
    )
