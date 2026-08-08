from __future__ import annotations

from pathlib import Path

from aethelis.evaluation import (
    default_regression_cases,
    formal_trace_to_evaluation_inputs,
    run_regression_case,
)
from aethelis.runtime.single_step import run_single_step
from aethelis.trace.formal import (
    build_formal_trace_preview,
    inspect_formal_trace_file,
    validate_formal_trace_file,
    write_formal_trace_preview,
)

VALID_SEED = Path("seeds/mistgate_v01")


def test_formal_trace_preview_writer_creates_parent_and_is_safe(tmp_path: Path) -> None:
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="mira",
        scenario_id="mira_search_archive_wrong_key",
        settings=object(),
    )
    trace_path = tmp_path / "nested" / "formal_preview.json"

    written = write_formal_trace_preview(result, trace_path, seed_id="mistgate_v01")
    raw = written.read_text(encoding="utf-8")
    report = validate_formal_trace_file(written)

    assert written == trace_path.resolve()
    assert report.success
    assert report.trace_type == "formal"
    assert report.formal_experiment_result is False
    assert report.has_raw_text is False
    assert report.has_secret_markers is False
    assert "raw_llm_text" not in raw
    assert "sk-" not in raw


def test_trace_validation_failure_and_safe_inspection(tmp_path: Path) -> None:
    trace_path = tmp_path / "invalid.json"
    trace_path.write_text(
        '{"trace_type": "formal", "formal_experiment_result": true}',
        encoding="utf-8",
    )

    validation = validate_formal_trace_file(trace_path)
    inspection = inspect_formal_trace_file(trace_path)

    assert not validation.success
    assert inspection["success"] is False
    assert "records" not in inspection


def test_evaluation_input_adapter_from_formal_trace() -> None:
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="player",
        scenario_id="player_claim_key_in_hand",
        settings=object(),
    )
    trace = build_formal_trace_preview(result, seed_id="mistgate_v01")

    inputs = formal_trace_to_evaluation_inputs(trace, case_id_prefix="reg")

    assert len(inputs) == 1
    assert inputs[0].scenario_id == "player_claim_key_in_hand"
    assert inputs[0].verification_decision.value == "reject"
    assert not inputs[0].committed_event_present
    assert not inputs[0].state_diff_present
    assert not inputs[0].state_diff_applied
    assert inputs[0].verification_check_count == 0
    assert inputs[0].verification_risk_flags == ("unverified_player_claim",)


def test_default_regression_case_pack_passes_without_real_llm() -> None:
    results = [run_regression_case(VALID_SEED, case) for case in default_regression_cases()]

    assert {result.case.expected_decision.value for result in results} == {
        "commit",
        "reject",
        "revise",
        "pending_gate",
    }
    assert all(result.passed for result in results)
    assert len(results) == 9
    assert results[0].result.structured_output is not None
    assert results[0].result.structured_output.provider_name == "fixture_test_provider"
