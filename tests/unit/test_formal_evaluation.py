from __future__ import annotations

import json
from pathlib import Path

import pytest

from aethelis.evaluation.harness import FormalEvaluationError, evaluate_formal_run
from aethelis.experiments.runner import load_formal_experiment_config, run_formal_experiment

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
STANDARD_EXPERIMENT = ROOT / "configs" / "standard_experiment_deterministic_regression.yaml"


def test_formal_run_evaluation_writes_metrics_and_empty_bad_cases(tmp_path: Path) -> None:
    config = load_formal_experiment_config(
        _write_config(tmp_path, run_id="unit_formal_evaluation_clean")
    )
    result = run_formal_experiment(
        seed_path=VALID_SEED,
        config=config,
        runs_dir=tmp_path / "runs",
    )

    summary = evaluate_formal_run(result.artifacts.run_dir)

    assert summary.formal_experiment_result is True
    assert summary.metric_count == 10
    assert summary.failed_metric_count == 0
    assert summary.bad_case_count == 0
    assert summary.provider_called is False
    assert summary.raw_text_saved is False
    metrics = json.loads(Path(summary.metrics_summary_path).read_text(encoding="utf-8"))
    evaluation = json.loads(Path(summary.evaluation_summary_path).read_text(encoding="utf-8"))
    bad_cases = json.loads(Path(summary.bad_cases_path).read_text(encoding="utf-8"))
    assert metrics["formal_experiment_result"] is True
    assert metrics["failed_count"] == 0
    assert metrics["canonical_metrics"]["knowledge_boundary_accuracy"] == "pass"
    assert metrics["canonical_metrics"]["causal_coherence"] == "pass"
    assert metrics["canonical_rates"]["invalid_event_rate"] == 0
    assert metrics["canonical_rates"]["canon_violation_rate"] == 0
    assert evaluation["formal_experiment_result"] is True
    assert bad_cases["bad_case_count"] == 0
    assert bad_cases["bad_cases"] == []


def test_formal_evaluation_rejects_preview_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "preview_like_run"
    run_dir.mkdir()
    config = load_formal_experiment_config(
        _write_config(tmp_path, run_id="unit_formal_evaluation_preview_reject")
    )
    result = run_formal_experiment(
        seed_path=VALID_SEED,
        config=config,
        runs_dir=tmp_path / "formal_runs",
    )
    for artifact_name in ("trace.json", "run_summary.json", "config_snapshot.json"):
        payload = json.loads((result.artifacts.run_dir / artifact_name).read_text(encoding="utf-8"))
        payload["formal_experiment_result"] = False
        (run_dir / artifact_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    with pytest.raises(FormalEvaluationError, match="formal_experiment_result=true"):
        evaluate_formal_run(run_dir)


def test_metric_failure_creates_bad_case(tmp_path: Path) -> None:
    config = load_formal_experiment_config(
        _write_config(tmp_path, run_id="unit_formal_evaluation_bad_case")
    )
    result = run_formal_experiment(
        seed_path=VALID_SEED,
        config=config,
        runs_dir=tmp_path / "runs",
    )
    trace_path = result.artifacts.run_dir / "trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    for record in trace["records"]:
        if record.get("player_input_summary") is not None:
            record["player_input_summary"]["canon_updated"] = True
            break
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = evaluate_formal_run(result.artifacts.run_dir)

    assert summary.failed_metric_count >= 1
    assert summary.bad_case_count >= 1
    bad_cases = json.loads(Path(summary.bad_cases_path).read_text(encoding="utf-8"))
    failure_types = {case["failure_type"] for case in bad_cases["bad_cases"]}
    assert "canon_safety" in failure_types
    assert "player_input_governance_safety" in failure_types


def test_formal_evaluation_artifacts_do_not_contain_raw_text_or_secrets(tmp_path: Path) -> None:
    config = load_formal_experiment_config(
        _write_config(tmp_path, run_id="unit_formal_evaluation_safe_artifacts")
    )
    result = run_formal_experiment(
        seed_path=VALID_SEED,
        config=config,
        runs_dir=tmp_path / "runs",
    )

    summary = evaluate_formal_run(result.artifacts.run_dir)

    combined = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            summary.metrics_summary_path,
            summary.evaluation_summary_path,
            summary.bad_cases_path,
        )
    )
    unsafe_markers = (
        "raw_llm_text",
        "full_raw_text",
        "prompt",
        "authorization",
        ".env",
        "sk-",
    )
    for marker in unsafe_markers:
        assert marker not in combined.lower()


def _write_config(tmp_path: Path, *, run_id: str) -> Path:
    path = tmp_path / f"{run_id}.yaml"
    path.write_text(
        STANDARD_EXPERIMENT.read_text(encoding="utf-8").replace(
            "standard_mistgate_formal_experiment",
            run_id,
        ),
        encoding="utf-8",
    )
    return path
