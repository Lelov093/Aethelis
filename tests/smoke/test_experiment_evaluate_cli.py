from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from aethelis.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
STANDARD_EXPERIMENT = ROOT / "configs" / "standard_experiment_deterministic_regression.yaml"


def test_experiment_evaluate_writes_metrics_and_bad_cases_without_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_if_provider_called(*args, **kwargs):
        raise AssertionError("experiment-evaluate must not call providers")

    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fail_if_provider_called,
    )
    runner = CliRunner()
    run_id = "pytest_formal_experiment_evaluate"
    config_path = _write_config(tmp_path, run_id=run_id)
    run_result = runner.invoke(
        app,
        [
            "experiment-run",
            "--seed",
            str(VALID_SEED),
            "--config",
            str(config_path),
        ],
    )
    assert run_result.exit_code == 0
    run_dir = ROOT / "runs" / run_id

    evaluate_result = runner.invoke(app, ["experiment-evaluate", "--run", str(run_dir)])

    assert evaluate_result.exit_code == 0
    assert '"formal_experiment_result": true' in evaluate_result.stdout
    assert '"metric_count": 10' in evaluate_result.stdout
    assert '"failed_metric_count": 0' in evaluate_result.stdout
    assert '"bad_case_count": 0' in evaluate_result.stdout
    assert '"provider_called": false' in evaluate_result.stdout
    metrics_path = run_dir / "metrics_summary.json"
    evaluation_path = run_dir / "evaluation_summary.json"
    bad_cases_path = run_dir / "bad_cases.json"
    assert metrics_path.exists()
    assert evaluation_path.exists()
    assert bad_cases_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    bad_cases = json.loads(bad_cases_path.read_text(encoding="utf-8"))
    assert metrics["formal_experiment_result"] is True
    assert metrics["failed_count"] == 0
    assert bad_cases["bad_case_count"] == 0
    assert bad_cases["bad_cases"] == []
    combined = (
        metrics_path.read_text(encoding="utf-8")
        + evaluation_path.read_text(encoding="utf-8")
        + bad_cases_path.read_text(encoding="utf-8")
    )
    assert "raw_llm_text" not in combined
    assert "prompt" not in combined.lower()
    assert "authorization" not in combined.lower()
    assert "sk-" not in combined


def test_experiment_evaluate_rejects_non_formal_run_directory(tmp_path: Path) -> None:
    runner = CliRunner()
    run_id = "pytest_formal_experiment_evaluate_reject"
    config_path = _write_config(tmp_path, run_id=run_id)
    run_result = runner.invoke(
        app,
        [
            "experiment-run",
            "--seed",
            str(VALID_SEED),
            "--config",
            str(config_path),
        ],
    )
    assert run_result.exit_code == 0
    run_dir = ROOT / "runs" / run_id
    trace_path = run_dir / "trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["formal_experiment_result"] = False
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    result = runner.invoke(app, ["experiment-evaluate", "--run", str(run_dir)])

    assert result.exit_code == 2
    assert "formal evaluation requires formal_experiment_result=true" in result.stderr


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
