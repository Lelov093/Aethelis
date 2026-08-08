from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from aethelis.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
STANDARD_RUN = ROOT / "configs" / "standard_run_deterministic_regression.yaml"
STANDARD_EXPERIMENT = ROOT / "configs" / "standard_experiment_deterministic_regression.yaml"


def test_experiment_run_writes_formal_artifacts_without_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_if_provider_called(*args, **kwargs):
        raise AssertionError("experiment-run formal deterministic mode must not call providers")

    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fail_if_provider_called,
    )
    config_path = tmp_path / "experiment.yaml"
    run_id = "pytest_formal_experiment"
    config_path.write_text(
        STANDARD_EXPERIMENT.read_text(encoding="utf-8").replace(
            "standard_mistgate_formal_experiment",
            run_id,
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "experiment-run",
            "--seed",
            str(VALID_SEED),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert '"formal_experiment_result": true' in result.stdout
    assert '"provider_called": false' in result.stdout
    assert '"baseline_implemented": false' in result.stdout
    assert '"ablation_implemented": false' in result.stdout
    run_dir = ROOT / "runs" / run_id
    trace_path = run_dir / "trace.json"
    summary_path = run_dir / "run_summary.json"
    config_snapshot_path = run_dir / "config_snapshot.json"
    assert trace_path.exists()
    assert summary_path.exists()
    assert config_snapshot_path.exists()

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    snapshot = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
    assert trace["formal_experiment_result"] is True
    assert trace["runtime_phase"] == "phase_7_formal_experiment"
    assert trace["metadata"]["trace_source"] == "experiment-run"
    assert trace["metadata"]["wrote_runs"] is True
    assert trace["metadata"]["wrote_reports"] is False
    assert trace["metadata"]["raw_text_saved"] is False
    assert trace["metadata"]["provider_called"] is False
    assert len(trace["records"]) == 20
    assert trace["records"][0]["safety_flags"][-1] == "formal_experiment_result_true"
    assert summary["formal_experiment_result"] is True
    assert summary["step_count"] == 20
    assert summary["plan_source"] == "scenario_cycle"
    assert summary["scenario_distribution"]["ivo_inspect_workshop_safe_fixture"] == 3
    assert summary["scenario_distribution"]["selka_consume_stabilizer_part_fixture"] == 3
    assert summary["scenario_distribution"]["selka_restock_market_credit_fixture"] == 3
    assert snapshot["formal_experiment_result"] is True
    raw = trace_path.read_text(encoding="utf-8")
    assert "raw_llm_text" not in raw
    assert "full_raw_text" not in raw
    assert "raw_text_content" not in raw
    assert "authorization" not in raw.lower()
    assert "sk-" not in raw

    validate_result = CliRunner().invoke(app, ["trace-validate", str(trace_path)])
    evaluation_result = CliRunner().invoke(app, ["evaluation-check", "--trace", str(trace_path)])
    assert validate_result.exit_code == 0
    assert '"formal_experiment_result": true' in validate_result.stdout
    assert evaluation_result.exit_code == 0
    assert '"formal_experiment_result": true' in evaluation_result.stdout
    assert '"case_count": 20' in evaluation_result.stdout
    assert '"failed_count": 0' in evaluation_result.stdout


def test_run_world_preview_does_not_write_runs(tmp_path: Path) -> None:
    run_id = "preview_must_not_write_runs"
    config_path = tmp_path / "preview.yaml"
    config_path.write_text(
        STANDARD_RUN.read_text(encoding="utf-8").replace(
            "standard_mistgate_deterministic_preview",
            run_id,
        ),
        encoding="utf-8",
    )
    before = _file_snapshot(ROOT / "runs")

    result = CliRunner().invoke(
        app,
        [
            "run-world",
            "--seed",
            str(VALID_SEED),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert '"formal_experiment_result": false' in result.stdout
    assert '"wrote_runs": false' in result.stdout
    assert _file_snapshot(ROOT / "runs") == before


def test_experiment_run_rejects_invalid_step_count(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_experiment.yaml"
    config_path.write_text(
        STANDARD_EXPERIMENT.read_text(encoding="utf-8").replace("step_count: 20", "step_count: 19"),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "experiment-run",
            "--seed",
            str(VALID_SEED),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 2
    assert "ValidationError: experiment config schema invalid" in result.stderr


def _file_snapshot(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(sorted(str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()))
