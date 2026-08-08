from __future__ import annotations

import importlib
from pathlib import Path

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
CLI_APP = importlib.import_module("aethelis.cli.app")


def write_valid_env(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=https://llm.example.test/v1",
                "OPENAI_MODEL=primary-model",
                "OPENAI_MODEL_FALLBACKS=",
                "OPENAI_API_KEY=sk-test-openai-secret",
                "EMBEDDING_PROVIDER=volcengine_ark",
                "EMBEDDING_BASE_URL=https://embedding.example.test/api/v3",
                "EMBEDDING_MODEL=embedding-model",
                "EMBEDDING_API_KEY=sk-test-embedding-secret",
                "EMBEDDING_DIMENSIONS=1024",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_evaluation_check_cli_safe_summary_and_no_runtime_rerun(
    monkeypatch,
    tmp_path: Path,
) -> None:
    env_file = write_valid_env(tmp_path / ".env")
    trace_path = tmp_path / "mira_formal_preview.json"
    runner = CliRunner()

    run_result = runner.invoke(
        CLI_APP.app,
        [
            "run-step",
            "--seed",
            str(VALID_SEED),
            "--agent",
            "mira",
            "--scenario",
            "mira_search_archive_wrong_key",
            "--env-file",
            str(env_file),
            "--write-formal-trace-preview",
            str(trace_path),
        ],
    )
    assert run_result.exit_code == 0

    def fail_if_runtime_called(*args, **kwargs):
        raise AssertionError("evaluation-check must not rerun scenarios")

    def fail_if_llm_called(*args, **kwargs):
        raise AssertionError("evaluation-check must not call the LLM provider")

    monkeypatch.setattr(CLI_APP, "run_single_step", fail_if_runtime_called)
    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fail_if_llm_called,
    )
    before_runs = _file_snapshot(ROOT / "runs")
    before_reports = _file_snapshot(ROOT / "reports")

    result = runner.invoke(CLI_APP.app, ["evaluation-check", "--trace", str(trace_path)])

    assert result.exit_code == 0
    assert '"case_count": 1' in result.stdout
    assert '"passed_count": 1' in result.stdout
    assert '"failed_count": 0' in result.stdout
    assert '"formal_experiment_result": false' in result.stdout
    assert '"reg_reject_mira_wrong_key"' in result.stdout
    assert "raw_text" not in result.stdout
    assert "sk-test-openai-secret" not in result.stdout
    assert "authorization" not in result.stdout.lower()
    assert _file_snapshot(ROOT / "runs") == before_runs
    assert _file_snapshot(ROOT / "reports") == before_reports


def test_evaluation_check_cli_invalid_trace_nonzero(tmp_path: Path) -> None:
    trace_path = tmp_path / "invalid_trace.json"
    trace_path.write_text(
        '{"trace_type":"formal","raw_text":"secret raw output"}',
        encoding="utf-8",
    )

    result = CliRunner().invoke(CLI_APP.app, ["evaluation-check", "--trace", str(trace_path)])

    assert result.exit_code == 1
    assert '"success": false' in result.stdout
    assert '"has_raw_text": true' in result.stdout
    assert "secret raw output" not in result.stdout


def test_evaluation_check_cli_unknown_scenario_nonzero(tmp_path: Path) -> None:
    trace_path = tmp_path / "unknown_formal_preview.json"
    trace_path.write_text(
        """{
  "trace_id": "trace_unknown_scenario_mira",
  "trace_type": "formal",
  "formal_experiment_result": false,
  "schema_version": "0.1",
  "seed_id": "mistgate_v01",
  "scenario_id": "unknown_scenario",
  "agent_id": "mira",
  "records": [
    {
      "step_id": "step_unknown_scenario_mira",
      "scenario_id": "unknown_scenario",
      "agent_id": "mira",
      "verification_decision": "reject",
      "state_diff_applied": false
    }
  ]
}""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(CLI_APP.app, ["evaluation-check", "--trace", str(trace_path)])

    assert result.exit_code == 1
    assert '"failed_count": 1' in result.stdout
    assert "unknown_regression_case" in result.stdout


def _file_snapshot(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(sorted(str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()))
