from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from aethelis.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
STANDARD_RUN = ROOT / "configs" / "standard_run_deterministic_regression.yaml"


def test_run_world_cli_deterministic_preview_without_provider(monkeypatch) -> None:
    def fail_if_provider_called(*args, **kwargs):
        raise AssertionError("run-world must not call providers in Phase 2B")

    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fail_if_provider_called,
    )

    result = CliRunner().invoke(
        app,
        [
            "run-world",
            "--seed",
            str(VALID_SEED),
            "--config",
            str(STANDARD_RUN),
        ],
    )

    assert result.exit_code == 0
    assert '"mode": "deterministic_preview"' in result.stdout
    assert '"formal_experiment_result": false' in result.stdout
    assert '"provider_called": false' in result.stdout
    assert '"wrote_runs": false' in result.stdout
    assert '"wrote_reports": false' in result.stdout
    assert '"raw_text_saved": false' in result.stdout
    assert '"activation_trace_included": true' in result.stdout
    assert '"activation_mode": "static_trace"' in result.stdout
    assert '"activation_provider_called": false' in result.stdout
    assert '"proposal_summary"' in result.stdout
    assert '"candidate_summary"' in result.stdout
    assert '"generated_by": "deterministic_fixture"' in result.stdout
    assert '"can_modify_world_state": false' in result.stdout
    assert '"step_count": 8' in result.stdout
    assert '"route": "event_candidate"' in result.stdout
    assert "sk-" not in result.stdout
    assert "authorization" not in result.stdout.lower()


def test_run_world_cli_writes_tmp_trace_preview(tmp_path: Path) -> None:
    trace_path = tmp_path / "world_run_preview.json"

    result = CliRunner().invoke(
        app,
        [
            "run-world",
            "--seed",
            str(VALID_SEED),
            "--config",
            str(STANDARD_RUN),
            "--write-formal-trace-preview",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["trace_type"] == "formal"
    assert payload["formal_experiment_result"] is False
    assert payload["runtime_phase"] == "runtime_foundation_preview"
    assert payload["metadata"]["wrote_runs"] is False
    assert payload["metadata"]["wrote_reports"] is False
    assert payload["metadata"]["raw_text_saved"] is False
    assert payload["metadata"]["provider_called"] is False
    assert payload["metadata"]["activation_trace_included"] is True
    assert payload["metadata"]["activation_mode"] == "static_trace"
    assert payload["metadata"]["activation_provider_called"] is False
    assert payload["records"][0]["activation_summary"]["mode"] == "static_trace"
    assert payload["records"][0]["proposal_summary"]["generated_by"] == "deterministic_fixture"
    assert payload["records"][0]["candidate_summary"]["can_modify_world_state"] is False
    assert payload["records"][0]["candidate_summary"]["predicted_state_diff_id"] is None
    assert payload["records"][0]["verification_checks"]
    assert payload["records"][-1]["player_input_summary"]["route"] == "event_candidate"
    assert payload["records"][-1]["player_input_summary"]["canon_updated"] is False
    assert payload["records"][-1]["player_input_summary"]["world_state_modified"] is False
    raw = trace_path.read_text(encoding="utf-8")
    assert '"raw_llm_text"' not in raw
    assert '"full_raw_text"' not in raw
    assert '"raw_text_content"' not in raw
    assert "secret_" not in raw
    assert "calibration key is in the workshop safe" not in raw.lower()


def test_run_world_cli_rejects_real_llm_scenario(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_real_llm_run.yaml"
    config_path.write_text(
        """
run_id: bad_real_llm_run
mode: deterministic_preview
formal_experiment_result: false
allow_real_llm: false
dry_run: true
apply: false
step_plan:
  - step_id: step_ivo_real
    agent_id: ivo
    actor_type: agent
    scenario_id: inspect_workshop_safe
    allow_real_llm: false
    apply: false
""",
        encoding="utf-8",
    )

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

    assert result.exit_code == 2
    assert "scenario_requires_real_llm: inspect_workshop_safe" in result.stderr
    assert "sk-" not in result.stderr
