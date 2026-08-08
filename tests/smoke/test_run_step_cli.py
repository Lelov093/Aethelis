from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from aethelis.cli.app import app
from aethelis.llm.base import LLMResult
from aethelis.providers import ProviderAttempt

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
PROVIDER_FLAGS = [
    "--proposal-source",
    "provider_structured",
    "--provider-proposals-enabled",
    "--allow-real-provider",
]


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


def test_run_step_deterministic_path_does_not_require_env() -> None:
    result = CliRunner().invoke(
        app,
        [
            "run-step",
            "--seed",
            str(VALID_SEED),
            "--agent",
            "mira",
            "--scenario",
            "mira_search_archive_wrong_key",
        ],
    )

    assert result.exit_code == 0
    assert '"proposal_source": "deterministic_fixture"' in result.stdout
    assert '"provider_called": false' in result.stdout


def test_run_step_real_provider_default_requires_env_without_call(tmp_path: Path) -> None:
    missing_env = tmp_path / "missing.env"
    result = CliRunner().invoke(
        app,
        [
            "run-step",
            "--seed",
            str(VALID_SEED),
            "--agent",
            "ivo",
            "--scenario",
            "inspect_workshop_safe",
            "--env-file",
            str(missing_env),
        ],
    )

    assert result.exit_code == 2
    assert "Configuration file not found" in result.stderr
    assert "sk-" not in result.stderr


def test_run_step_cli_uses_safe_summary_without_real_llm(monkeypatch, tmp_path: Path) -> None:
    env_file = write_valid_env(tmp_path / ".env")

    def fake_generate(self, prompt: str, *, max_tokens: int = 32, temperature: float = 0.0):
        assert "canon_key_in_workshop_safe" not in prompt
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
            model="fake-test-model",
            latency_ms=1,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            attempts=(ProviderAttempt(model="fake-test-model", success=True, latency_ms=1),),
        )

    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fake_generate,
    )

    result = CliRunner().invoke(
        app,
        [
            "run-step",
            "--seed",
            str(VALID_SEED),
            "--agent",
            "ivo",
            "--scenario",
            "inspect_workshop_safe",
            "--env-file",
            str(env_file),
            *PROVIDER_FLAGS,
        ],
    )

    assert result.exit_code == 0
    assert '"decision": "commit"' in result.stdout
    assert '"dry_run": true' in result.stdout
    assert '"state_diff_applied": false' in result.stdout
    assert '"provider_mode": "real_provider"' in result.stdout
    assert '"evidence_class": "real_provider"' in result.stdout
    assert '"action_proposal_summary"' in result.stdout
    assert '"proposer_agent_id": "ivo"' in result.stdout
    assert '"intent": "investigate"' in result.stdout
    assert '"contains_state_diff": false' in result.stdout
    assert '"contains_canon_mutation": false' in result.stdout
    assert "sk-test-openai-secret" not in result.stdout
    assert "proposal_inspect_workshop_safe_ivo" in result.stdout
    assert "Inspect the workshop safe using Ivo own lawful access" not in result.stdout


def test_run_step_cli_apply_and_debug_trace_are_explicit(monkeypatch, tmp_path: Path) -> None:
    env_file = write_valid_env(tmp_path / ".env")
    trace_path = tmp_path / "ivo_step_debug.json"

    def fake_generate(self, prompt: str, *, max_tokens: int = 32, temperature: float = 0.0):
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
            model="fake-test-model",
            latency_ms=1,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            attempts=(ProviderAttempt(model="fake-test-model", success=True, latency_ms=1),),
        )

    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fake_generate,
    )

    dry_run_result = CliRunner().invoke(
        app,
        [
            "run-step",
            "--seed",
            str(VALID_SEED),
            "--agent",
            "ivo",
            "--scenario",
            "inspect_workshop_safe",
            "--env-file",
            str(env_file),
            *PROVIDER_FLAGS,
        ],
    )
    assert dry_run_result.exit_code == 0
    assert not trace_path.exists()
    assert '"dry_run": true' in dry_run_result.stdout
    assert '"state_diff_applied": false' in dry_run_result.stdout

    apply_result = CliRunner().invoke(
        app,
        [
            "run-step",
            "--seed",
            str(VALID_SEED),
            "--agent",
            "ivo",
            "--scenario",
            "inspect_workshop_safe",
            "--env-file",
            str(env_file),
            *PROVIDER_FLAGS,
            "--apply",
            "--write-debug-trace",
            str(trace_path),
        ],
    )
    assert apply_result.exit_code == 0
    assert '"dry_run": false' in apply_result.stdout
    assert '"state_diff_applied": true' in apply_result.stdout
    assert '"applied_patch_count": 1' in apply_result.stdout
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["trace_type"] == "debug"
    assert trace["formal_experiment_result"] is False
    assert trace["state_diff_applied"] is True
    assert "sk-test-openai-secret" not in trace_path.read_text(encoding="utf-8")
    assert "Inspect the workshop safe using Ivo own lawful access" not in trace_path.read_text(
        encoding="utf-8"
    )


def test_non_ivo_scenarios_are_deterministic_and_observable(monkeypatch, tmp_path: Path) -> None:
    env_file = write_valid_env(tmp_path / ".env")

    def fail_if_real_llm_called(
        self,
        prompt: str,
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ):
        raise AssertionError("deterministic scenarios must not call the real LLM provider")

    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fail_if_real_llm_called,
    )

    cases = [
        ("mira", "mira_search_archive_wrong_key", "reject"),
        ("selka", "selka_consume_stabilizer_part_fixture", "commit"),
        ("selka", "selka_restock_market_credit_fixture", "commit"),
        ("rowan", "unsafe_force_open_safe", "pending_gate"),
        ("ivo", "malformed_or_incomplete_action", "revise"),
        ("player", "player_claim_key_in_hand", "reject"),
    ]
    for agent_id, scenario_id, decision in cases:
        result = CliRunner().invoke(
            app,
            [
                "run-step",
                "--seed",
                str(VALID_SEED),
                "--agent",
                agent_id,
                "--scenario",
                scenario_id,
                "--env-file",
                str(env_file),
                "--apply",
            ],
        )
        assert result.exit_code == 0
        assert f'"decision": "{decision}"' in result.stdout
        if decision == "commit":
            assert '"committed_event_id": "committed_' in result.stdout
            assert '"state_diff_applied": true' in result.stdout
        else:
            assert '"committed_event_id": null' in result.stdout
            assert '"state_diff_id": null' in result.stdout
            assert '"state_diff_applied": false' in result.stdout
        assert '"model_name": null' in result.stdout
        assert "sk-test-openai-secret" not in result.stdout


def test_run_step_writes_formal_trace_preview_and_trace_tools(monkeypatch, tmp_path: Path) -> None:
    env_file = write_valid_env(tmp_path / ".env")
    trace_path = tmp_path / "ivo_formal_preview.json"

    def fake_generate(self, prompt: str, *, max_tokens: int = 32, temperature: float = 0.0):
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
            model="fake-test-model",
            latency_ms=1,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            attempts=(ProviderAttempt(model="fake-test-model", success=True, latency_ms=1),),
        )

    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fake_generate,
    )

    run_result = CliRunner().invoke(
        app,
        [
            "run-step",
            "--seed",
            str(VALID_SEED),
            "--agent",
            "ivo",
            "--scenario",
            "inspect_workshop_safe",
            "--env-file",
            str(env_file),
            *PROVIDER_FLAGS,
            "--write-formal-trace-preview",
            str(trace_path),
        ],
    )
    assert run_result.exit_code == 0
    assert trace_path.exists()

    validate_result = CliRunner().invoke(app, ["trace-validate", str(trace_path)])
    inspect_result = CliRunner().invoke(app, ["trace-inspect", str(trace_path)])

    assert validate_result.exit_code == 0
    assert inspect_result.exit_code == 0
    assert '"trace_type": "formal"' in validate_result.stdout
    assert '"formal_experiment_result": false' in inspect_result.stdout
    assert '"decisions": [' in inspect_result.stdout
    assert "Inspect the workshop safe using Ivo own lawful access" not in inspect_result.stdout
    assert "sk-test-openai-secret" not in trace_path.read_text(encoding="utf-8")
