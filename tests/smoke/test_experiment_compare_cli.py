from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from aethelis.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
STANDARD_EXPERIMENT = ROOT / "configs" / "standard_experiment_deterministic_regression.yaml"


def test_experiment_compare_writes_variant_artifacts_without_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_if_provider_called(*args, **kwargs):
        raise AssertionError("experiment-compare must not call providers")

    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fail_if_provider_called,
    )
    run_id = "pytest_experiment_compare"
    config_path = _write_config(tmp_path, run_id=run_id)

    result = CliRunner().invoke(
        app,
        [
            "experiment-compare",
            "--seed",
            str(VALID_SEED),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert '"formal_experiment_result": true' in result.stdout
    assert '"provider_called": false' in result.stdout
    assert '"variant_count": 11' in result.stdout
    run_dir = ROOT / "runs" / run_id
    variant_results_path = run_dir / "variant_results.json"
    comparison_summary_path = run_dir / "comparison_summary.json"
    assert variant_results_path.exists()
    assert comparison_summary_path.exists()
    variant_results = json.loads(variant_results_path.read_text(encoding="utf-8"))
    comparison_summary = json.loads(comparison_summary_path.read_text(encoding="utf-8"))
    assert variant_results["variant_count"] == 11
    assert comparison_summary["provider_called"] is False
    assert "free_multi_agent_chat" in comparison_summary["canonical_baseline_ids"]
    assert "rag_memory_agents" in comparison_summary["canonical_baseline_ids"]
    assert "free_multi_agent_chat" not in comparison_summary["unsupported_variant_ids"]
    assert "without_world_pressure_field" not in comparison_summary["unsupported_variant_ids"]
    by_id = {variant["variant_id"]: variant for variant in variant_results["variants"]}
    assert by_id["aethelis_proposed_runtime"]["bad_case_count"] == 0
    assert by_id["free_multi_agent_chat"]["variant_type"] == "baseline"
    assert by_id["rag_memory_agents"]["variant_type"] == "baseline"
    assert by_id["without_event_verification"]["bad_case_count"] > 0
    assert by_id["without_causal_event_graph"]["bad_case_count"] > 0
    assert by_id["without_world_pressure_field"]["bad_case_count"] > 0
    assert by_id["without_player_input_verification"]["bad_case_count"] > 0
    assert Path(by_id["shared_context_agents"]["artifact_path"], "trace.json").exists()


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
