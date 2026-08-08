from __future__ import annotations

import json
from pathlib import Path

from aethelis.evaluation.harness import evaluate_formal_run
from aethelis.experiments.runner import load_formal_experiment_config
from aethelis.experiments.variants import DEFAULT_VARIANTS, run_experiment_comparison

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
STANDARD_EXPERIMENT = ROOT / "configs" / "standard_experiment_deterministic_regression.yaml"


def test_experiment_comparison_generates_controlled_variant_artifacts(tmp_path: Path) -> None:
    config = load_formal_experiment_config(
        _write_config(tmp_path, run_id="unit_experiment_comparison")
    )

    result = run_experiment_comparison(
        seed_path=VALID_SEED,
        config=config,
        runs_dir=tmp_path / "runs",
    )

    assert result.formal_experiment_result is True
    assert result.provider_called is False
    assert result.variant_count == len(DEFAULT_VARIANTS)
    assert result.artifacts.variant_results_path.exists()
    assert result.artifacts.comparison_summary_path.exists()
    variants = {variant.variant_id: variant for variant in result.variants}
    assert variants["aethelis_proposed_runtime"].variant_type.value == "proposed"
    assert variants["aethelis_proposed_runtime"].bad_case_count == 0
    assert variants["free_multi_agent_chat"].variant_type.value == "baseline"
    assert variants["shared_context_agents"].variant_type.value == "baseline"
    assert variants["rag_memory_agents"].variant_type.value == "baseline"
    assert variants["without_event_verification"].variant_type.value == "ablation"
    assert variants["without_causal_event_graph"].variant_type.value == "ablation"
    assert variants["without_world_pressure_field"].variant_type.value == "ablation"
    assert variants["without_player_input_verification"].bad_case_count > 0
    assert variants["without_causal_event_graph"].bad_case_count > 0
    assert variants["without_world_pressure_field"].bad_case_count > 0
    for variant in result.variants:
        assert Path(variant.artifact_path, "trace.json").exists()
        assert Path(variant.artifact_path, "metrics_summary.json").exists()
        assert Path(variant.artifact_path, "bad_cases.json").exists()
        evaluation = evaluate_formal_run(Path(variant.artifact_path))
        assert evaluation.formal_experiment_result is True
        assert evaluation.provider_called is False


def test_unsafe_ablation_is_simulated_without_state_apply(tmp_path: Path) -> None:
    config = load_formal_experiment_config(
        _write_config(tmp_path, run_id="unit_experiment_comparison_safety")
    )
    result = run_experiment_comparison(
        seed_path=VALID_SEED,
        config=config,
        runs_dir=tmp_path / "runs",
    )
    variant = next(
        item for item in result.variants if item.variant_id == "without_player_input_verification"
    )
    trace = json.loads(Path(variant.artifact_path, "trace.json").read_text(encoding="utf-8"))
    summary = json.loads(
        Path(variant.artifact_path, "run_summary.json").read_text(encoding="utf-8")
    )

    assert trace["metadata"]["controlled_variant"] is True
    assert trace["metadata"]["variant_implemented"] is True
    assert trace["metadata"]["provider_called"] is False
    assert summary["state_diff_applied_count"] == 0
    assert summary["provider_called"] is False
    assert any(
        "simulated_without_player_input_verification" in record["safety_flags"]
        for record in trace["records"]
    )
    assert all(
        not (
            record["verification_decision"] != "commit" and record.get("state_diff_applied") is True
        )
        for record in trace["records"]
    )


def test_baseline_scope_excludes_ablation_variants(tmp_path: Path) -> None:
    config = load_formal_experiment_config(
        _write_config(tmp_path, run_id="unit_experiment_baseline_scope")
    )
    scoped_config = config.model_copy(update={"comparison_variant_scope": "baselines"})

    result = run_experiment_comparison(
        seed_path=VALID_SEED,
        config=scoped_config,
        runs_dir=tmp_path / "runs",
    )

    assert result.variant_count == 5
    assert {variant.variant_type.value for variant in result.variants} == {
        "proposed",
        "baseline",
    }
    assert all(not variant.variant_id.startswith("without_") for variant in result.variants)


def test_ablation_scope_excludes_baseline_variants(tmp_path: Path) -> None:
    config = load_formal_experiment_config(
        _write_config(tmp_path, run_id="unit_experiment_ablation_scope")
    )
    scoped_config = config.model_copy(update={"comparison_variant_scope": "ablations"})

    result = run_experiment_comparison(
        seed_path=VALID_SEED,
        config=scoped_config,
        runs_dir=tmp_path / "runs",
    )

    assert result.variant_count == 7
    assert {variant.variant_type.value for variant in result.variants} == {
        "proposed",
        "ablation",
    }
    assert all(variant.variant_type.value != "baseline" for variant in result.variants)


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
