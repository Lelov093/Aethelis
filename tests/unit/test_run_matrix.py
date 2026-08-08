from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aethelis.experiments.matrix import (
    AggregateEvaluationConfig,
    RunMatrixConfig,
    RunMatrixEntry,
    _is_expected_failure,
    inspect_matrix_summary,
    load_aggregate_evaluation_config,
    load_run_matrix_config,
    render_matrix_review_markdown,
    run_matrix,
)

V04_MATRIX_CONFIG = Path("configs/v04_run_matrix.yaml")
V04_EVALUATION_CONFIG = Path("configs/v04_evaluation.yaml")
V05_MATRIX_CONFIG = Path("configs/v05_run_matrix.yaml")
V05_EVALUATION_CONFIG = Path("configs/v05_evaluation.yaml")
VARIANT_SEED = "seeds/mistgate_v01_civic_pressure_variant"


def test_v04_run_matrix_config_loads_mistgate_and_variant_entries() -> None:
    config = load_run_matrix_config(V04_MATRIX_CONFIG)
    evaluation = load_aggregate_evaluation_config(V04_EVALUATION_CONFIG)

    assert config.matrix_id == "v04_run_matrix"
    assert len(config.runs) == 2
    assert config.runs[0].seed == "seeds/mistgate_v01"
    assert config.runs[0].config == "configs/standard_experiment_deterministic_regression.yaml"
    assert config.runs[0].include_comparison is True
    assert config.runs[1].seed == VARIANT_SEED
    assert config.runs[1].random_seed == 11
    assert config.runs[1].include_comparison is True
    assert evaluation.allow_expected_ablation_failures is True
    assert evaluation.expected_failure_variant_ids == (
        "free_multi_agent_chat",
        "shared_context_agents",
        "rag_memory_agents",
        "rule_only_world_simulation",
    )
    assert evaluation.fail_on_provider_called is True
    assert evaluation.max_unexpected_bad_cases == 0


def test_v05_run_matrix_config_loads_cross_family_entries() -> None:
    config = load_run_matrix_config(V05_MATRIX_CONFIG)
    evaluation = load_aggregate_evaluation_config(V05_EVALUATION_CONFIG)

    assert config.matrix_id == "v05_run_matrix"
    assert len(config.runs) == 3
    assert {run.seed_family for run in config.runs} == {"mistgate", "harbor_lantern"}
    assert config.runs[2].seed == "seeds/harbor_lantern_v01"
    assert (
        config.runs[2].config
        == "configs/harbor_lantern_standard_experiment_deterministic_regression.yaml"
    )
    assert all(run.include_comparison is True for run in config.runs)
    assert evaluation.expected_failure_variant_ids == (
        "free_multi_agent_chat",
        "shared_context_agents",
        "rag_memory_agents",
        "rule_only_world_simulation",
    )
    assert evaluation.max_unexpected_bad_cases == 0


def test_run_matrix_writes_per_run_seed_variant_and_overall_summaries(tmp_path: Path) -> None:
    config = RunMatrixConfig(
        matrix_id="pytest_v04_matrix",
        artifact_dir=str(tmp_path / "runs" / "pytest_v04_matrix"),
        runs=(
            RunMatrixEntry(
                run_id="pytest_mistgate_seed7",
                seed="seeds/mistgate_v01",
                seed_family="mistgate",
                config="configs/standard_experiment_deterministic_regression.yaml",
                random_seed=7,
                step_count=20,
                include_comparison=True,
            ),
        ),
    )
    evaluation = AggregateEvaluationConfig(
        evaluation_id="pytest_v04_eval",
        expected_failure_variant_ids=(
            "free_multi_agent_chat",
            "shared_context_agents",
            "rag_memory_agents",
            "rule_only_world_simulation",
        ),
    )

    summary = run_matrix(config=config, evaluation_config=evaluation)

    assert summary.overall.passed is True
    assert summary.overall.run_count == 1
    assert summary.overall.seed_count == 1
    assert summary.overall.family_count == 1
    assert summary.overall.variant_count == 11
    assert summary.overall.provider_called_any is False
    assert summary.overall.raw_text_saved_any is False
    assert summary.runs[0].status == "passed"
    assert summary.seeds[0].seed == "seeds/mistgate_v01"
    assert summary.families[0].seed_family == "mistgate"
    assert {variant.variant_type for variant in summary.variants} == {
        "proposed",
        "baseline",
        "ablation",
    }
    assert summary.bad_cases
    assert summary.overall.bad_case_count == len(summary.bad_cases)
    assert summary.overall.expected_bad_case_count == len(summary.bad_cases)
    assert summary.overall.unexpected_bad_case_count == 0
    assert summary.overall.bad_case_taxonomy["canon_governance"] >= 1
    assert summary.overall.thresholds.max_unexpected_bad_cases == 0
    assert summary.bad_cases[0].seed == "seeds/mistgate_v01"
    assert summary.bad_cases[0].metric_name
    assert summary.bad_cases[0].governance_category
    assert Path(summary.matrix_summary_path).exists()
    assert Path(summary.aggregate_summary_path).exists()
    aggregate = json.loads(Path(summary.aggregate_summary_path).read_text(encoding="utf-8"))
    assert aggregate["bad_cases"]
    assert aggregate["families"][0]["seed_family"] == "mistgate"
    assert aggregate["overall"]["thresholds"]["max_unexpected_bad_cases"] == 0


def test_run_matrix_aggregates_two_seed_paths_without_provider(tmp_path: Path) -> None:
    config = RunMatrixConfig(
        matrix_id="pytest_v04_two_seed_matrix",
        artifact_dir=str(tmp_path / "runs" / "pytest_v04_two_seed_matrix"),
        runs=(
            RunMatrixEntry(
                run_id="pytest_mistgate_seed7",
                seed="seeds/mistgate_v01",
                seed_family="mistgate",
                config="configs/standard_experiment_deterministic_regression.yaml",
                random_seed=7,
                include_comparison=False,
            ),
            RunMatrixEntry(
                run_id="pytest_mistgate_variant_seed11",
                seed=VARIANT_SEED,
                seed_family="mistgate",
                config="configs/standard_experiment_deterministic_regression.yaml",
                random_seed=11,
                include_comparison=False,
            ),
        ),
    )

    summary = run_matrix(
        config=config,
        evaluation_config=AggregateEvaluationConfig(evaluation_id="pytest_v04_eval"),
    )

    assert summary.overall.passed is True
    assert summary.overall.run_count == 2
    assert summary.overall.seed_count == 2
    assert summary.overall.family_count == 1
    assert summary.overall.provider_called_any is False
    assert summary.overall.unexpected_variant_failure_count == 0
    assert {run.status for run in summary.runs} == {"passed"}
    assert {seed.seed for seed in summary.seeds} == {
        "seeds/mistgate_v01",
        VARIANT_SEED,
    }
    assert {family.seed_family for family in summary.families} == {"mistgate"}


def test_run_matrix_aggregates_cross_family_without_provider(tmp_path: Path) -> None:
    config = RunMatrixConfig(
        matrix_id="pytest_v05_cross_family_matrix",
        artifact_dir=str(tmp_path / "runs" / "pytest_v05_cross_family_matrix"),
        runs=(
            RunMatrixEntry(
                run_id="pytest_mistgate_seed7",
                seed="seeds/mistgate_v01",
                seed_family="mistgate",
                config="configs/standard_experiment_deterministic_regression.yaml",
                random_seed=7,
                include_comparison=True,
            ),
            RunMatrixEntry(
                run_id="pytest_mistgate_variant_seed11",
                seed=VARIANT_SEED,
                seed_family="mistgate",
                config="configs/standard_experiment_deterministic_regression.yaml",
                random_seed=11,
                include_comparison=True,
            ),
            RunMatrixEntry(
                run_id="pytest_harbor_seed13",
                seed="seeds/harbor_lantern_v01",
                seed_family="harbor_lantern",
                config="configs/harbor_lantern_standard_experiment_deterministic_regression.yaml",
                random_seed=13,
                include_comparison=True,
            ),
        ),
    )

    summary = run_matrix(
        config=config,
        evaluation_config=AggregateEvaluationConfig(
            evaluation_id="pytest_v05_eval",
            expected_failure_variant_ids=(
                "free_multi_agent_chat",
                "shared_context_agents",
                "rag_memory_agents",
                "rule_only_world_simulation",
            ),
        ),
    )

    assert summary.overall.passed is True
    assert summary.overall.run_count == 3
    assert summary.overall.seed_count == 3
    assert summary.overall.family_count == 2
    assert summary.overall.variant_count == 33
    assert summary.overall.provider_called_any is False
    assert summary.overall.raw_text_saved_any is False
    assert summary.overall.unexpected_variant_failure_count == 0
    assert summary.overall.unexpected_bad_case_count == 0
    assert {family.seed_family for family in summary.families} == {
        "mistgate",
        "harbor_lantern",
    }
    harbor = next(family for family in summary.families if family.seed_family == "harbor_lantern")
    assert harbor.run_count == 1
    assert harbor.seed_count == 1
    assert harbor.variant_count == 11
    assert all(run.provider_called is False for run in summary.runs)


def test_expected_ablation_failures_do_not_fail_aggregate(tmp_path: Path) -> None:
    config = RunMatrixConfig(
        matrix_id="pytest_v04_matrix_expected_ablation",
        artifact_dir=str(tmp_path / "runs" / "pytest_v04_matrix_expected_ablation"),
        runs=(
            RunMatrixEntry(
                run_id="pytest_mistgate_expected_ablation",
                seed="seeds/mistgate_v01",
                seed_family="mistgate",
                config="configs/standard_experiment_deterministic_regression.yaml",
                include_comparison=True,
            ),
        ),
    )

    summary = run_matrix(
        config=config,
        evaluation_config=AggregateEvaluationConfig(
            evaluation_id="pytest_v04_eval",
            allow_expected_ablation_failures=True,
            expected_failure_variant_ids=(
                "free_multi_agent_chat",
                "shared_context_agents",
                "rag_memory_agents",
                "rule_only_world_simulation",
            ),
        ),
    )

    assert summary.overall.passed is True
    assert summary.overall.unexpected_variant_failure_count == 0
    assert all(
        variant.status != "failed"
        for variant in summary.variants
        if variant.variant_type == "ablation"
    )
    assert summary.overall.expected_bad_case_count == summary.overall.bad_case_count
    assert summary.overall.unexpected_bad_case_count == 0


def test_matrix_inspect_returns_compact_review_without_raw_trace_body(tmp_path: Path) -> None:
    config = RunMatrixConfig(
        matrix_id="pytest_v06_inspect_matrix",
        artifact_dir=str(tmp_path / "runs" / "pytest_v06_inspect_matrix"),
        runs=(
            RunMatrixEntry(
                run_id="pytest_mistgate_inspect_seed7",
                seed="seeds/mistgate_v01",
                seed_family="mistgate",
                config="configs/standard_experiment_deterministic_regression.yaml",
                random_seed=7,
                include_comparison=True,
            ),
        ),
    )
    summary = run_matrix(
        config=config,
        evaluation_config=AggregateEvaluationConfig(
            evaluation_id="pytest_v06_inspect_eval",
            expected_failure_variant_ids=(
                "free_multi_agent_chat",
                "shared_context_agents",
                "rag_memory_agents",
                "rule_only_world_simulation",
            ),
        ),
    )

    review = inspect_matrix_summary(Path(summary.matrix_summary_path))
    markdown = render_matrix_review_markdown(review)

    assert review["matrix_id"] == "pytest_v06_inspect_matrix"
    assert review["run_count"] == 1
    assert review["provider_called_any"] is False
    assert review["raw_text_saved_any"] is False
    assert review["artifact_safety_passed"] is True
    assert review["proposed_runtime_failure_count"] == 0
    assert review["variant_status_by_type"]["proposed"]["passed"] == 1
    assert review["review_flags"]["unexpected_bad_case_count"] == 0
    assert "baseline and ablation expected failures remain review records" in markdown
    assert "raw_llm_text" not in markdown
    assert "records" not in review


def test_expected_failure_semantics_do_not_hide_proposed_runtime_failure() -> None:
    assert (
        _is_expected_failure(
            variant_id="aethelis_proposed_runtime",
            variant_type="proposed",
            failed_metric_count=1,
            allow_expected_ablation_failures=True,
            expected_failure_variant_ids=("aethelis_proposed_runtime",),
        )
        is False
    )
    assert (
        _is_expected_failure(
            variant_id="shared_context_agents",
            variant_type="baseline",
            failed_metric_count=1,
            allow_expected_ablation_failures=False,
            expected_failure_variant_ids=("shared_context_agents",),
        )
        is True
    )
    assert (
        _is_expected_failure(
            variant_id="rule_only_world_simulation",
            variant_type="baseline",
            failed_metric_count=1,
            allow_expected_ablation_failures=True,
            expected_failure_variant_ids=(),
        )
        is False
    )
    assert (
        _is_expected_failure(
            variant_id="without_event_verification",
            variant_type="ablation",
            failed_metric_count=1,
            allow_expected_ablation_failures=True,
            expected_failure_variant_ids=(),
        )
        is True
    )
    assert (
        _is_expected_failure(
            variant_id="without_event_verification",
            variant_type="ablation",
            failed_metric_count=1,
            allow_expected_ablation_failures=False,
            expected_failure_variant_ids=("without_event_verification",),
        )
        is False
    )


def test_run_matrix_rejects_invalid_artifact_dir(tmp_path: Path) -> None:
    config = RunMatrixConfig(
        matrix_id="bad_artifact_dir",
        artifact_dir=str(tmp_path / "not_runs"),
        runs=(
            RunMatrixEntry(
                run_id="bad_artifact_run",
                seed="seeds/mistgate_v01",
                seed_family="mistgate",
                config="configs/standard_experiment_deterministic_regression.yaml",
            ),
        ),
    )

    summary = run_matrix(
        config=config,
        evaluation_config=AggregateEvaluationConfig(evaluation_id="bad_artifact_eval"),
    )

    assert summary.overall.artifact_safety_passed is False
    assert summary.overall.passed is False


def test_run_matrix_requires_runs() -> None:
    with pytest.raises(ValidationError):
        RunMatrixConfig(matrix_id="empty_matrix", runs=())
