from __future__ import annotations

from pathlib import Path

from aethelis.experiments.matrix import (
    AggregateEvaluationConfig,
    RunMatrixConfig,
    RunMatrixEntry,
    load_matrix_summary,
    run_matrix,
)
from aethelis.experiments.regression import (
    RegressionTargetConfig,
    RegressionTargetKind,
    RuntimeRegressionConfig,
    load_runtime_regression_config,
    run_runtime_regression,
)

REGRESSION_CONFIG = Path("configs/v02_runtime_regression.yaml")
V03_REGRESSION_CONFIG = Path("configs/v03_runtime_regression.yaml")
V04_REGRESSION_CONFIG = Path("configs/v04_runtime_regression.yaml")
V05_REGRESSION_CONFIG = Path("configs/v05_runtime_regression.yaml")
V06_REGRESSION_CONFIG = Path("configs/v06_runtime_regression.yaml")
V07_REGRESSION_CONFIG = Path("configs/v07_runtime_regression.yaml")
V08_REGRESSION_CONFIG = Path("configs/v08_runtime_regression.yaml")


def test_runtime_regression_config_loads_deterministic_targets() -> None:
    config = load_runtime_regression_config(REGRESSION_CONFIG)

    assert config.regression_id == "v02_runtime_regression"
    assert [target.kind for target in config.targets] == [
        RegressionTargetKind.RUN_WORLD_PREVIEW,
        RegressionTargetKind.RUN_WORLD_APPLY,
        RegressionTargetKind.EXPERIMENT_RUN,
        RegressionTargetKind.TRACE_VALIDATE,
        RegressionTargetKind.EVALUATION_CHECK,
        RegressionTargetKind.EXPERIMENT_EVALUATE,
        RegressionTargetKind.EXPERIMENT_COMPARE,
    ]


def test_v03_runtime_regression_config_adds_contract_targets() -> None:
    config = load_runtime_regression_config(V03_REGRESSION_CONFIG)

    assert config.regression_id == "v03_runtime_regression"
    assert len(config.targets) == 9
    assert [target.id for target in config.targets[-2:]] == [
        "state_apply_contracts",
        "scheduler_evolution_contracts",
    ]
    assert [target.kind for target in config.targets[-2:]] == [
        RegressionTargetKind.STATE_APPLY_CONTRACTS,
        RegressionTargetKind.SCHEDULER_EVOLUTION_CONTRACTS,
    ]


def test_v04_runtime_regression_config_adds_matrix_targets() -> None:
    config = load_runtime_regression_config(V04_REGRESSION_CONFIG)

    assert config.regression_id == "v04_runtime_regression"
    assert [target.kind for target in config.targets] == [
        RegressionTargetKind.MATRIX_CONFIG_LOAD,
        RegressionTargetKind.MATRIX_RUN,
        RegressionTargetKind.AGGREGATE_SUMMARY,
        RegressionTargetKind.ARTIFACT_SAFETY,
    ]


def test_v05_runtime_regression_config_adds_cross_family_matrix_targets() -> None:
    config = load_runtime_regression_config(V05_REGRESSION_CONFIG)

    assert config.regression_id == "v05_runtime_regression"
    assert [target.kind for target in config.targets] == [
        RegressionTargetKind.MATRIX_CONFIG_LOAD,
        RegressionTargetKind.MATRIX_RUN,
        RegressionTargetKind.AGGREGATE_SUMMARY,
        RegressionTargetKind.ARTIFACT_SAFETY,
    ]
    assert config.targets[0].matrix == "configs/v05_run_matrix.yaml"
    assert config.targets[1].evaluation == "configs/v05_evaluation.yaml"


def test_v06_runtime_regression_config_adds_matrix_inspect_targets() -> None:
    config = load_runtime_regression_config(V06_REGRESSION_CONFIG)

    assert config.regression_id == "v06_runtime_regression"
    assert [target.kind for target in config.targets] == [
        RegressionTargetKind.MATRIX_INSPECT_LOAD,
        RegressionTargetKind.MATRIX_INSPECT_REVIEW_PROTOCOL,
        RegressionTargetKind.MATRIX_INSPECT_SAFETY,
    ]
    assert all(
        target.summary == "runs/v05_run_matrix/matrix_summary.json" for target in config.targets
    )


def test_v07_runtime_regression_config_adds_scenario_contract_targets() -> None:
    config = load_runtime_regression_config(V07_REGRESSION_CONFIG)

    assert config.regression_id == "v07_runtime_regression"
    assert [target.kind for target in config.targets] == [
        RegressionTargetKind.SCENARIO_CONTRACT_LOAD,
        RegressionTargetKind.SCENARIO_CONTRACT_FIXTURE_COVERAGE,
        RegressionTargetKind.SCENARIO_CONTRACT_VERIFIER_RULE_PACK_PARITY,
        RegressionTargetKind.SCENARIO_CONTRACT_MATRIX_COMPATIBILITY,
        RegressionTargetKind.SCENARIO_CONTRACT_NO_PROVIDER,
    ]
    assert config.targets[3].summary == "runs/v05_run_matrix/matrix_summary.json"


def test_v08_runtime_regression_config_adds_provider_proposal_targets() -> None:
    config = load_runtime_regression_config(V08_REGRESSION_CONFIG)

    assert config.regression_id == "v08_runtime_regression"
    assert [target.kind for target in config.targets] == [
        RegressionTargetKind.PROVIDER_PROPOSAL_MISSING_SETTINGS_STOPS_BEFORE_CHAIN,
        RegressionTargetKind.PROVIDER_PROPOSAL_SCHEMA_FAILURE_NO_EVENT_CANDIDATE,
        RegressionTargetKind.PROVIDER_PROPOSAL_GOVERNANCE_CHAIN,
        RegressionTargetKind.PROVIDER_PROPOSAL_NO_RAW_TEXT_ARTIFACTS,
        RegressionTargetKind.DETERMINISTIC_REGRESSIONS_STILL_NO_PROVIDER,
    ]


def test_v03_contract_targets_pass_without_provider(tmp_path: Path) -> None:
    config = RuntimeRegressionConfig(
        regression_id="pytest_v03_contracts",
        artifact_dir=str(tmp_path / "runs" / "pytest_v03_contracts"),
        targets=(
            RegressionTargetConfig(
                id="state_apply_contracts",
                kind=RegressionTargetKind.STATE_APPLY_CONTRACTS,
                seed="seeds/mistgate_v01",
            ),
            RegressionTargetConfig(
                id="scheduler_evolution_contracts",
                kind=RegressionTargetKind.SCHEDULER_EVOLUTION_CONTRACTS,
                seed="seeds/mistgate_v01",
                config="configs/standard_run_deterministic_regression.yaml",
            ),
        ),
    )

    summary = run_runtime_regression(config=config)

    assert summary.target_count == 2
    assert summary.passed_count == 2
    assert summary.failed_count == 0
    assert summary.provider_called_any is False
    assert all(target.failed_metric_count == 0 for target in summary.targets)


def test_v04_matrix_targets_pass_without_provider(tmp_path: Path) -> None:
    matrix_summary = tmp_path / "runs" / "pytest_v04_matrix" / "matrix_summary.json"
    config = RuntimeRegressionConfig(
        regression_id="pytest_v04_matrix_regression",
        artifact_dir=str(tmp_path / "runs" / "pytest_v04_regression"),
        targets=(
            RegressionTargetConfig(
                id="v04_matrix_config_load",
                kind=RegressionTargetKind.MATRIX_CONFIG_LOAD,
                matrix="configs/v04_run_matrix.yaml",
                evaluation="configs/v04_evaluation.yaml",
            ),
            RegressionTargetConfig(
                id="v04_matrix_run",
                kind=RegressionTargetKind.MATRIX_RUN,
                matrix="configs/v04_run_matrix.yaml",
                evaluation="configs/v04_evaluation.yaml",
            ),
            RegressionTargetConfig(
                id="v04_aggregate_summary",
                kind=RegressionTargetKind.AGGREGATE_SUMMARY,
                summary="runs/v04_run_matrix/matrix_summary.json",
            ),
        ),
    )

    summary = run_runtime_regression(config=config)

    assert summary.target_count == 3
    assert summary.passed_count == 3
    assert summary.failed_count == 0
    assert summary.provider_called_any is False
    aggregate_target = next(
        target
        for target in summary.targets
        if target.kind == RegressionTargetKind.AGGREGATE_SUMMARY
    )
    assert aggregate_target.metric_count == 11
    matrix = load_matrix_summary(Path("runs/v04_run_matrix/matrix_summary.json"))
    assert matrix.bad_cases
    assert matrix.overall.bad_case_taxonomy
    assert matrix.overall.unexpected_bad_case_count == 0
    assert matrix.overall.thresholds.max_unexpected_bad_cases == 0
    assert matrix_summary.name == "matrix_summary.json"


def test_v05_matrix_targets_pass_without_provider(tmp_path: Path) -> None:
    matrix_summary = tmp_path / "runs" / "pytest_v05_matrix" / "matrix_summary.json"
    config = RuntimeRegressionConfig(
        regression_id="pytest_v05_matrix_regression",
        artifact_dir=str(tmp_path / "runs" / "pytest_v05_regression"),
        targets=(
            RegressionTargetConfig(
                id="v05_matrix_config_load",
                kind=RegressionTargetKind.MATRIX_CONFIG_LOAD,
                matrix="configs/v05_run_matrix.yaml",
                evaluation="configs/v05_evaluation.yaml",
            ),
            RegressionTargetConfig(
                id="v05_matrix_run",
                kind=RegressionTargetKind.MATRIX_RUN,
                matrix="configs/v05_run_matrix.yaml",
                evaluation="configs/v05_evaluation.yaml",
            ),
            RegressionTargetConfig(
                id="v05_aggregate_summary",
                kind=RegressionTargetKind.AGGREGATE_SUMMARY,
                summary="runs/v05_run_matrix/matrix_summary.json",
            ),
        ),
    )

    summary = run_runtime_regression(config=config)

    assert summary.target_count == 3
    assert summary.passed_count == 3
    assert summary.failed_count == 0
    assert summary.provider_called_any is False
    aggregate_target = next(
        target
        for target in summary.targets
        if target.kind == RegressionTargetKind.AGGREGATE_SUMMARY
    )
    assert aggregate_target.metric_count == 11
    matrix = load_matrix_summary(Path("runs/v05_run_matrix/matrix_summary.json"))
    assert matrix.overall.run_count == 3
    assert matrix.overall.seed_count == 3
    assert matrix.overall.family_count == 2
    assert len(matrix.families) == 2
    assert matrix.overall.provider_called_any is False
    assert matrix.overall.raw_text_saved_any is False
    assert matrix.overall.unexpected_bad_case_count == 0
    assert matrix_summary.name == "matrix_summary.json"


def test_v06_matrix_inspect_targets_pass_without_provider(tmp_path: Path) -> None:
    matrix = run_matrix(
        config=RunMatrixConfig(
            matrix_id="pytest_v06_inspect_regression_matrix",
            artifact_dir=str(tmp_path / "runs" / "pytest_v06_matrix"),
            runs=(
                RunMatrixEntry(
                    run_id="pytest_v06_mistgate_seed7",
                    seed="seeds/mistgate_v01",
                    seed_family="mistgate",
                    config="configs/standard_experiment_deterministic_regression.yaml",
                    include_comparison=True,
                ),
            ),
        ),
        evaluation_config=AggregateEvaluationConfig(
            evaluation_id="pytest_v06_inspect_eval",
            expected_failure_variant_ids=("shared_context_agents", "rule_only_world_simulation"),
        ),
    )
    config = RuntimeRegressionConfig(
        regression_id="pytest_v06_inspect_regression",
        artifact_dir=str(tmp_path / "runs" / "pytest_v06_regression"),
        targets=(
            RegressionTargetConfig(
                id="v06_matrix_inspect_load",
                kind=RegressionTargetKind.MATRIX_INSPECT_LOAD,
                summary=matrix.matrix_summary_path,
            ),
            RegressionTargetConfig(
                id="v06_matrix_inspect_review_protocol",
                kind=RegressionTargetKind.MATRIX_INSPECT_REVIEW_PROTOCOL,
                summary=matrix.matrix_summary_path,
            ),
            RegressionTargetConfig(
                id="v06_matrix_inspect_safety",
                kind=RegressionTargetKind.MATRIX_INSPECT_SAFETY,
                summary=matrix.matrix_summary_path,
            ),
        ),
    )

    summary = run_runtime_regression(config=config)

    assert summary.target_count == 3
    assert summary.passed_count == 3
    assert summary.failed_count == 0
    assert summary.provider_called_any is False
    assert all(target.failed_metric_count == 0 for target in summary.targets)


def test_v07_scenario_contract_targets_pass_without_provider(tmp_path: Path) -> None:
    config = RuntimeRegressionConfig(
        regression_id="pytest_v07_contract_regression",
        artifact_dir=str(tmp_path / "runs" / "pytest_v07_regression"),
        targets=(
            RegressionTargetConfig(
                id="v07_scenario_contract_load",
                kind=RegressionTargetKind.SCENARIO_CONTRACT_LOAD,
            ),
            RegressionTargetConfig(
                id="v07_scenario_contract_fixture_coverage",
                kind=RegressionTargetKind.SCENARIO_CONTRACT_FIXTURE_COVERAGE,
            ),
            RegressionTargetConfig(
                id="v07_scenario_contract_verifier_rule_pack_parity",
                kind=RegressionTargetKind.SCENARIO_CONTRACT_VERIFIER_RULE_PACK_PARITY,
            ),
            RegressionTargetConfig(
                id="v07_scenario_contract_matrix_compatibility",
                kind=RegressionTargetKind.SCENARIO_CONTRACT_MATRIX_COMPATIBILITY,
                summary="runs/v05_run_matrix/matrix_summary.json",
            ),
            RegressionTargetConfig(
                id="v07_scenario_contract_no_provider",
                kind=RegressionTargetKind.SCENARIO_CONTRACT_NO_PROVIDER,
            ),
        ),
    )

    summary = run_runtime_regression(config=config)

    assert summary.target_count == 5
    assert summary.passed_count == 5
    assert summary.failed_count == 0
    assert summary.provider_called_any is False
    assert all(target.failed_metric_count == 0 for target in summary.targets)


def test_v08_provider_proposal_targets_pass_without_real_provider(tmp_path: Path) -> None:
    config = RuntimeRegressionConfig(
        regression_id="pytest_v08_provider_proposal_regression",
        artifact_dir=str(tmp_path / "runs" / "pytest_v08_regression"),
        targets=(
            RegressionTargetConfig(
                id="v08_provider_proposal_missing_settings_stops_before_chain",
                kind=RegressionTargetKind.PROVIDER_PROPOSAL_MISSING_SETTINGS_STOPS_BEFORE_CHAIN,
            ),
            RegressionTargetConfig(
                id="v08_provider_proposal_schema_failure_no_event_candidate",
                kind=RegressionTargetKind.PROVIDER_PROPOSAL_SCHEMA_FAILURE_NO_EVENT_CANDIDATE,
            ),
            RegressionTargetConfig(
                id="v08_provider_proposal_governance_chain",
                kind=RegressionTargetKind.PROVIDER_PROPOSAL_GOVERNANCE_CHAIN,
            ),
            RegressionTargetConfig(
                id="v08_provider_proposal_no_raw_text_artifacts",
                kind=RegressionTargetKind.PROVIDER_PROPOSAL_NO_RAW_TEXT_ARTIFACTS,
            ),
            RegressionTargetConfig(
                id="v08_deterministic_regressions_still_no_provider",
                kind=RegressionTargetKind.DETERMINISTIC_REGRESSIONS_STILL_NO_PROVIDER,
            ),
        ),
    )

    summary = run_runtime_regression(config=config)

    assert summary.target_count == 5
    assert summary.passed_count == 5
    assert summary.failed_count == 0
    assert summary.provider_called_any is False
    assert all(target.failed_metric_count == 0 for target in summary.targets)


def test_runtime_regression_failed_target_reports_reason(tmp_path: Path) -> None:
    config = RuntimeRegressionConfig(
        regression_id="pytest_bad_runtime_regression",
        artifact_dir=str(tmp_path / "runs" / "pytest_bad_runtime_regression"),
        targets=(
            RegressionTargetConfig(
                id="bad_preview",
                kind=RegressionTargetKind.RUN_WORLD_PREVIEW,
                seed="seeds/mistgate_v01",
                config="configs/missing_run_config.yaml",
            ),
        ),
    )

    summary = run_runtime_regression(config=config)

    assert summary.target_count == 1
    assert summary.passed_count == 0
    assert summary.failed_count == 1
    assert summary.provider_called_any is False
    assert summary.targets[0].status == "failed"
    assert summary.targets[0].failure_reason is not None
    assert Path(summary.summary_path).exists()
