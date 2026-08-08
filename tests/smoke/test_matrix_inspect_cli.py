from pathlib import Path

from typer.testing import CliRunner

from aethelis.cli.app import app
from aethelis.experiments.matrix import (
    AggregateEvaluationConfig,
    RunMatrixConfig,
    RunMatrixEntry,
    run_matrix,
)

runner = CliRunner()


def test_matrix_inspect_cli_prints_json_and_markdown(tmp_path: Path) -> None:
    summary = run_matrix(
        config=RunMatrixConfig(
            matrix_id="pytest_cli_matrix_inspect",
            artifact_dir=str(tmp_path / "runs" / "pytest_cli_matrix_inspect"),
            runs=(
                RunMatrixEntry(
                    run_id="pytest_cli_mistgate_seed7",
                    seed="seeds/mistgate_v01",
                    seed_family="mistgate",
                    config="configs/standard_experiment_deterministic_regression.yaml",
                    include_comparison=True,
                ),
            ),
        ),
        evaluation_config=AggregateEvaluationConfig(
            evaluation_id="pytest_cli_matrix_inspect_eval",
            expected_failure_variant_ids=(
                "free_multi_agent_chat",
                "shared_context_agents",
                "rag_memory_agents",
                "rule_only_world_simulation",
            ),
        ),
    )

    json_result = runner.invoke(
        app,
        ["matrix-inspect", summary.matrix_summary_path, "--format", "json"],
    )
    markdown_result = runner.invoke(
        app,
        ["matrix-inspect", summary.matrix_summary_path, "--format", "markdown"],
    )

    assert json_result.exit_code == 0
    assert '"matrix_id": "pytest_cli_matrix_inspect"' in json_result.stdout
    assert '"provider_called_any": false' in json_result.stdout
    assert "raw_llm_text" not in json_result.stdout
    assert markdown_result.exit_code == 0
    assert "# Matrix Review: pytest_cli_matrix_inspect" in markdown_result.stdout
    assert "proposed runtime failures remain unexpected" in markdown_result.stdout
