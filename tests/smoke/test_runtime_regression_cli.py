from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from aethelis.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
REGRESSION_CONFIG = ROOT / "configs" / "v02_runtime_regression.yaml"


def test_regression_run_cli_writes_summary_without_provider(monkeypatch) -> None:
    def fail_if_provider_called(*args, **kwargs):
        raise AssertionError("regression-run must not call providers")

    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fail_if_provider_called,
    )

    result = CliRunner().invoke(
        app,
        [
            "regression-run",
            "--config",
            str(REGRESSION_CONFIG),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["target_count"] == 7
    assert payload["passed_count"] == 7
    assert payload["failed_count"] == 0
    assert payload["provider_called_any"] is False
    by_id = {target["target_id"]: target for target in payload["targets"]}
    assert by_id["standard_preview"]["formal_experiment_result"] is False
    assert by_id["standard_apply"]["state_diff_applied_count"] == 3
    assert by_id["standard_formal"]["formal_experiment_result"] is True
    assert by_id["standard_evaluate"]["failed_metric_count"] == 0
    assert by_id["standard_evaluate"]["bad_case_count"] == 0
    assert by_id["standard_compare"]["variant_count"] == 11
    assert Path(payload["summary_path"]).exists()
