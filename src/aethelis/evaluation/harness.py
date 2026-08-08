from __future__ import annotations

import json
from pathlib import Path

from aethelis.evaluation.bad_cases import BadCaseSummary, collect_bad_cases
from aethelis.evaluation.metrics import MetricsSummary, calculate_metrics
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.trace.formal import load_formal_trace


class FormalEvaluationError(ValueError):
    """Safe formal evaluation error."""


class FormalEvaluationSummary(AethelisModel):
    run_id: Identifier
    formal_experiment_result: bool
    metric_count: int
    failed_metric_count: int
    bad_case_count: int
    metrics_summary_path: str
    evaluation_summary_path: str
    bad_cases_path: str
    provider_called: bool = False
    raw_text_saved: bool = False

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def evaluate_formal_run(run_dir: Path) -> FormalEvaluationSummary:
    run_dir = run_dir.resolve()
    trace_path = run_dir / "trace.json"
    run_summary_path = run_dir / "run_summary.json"
    config_snapshot_path = run_dir / "config_snapshot.json"
    _require_files(trace_path, run_summary_path, config_snapshot_path)

    trace = load_formal_trace(trace_path)
    run_summary = _read_json(run_summary_path)
    config_snapshot = _read_json(config_snapshot_path)
    if trace.formal_experiment_result is not True:
        raise FormalEvaluationError("formal evaluation requires formal_experiment_result=true")
    if run_summary.get("formal_experiment_result") is not True:
        raise FormalEvaluationError("run_summary is not a formal experiment result")
    if config_snapshot.get("formal_experiment_result") is not True:
        raise FormalEvaluationError("config_snapshot is not a formal experiment result")

    metrics = calculate_metrics(
        trace,
        run_summary=run_summary,
        config_snapshot=config_snapshot,
    )
    bad_cases = collect_bad_cases(trace, metrics=metrics.metrics)
    metrics_path = run_dir / "metrics_summary.json"
    evaluation_path = run_dir / "evaluation_summary.json"
    bad_cases_path = run_dir / "bad_cases.json"

    metrics_path.write_text(
        json.dumps(metrics.safe_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    bad_cases_path.write_text(
        json.dumps(bad_cases.safe_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = _summary(
        trace=trace,
        metrics=metrics,
        bad_cases=bad_cases,
        metrics_path=metrics_path,
        evaluation_path=evaluation_path,
        bad_cases_path=bad_cases_path,
    )
    evaluation_path.write_text(
        json.dumps(summary.safe_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _summary(
    *,
    trace,
    metrics: MetricsSummary,
    bad_cases: BadCaseSummary,
    metrics_path: Path,
    evaluation_path: Path,
    bad_cases_path: Path,
) -> FormalEvaluationSummary:
    return FormalEvaluationSummary(
        run_id=metrics.run_id,
        formal_experiment_result=trace.formal_experiment_result,
        metric_count=metrics.metric_count,
        failed_metric_count=metrics.failed_count,
        bad_case_count=bad_cases.bad_case_count,
        metrics_summary_path=str(metrics_path),
        evaluation_summary_path=str(evaluation_path),
        bad_cases_path=str(bad_cases_path),
        provider_called=bool(trace.metadata.get("provider_called", False)),
        raw_text_saved=bool(trace.metadata.get("raw_text_saved", False)),
    )


def _require_files(*paths: Path) -> None:
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FormalEvaluationError(f"formal run artifacts missing: {', '.join(missing)}")


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalEvaluationError(f"{exc.__class__.__name__}: {path.name} invalid") from None
    if not isinstance(payload, dict):
        raise FormalEvaluationError(f"{path.name} must contain a JSON object")
    return payload
