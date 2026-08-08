from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from aethelis.evaluation import evaluate_formal_run
from aethelis.experiments.runner import load_formal_experiment_config, run_formal_experiment
from aethelis.experiments.variants import run_experiment_comparison
from aethelis.schemas.common import AethelisModel, Identifier


class RunMatrixConfigurationError(ValueError):
    """Safe run-matrix configuration error."""


class RunMatrixEntry(AethelisModel):
    run_id: Identifier
    seed: str = Field(min_length=1)
    config: str = Field(min_length=1)
    seed_family: Identifier | None = None
    random_seed: int | None = Field(default=None, ge=0)
    step_count: int | None = Field(default=None, ge=20, le=50)
    include_comparison: bool = True


class RunMatrixConfig(AethelisModel):
    matrix_id: Identifier
    artifact_dir: str = "runs/v04_run_matrix"
    runs: tuple[RunMatrixEntry, ...] = Field(min_length=1)


class AggregateEvaluationConfig(AethelisModel):
    evaluation_id: Identifier
    allow_expected_ablation_failures: bool = True
    expected_failure_variant_ids: tuple[Identifier, ...] = ()
    fail_on_provider_called: bool = True
    fail_on_raw_text_saved: bool = True
    max_failed_base_runs: int = Field(default=0, ge=0)
    max_unexpected_variant_failures: int = Field(default=0, ge=0)
    max_unexpected_bad_cases: int = Field(default=0, ge=0)


class MatrixRunSummary(AethelisModel):
    run_id: Identifier
    seed: str
    seed_family: Identifier
    random_seed: int
    step_count: int
    provider_called: bool
    raw_text_saved: bool
    metric_count: int
    failed_metric_count: int
    bad_case_count: int
    status: str
    run_dir: str


class MatrixVariantSummary(AethelisModel):
    run_id: Identifier
    seed: str
    seed_family: Identifier
    variant_id: Identifier
    variant_type: str
    failed_metric_count: int
    bad_case_count: int
    expected_failure: bool
    status: str
    artifact_path: str


class MatrixSeedSummary(AethelisModel):
    seed: str
    run_count: int
    passed_run_count: int
    failed_run_count: int
    variant_count: int
    unexpected_variant_failure_count: int


class MatrixFamilySummary(AethelisModel):
    seed_family: Identifier
    run_count: int
    seed_count: int
    passed_run_count: int
    failed_run_count: int
    variant_count: int
    unexpected_variant_failure_count: int


class MatrixBadCaseRecord(AethelisModel):
    case_id: Identifier
    seed: str
    run_id: Identifier
    variant_id: Identifier
    variant_type: str
    metric_name: Identifier
    severity: str
    governance_category: Identifier
    expected_failure: bool
    status: str
    trace_reference: Identifier
    artifact_path: str


class MatrixThresholdSummary(AethelisModel):
    allow_expected_ablation_failures: bool
    expected_failure_variant_ids: tuple[Identifier, ...]
    fail_on_provider_called: bool
    fail_on_raw_text_saved: bool
    max_failed_base_runs: int
    max_unexpected_variant_failures: int
    max_unexpected_bad_cases: int


class MatrixOverallSummary(AethelisModel):
    matrix_id: Identifier
    run_count: int
    passed_run_count: int
    failed_run_count: int
    seed_count: int
    family_count: int = 0
    variant_count: int
    expected_variant_failure_count: int
    unexpected_variant_failure_count: int
    bad_case_count: int
    expected_bad_case_count: int
    unexpected_bad_case_count: int
    bad_case_taxonomy: dict[Identifier, int]
    thresholds: MatrixThresholdSummary
    provider_called_any: bool
    raw_text_saved_any: bool
    artifact_safety_passed: bool
    passed: bool


class RunMatrixSummary(AethelisModel):
    matrix_id: Identifier
    evaluation_id: Identifier
    artifact_dir: str
    matrix_summary_path: str
    aggregate_summary_path: str
    runs: tuple[MatrixRunSummary, ...]
    seeds: tuple[MatrixSeedSummary, ...]
    families: tuple[MatrixFamilySummary, ...] = ()
    variants: tuple[MatrixVariantSummary, ...]
    bad_cases: tuple[MatrixBadCaseRecord, ...]
    overall: MatrixOverallSummary

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def load_run_matrix_config(path: Path) -> RunMatrixConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunMatrixConfigurationError(
            f"{exc.__class__.__name__}: run matrix config could not be read"
        ) from None
    if payload is None:
        raise RunMatrixConfigurationError("Run matrix config is empty.")
    try:
        return RunMatrixConfig.model_validate(payload)
    except ValidationError as exc:
        raise RunMatrixConfigurationError("ValidationError: run matrix config invalid") from exc


def load_aggregate_evaluation_config(path: Path) -> AggregateEvaluationConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunMatrixConfigurationError(
            f"{exc.__class__.__name__}: aggregate evaluation config could not be read"
        ) from None
    if payload is None:
        raise RunMatrixConfigurationError("Aggregate evaluation config is empty.")
    try:
        return AggregateEvaluationConfig.model_validate(payload)
    except ValidationError as exc:
        raise RunMatrixConfigurationError(
            "ValidationError: aggregate evaluation config invalid"
        ) from exc


def run_matrix(
    *,
    config: RunMatrixConfig,
    evaluation_config: AggregateEvaluationConfig,
) -> RunMatrixSummary:
    artifact_dir = Path(config.artifact_dir).resolve()
    formal_runs_dir = artifact_dir / "formal_runs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    formal_runs_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[MatrixRunSummary] = []
    variant_summaries: list[MatrixVariantSummary] = []
    for entry in config.runs:
        formal_config = _formal_config(entry)
        run_result = run_formal_experiment(
            seed_path=Path(entry.seed),
            config=formal_config,
            runs_dir=formal_runs_dir,
        )
        evaluation = evaluate_formal_run(run_result.artifacts.run_dir)
        run_summaries.append(
            MatrixRunSummary(
                run_id=entry.run_id,
                seed=entry.seed,
                seed_family=_seed_family(entry),
                random_seed=formal_config.random_seed,
                step_count=run_result.plan.step_count,
                provider_called=run_result.provider_called or evaluation.provider_called,
                raw_text_saved=evaluation.raw_text_saved,
                metric_count=evaluation.metric_count,
                failed_metric_count=evaluation.failed_metric_count,
                bad_case_count=evaluation.bad_case_count,
                status=(
                    "passed"
                    if evaluation.failed_metric_count == 0
                    and evaluation.bad_case_count == 0
                    and not run_result.provider_called
                    and not evaluation.provider_called
                    and not evaluation.raw_text_saved
                    else "failed"
                ),
                run_dir=str(run_result.artifacts.run_dir),
            )
        )
        if entry.include_comparison:
            comparison = run_experiment_comparison(
                seed_path=Path(entry.seed),
                config=formal_config,
                runs_dir=formal_runs_dir,
            )
            variant_summaries.extend(
                _variant_summaries(
                    entry=entry,
                    variants=comparison.variants,
                    allow_expected_ablation_failures=(
                        evaluation_config.allow_expected_ablation_failures
                    ),
                    expected_failure_variant_ids=evaluation_config.expected_failure_variant_ids,
                )
            )

    seed_summaries = _seed_summaries(run_summaries, variant_summaries)
    family_summaries = _family_summaries(run_summaries, variant_summaries)
    bad_cases = _matrix_bad_cases(variant_summaries)
    overall = _overall_summary(
        config=config,
        evaluation_config=evaluation_config,
        artifact_dir=artifact_dir,
        runs=run_summaries,
        variants=variant_summaries,
        bad_cases=bad_cases,
    )
    matrix_path = artifact_dir / "matrix_summary.json"
    aggregate_path = artifact_dir / "aggregate_summary.json"
    summary = RunMatrixSummary(
        matrix_id=config.matrix_id,
        evaluation_id=evaluation_config.evaluation_id,
        artifact_dir=str(artifact_dir),
        matrix_summary_path=str(matrix_path),
        aggregate_summary_path=str(aggregate_path),
        runs=tuple(run_summaries),
        seeds=tuple(seed_summaries),
        families=tuple(family_summaries),
        variants=tuple(variant_summaries),
        bad_cases=bad_cases,
        overall=overall,
    )
    payload = summary.safe_dict()
    matrix_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate_path.write_text(
        json.dumps(
            {
                "matrix_id": summary.matrix_id,
                "evaluation_id": summary.evaluation_id,
                "seeds": [seed.model_dump(mode="json") for seed in summary.seeds],
                "families": [family.model_dump(mode="json") for family in summary.families],
                "variants": [variant.model_dump(mode="json") for variant in summary.variants],
                "bad_cases": [case.model_dump(mode="json") for case in summary.bad_cases],
                "overall": summary.overall.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def load_matrix_summary(path: Path) -> RunMatrixSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunMatrixSummary.model_validate(payload)


def inspect_matrix_summary(path: Path) -> dict[str, object]:
    return compact_matrix_review(load_matrix_summary(path))


def compact_matrix_review(summary: RunMatrixSummary) -> dict[str, object]:
    variant_status_by_type: dict[str, dict[str, int]] = {}
    for (variant_type, status), count in sorted(
        Counter((variant.variant_type, variant.status) for variant in summary.variants).items()
    ):
        variant_status_by_type.setdefault(variant_type, {})[status] = count

    proposed_runtime_failure_count = sum(
        1
        for variant in summary.variants
        if variant.variant_type == "proposed" and variant.status != "passed"
    )
    review_flags = {
        "passed": summary.overall.passed,
        "provider_called_any": summary.overall.provider_called_any,
        "raw_text_saved_any": summary.overall.raw_text_saved_any,
        "artifact_safety_passed": summary.overall.artifact_safety_passed,
        "proposed_runtime_failure_count": proposed_runtime_failure_count,
        "unexpected_variant_failure_count": summary.overall.unexpected_variant_failure_count,
        "unexpected_bad_case_count": summary.overall.unexpected_bad_case_count,
    }
    return {
        "matrix_id": summary.matrix_id,
        "evaluation_id": summary.evaluation_id,
        "passed": summary.overall.passed,
        "run_count": summary.overall.run_count,
        "seed_count": summary.overall.seed_count,
        "family_count": summary.overall.family_count,
        "variant_count": summary.overall.variant_count,
        "bad_case_count": summary.overall.bad_case_count,
        "provider_called_any": summary.overall.provider_called_any,
        "raw_text_saved_any": summary.overall.raw_text_saved_any,
        "artifact_safety_passed": summary.overall.artifact_safety_passed,
        "expected_variant_failure_count": summary.overall.expected_variant_failure_count,
        "unexpected_variant_failure_count": summary.overall.unexpected_variant_failure_count,
        "expected_bad_case_count": summary.overall.expected_bad_case_count,
        "unexpected_bad_case_count": summary.overall.unexpected_bad_case_count,
        "proposed_runtime_failure_count": proposed_runtime_failure_count,
        "families": [family.model_dump(mode="json") for family in summary.families],
        "variant_status_by_type": variant_status_by_type,
        "bad_case_taxonomy": summary.overall.bad_case_taxonomy,
        "thresholds": summary.overall.thresholds.model_dump(mode="json"),
        "review_flags": review_flags,
    }


def render_matrix_review_markdown(review: dict[str, object]) -> str:
    flags = review["review_flags"]
    thresholds = review["thresholds"]
    lines = [
        f"# Matrix Review: {review['matrix_id']}",
        "",
        "## Overall",
        f"- evaluation_id: {review['evaluation_id']}",
        f"- passed: {str(review['passed']).lower()}",
        f"- runs/seeds/families: "
        f"{review['run_count']}/{review['seed_count']}/{review['family_count']}",
        f"- variants/bad_cases: {review['variant_count']}/{review['bad_case_count']}",
        f"- provider_called_any: {str(review['provider_called_any']).lower()}",
        f"- raw_text_saved_any: {str(review['raw_text_saved_any']).lower()}",
        f"- artifact_safety_passed: {str(review['artifact_safety_passed']).lower()}",
        "",
        "## Failure Review",
        f"- expected_variant_failure_count: {review['expected_variant_failure_count']}",
        f"- unexpected_variant_failure_count: {review['unexpected_variant_failure_count']}",
        f"- expected_bad_case_count: {review['expected_bad_case_count']}",
        f"- unexpected_bad_case_count: {review['unexpected_bad_case_count']}",
        f"- proposed_runtime_failure_count: {review['proposed_runtime_failure_count']}",
        "",
        "## Variant Status By Type",
    ]
    for variant_type, statuses in review["variant_status_by_type"].items():
        lines.append(
            f"- {variant_type}: "
            + ", ".join(f"{status}={count}" for status, count in statuses.items())
        )
    lines.extend(["", "## Bad-Case Taxonomy"])
    for category, count in review["bad_case_taxonomy"].items():
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Families"])
    for family in review["families"]:
        lines.append(
            f"- {family['seed_family']}: runs={family['run_count']}, "
            f"seeds={family['seed_count']}, variants={family['variant_count']}, "
            f"unexpected_variant_failures={family['unexpected_variant_failure_count']}"
        )
    lines.extend(
        [
            "",
            "## Expected-Failure Protocol",
            f"- allow_expected_ablation_failures: "
            f"{str(thresholds['allow_expected_ablation_failures']).lower()}",
            f"- expected_failure_variant_ids: "
            f"{', '.join(thresholds['expected_failure_variant_ids']) or 'none'}",
            "- proposed runtime failures remain unexpected.",
            "- baseline and ablation expected failures remain review records, "
            "not success evidence.",
            "",
            "## Review Flags",
        ]
    )
    for name, value in flags.items():
        lines.append(f"- {name}: {str(value).lower() if isinstance(value, bool) else value}")
    return "\n".join(lines) + "\n"


def _formal_config(entry: RunMatrixEntry):
    base = load_formal_experiment_config(Path(entry.config))
    updates: dict[str, object] = {
        "run_id": entry.run_id,
        "seed_path": entry.seed,
    }
    if entry.random_seed is not None:
        updates["random_seed"] = entry.random_seed
    if entry.step_count is not None:
        updates["step_count"] = entry.step_count
    return base.model_copy(update=updates)


def _variant_summaries(
    *,
    entry: RunMatrixEntry,
    variants,
    allow_expected_ablation_failures: bool,
    expected_failure_variant_ids: tuple[str, ...],
) -> tuple[MatrixVariantSummary, ...]:
    summaries: list[MatrixVariantSummary] = []
    for variant in variants:
        failed_metric_count = int(variant.metric_summary.get("failed_metric_count", 0))
        variant_type = variant.variant_type.value
        expected_failure = _is_expected_failure(
            variant_id=variant.variant_id,
            variant_type=variant_type,
            failed_metric_count=failed_metric_count,
            allow_expected_ablation_failures=allow_expected_ablation_failures,
            expected_failure_variant_ids=expected_failure_variant_ids,
        )
        status = (
            "expected_failure"
            if expected_failure
            else "failed"
            if failed_metric_count
            else "passed"
        )
        summaries.append(
            MatrixVariantSummary(
                run_id=entry.run_id,
                seed=entry.seed,
                seed_family=_seed_family(entry),
                variant_id=variant.variant_id,
                variant_type=variant_type,
                failed_metric_count=failed_metric_count,
                bad_case_count=variant.bad_case_count,
                expected_failure=expected_failure,
                status=status,
                artifact_path=variant.artifact_path,
            )
        )
    return tuple(summaries)


def _is_expected_failure(
    *,
    variant_id: str,
    variant_type: str,
    failed_metric_count: int,
    allow_expected_ablation_failures: bool,
    expected_failure_variant_ids: tuple[str, ...],
) -> bool:
    if failed_metric_count <= 0 or variant_type == "proposed":
        return False
    if variant_type == "baseline":
        return variant_id in expected_failure_variant_ids
    if variant_type == "ablation":
        return allow_expected_ablation_failures
    return False


def _matrix_bad_cases(
    variants: list[MatrixVariantSummary],
) -> tuple[MatrixBadCaseRecord, ...]:
    records: list[MatrixBadCaseRecord] = []
    for variant in variants:
        for case in _read_variant_bad_cases(Path(variant.artifact_path) / "bad_cases.json"):
            metric_name = str(case.get("failure_type", "unknown_metric"))
            records.append(
                MatrixBadCaseRecord(
                    case_id=f"matrix_bad_case_{len(records) + 1:03d}",
                    seed=variant.seed,
                    run_id=variant.run_id,
                    variant_id=variant.variant_id,
                    variant_type=variant.variant_type,
                    metric_name=metric_name,
                    severity=str(case.get("severity", "low")),
                    governance_category=_governance_category(metric_name),
                    expected_failure=variant.expected_failure,
                    status=variant.status,
                    trace_reference=str(case.get("trace_reference", variant.run_id)),
                    artifact_path=variant.artifact_path,
                )
            )
    return tuple(records)


def _read_variant_bad_cases(path: Path) -> tuple[dict[str, object], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("bad_cases")
    if not isinstance(cases, list):
        return ()
    return tuple(case for case in cases if isinstance(case, dict))


def _governance_category(metric_name: str) -> str:
    return {
        "canon_safety": "canon_governance",
        "player_input_governance_safety": "player_input_governance",
        "agent_knowledge_boundary_safety": "agent_boundary",
        "belief_canon_separation": "belief_canon_boundary",
        "event_validity": "verification_governance",
        "verification_decision_validity": "verification_governance",
    }.get(metric_name, "evaluation_quality")


def _seed_summaries(
    runs: list[MatrixRunSummary],
    variants: list[MatrixVariantSummary],
) -> tuple[MatrixSeedSummary, ...]:
    by_seed_runs: dict[str, list[MatrixRunSummary]] = defaultdict(list)
    by_seed_variants: dict[str, list[MatrixVariantSummary]] = defaultdict(list)
    for run in runs:
        by_seed_runs[run.seed].append(run)
    for variant in variants:
        by_seed_variants[variant.seed].append(variant)
    return tuple(
        MatrixSeedSummary(
            seed=seed,
            run_count=len(seed_runs),
            passed_run_count=sum(1 for run in seed_runs if run.status == "passed"),
            failed_run_count=sum(1 for run in seed_runs if run.status != "passed"),
            variant_count=len(by_seed_variants.get(seed, [])),
            unexpected_variant_failure_count=sum(
                1
                for variant in by_seed_variants.get(seed, [])
                if variant.status == "failed" and not variant.expected_failure
            ),
        )
        for seed, seed_runs in sorted(by_seed_runs.items())
    )


def _family_summaries(
    runs: list[MatrixRunSummary],
    variants: list[MatrixVariantSummary],
) -> tuple[MatrixFamilySummary, ...]:
    by_family_runs: dict[str, list[MatrixRunSummary]] = defaultdict(list)
    by_family_variants: dict[str, list[MatrixVariantSummary]] = defaultdict(list)
    for run in runs:
        by_family_runs[run.seed_family].append(run)
    for variant in variants:
        by_family_variants[variant.seed_family].append(variant)
    return tuple(
        MatrixFamilySummary(
            seed_family=family,
            run_count=len(family_runs),
            seed_count=len({run.seed for run in family_runs}),
            passed_run_count=sum(1 for run in family_runs if run.status == "passed"),
            failed_run_count=sum(1 for run in family_runs if run.status != "passed"),
            variant_count=len(by_family_variants.get(family, [])),
            unexpected_variant_failure_count=sum(
                1
                for variant in by_family_variants.get(family, [])
                if variant.status == "failed" and not variant.expected_failure
            ),
        )
        for family, family_runs in sorted(by_family_runs.items())
    )


def _overall_summary(
    *,
    config: RunMatrixConfig,
    evaluation_config: AggregateEvaluationConfig,
    artifact_dir: Path,
    runs: list[MatrixRunSummary],
    variants: list[MatrixVariantSummary],
    bad_cases: tuple[MatrixBadCaseRecord, ...],
) -> MatrixOverallSummary:
    failed_run_count = sum(1 for run in runs if run.status != "passed")
    unexpected_variant_failures = sum(
        1 for variant in variants if variant.status == "failed" and not variant.expected_failure
    )
    unexpected_bad_cases = sum(1 for case in bad_cases if not case.expected_failure)
    provider_called_any = any(run.provider_called for run in runs)
    raw_text_saved_any = any(run.raw_text_saved for run in runs)
    artifact_safety_passed = _artifact_safety_passed(config.artifact_dir, artifact_dir)
    passed = (
        failed_run_count <= evaluation_config.max_failed_base_runs
        and unexpected_variant_failures <= evaluation_config.max_unexpected_variant_failures
        and unexpected_bad_cases <= evaluation_config.max_unexpected_bad_cases
        and artifact_safety_passed
        and (not evaluation_config.fail_on_provider_called or not provider_called_any)
        and (not evaluation_config.fail_on_raw_text_saved or not raw_text_saved_any)
    )
    return MatrixOverallSummary(
        matrix_id=config.matrix_id,
        run_count=len(runs),
        passed_run_count=sum(1 for run in runs if run.status == "passed"),
        failed_run_count=failed_run_count,
        seed_count=len({run.seed for run in runs}),
        family_count=len({run.seed_family for run in runs}),
        variant_count=len(variants),
        expected_variant_failure_count=sum(1 for variant in variants if variant.expected_failure),
        unexpected_variant_failure_count=unexpected_variant_failures,
        bad_case_count=len(bad_cases),
        expected_bad_case_count=sum(1 for case in bad_cases if case.expected_failure),
        unexpected_bad_case_count=unexpected_bad_cases,
        bad_case_taxonomy=_bad_case_taxonomy(bad_cases),
        thresholds=_thresholds(evaluation_config),
        provider_called_any=provider_called_any,
        raw_text_saved_any=raw_text_saved_any,
        artifact_safety_passed=artifact_safety_passed,
        passed=passed,
    )


def _bad_case_taxonomy(bad_cases: tuple[MatrixBadCaseRecord, ...]) -> dict[str, int]:
    taxonomy: dict[str, int] = defaultdict(int)
    for case in bad_cases:
        taxonomy[case.governance_category] += 1
    return dict(sorted(taxonomy.items()))


def _thresholds(evaluation_config: AggregateEvaluationConfig) -> MatrixThresholdSummary:
    return MatrixThresholdSummary(
        allow_expected_ablation_failures=evaluation_config.allow_expected_ablation_failures,
        expected_failure_variant_ids=evaluation_config.expected_failure_variant_ids,
        fail_on_provider_called=evaluation_config.fail_on_provider_called,
        fail_on_raw_text_saved=evaluation_config.fail_on_raw_text_saved,
        max_failed_base_runs=evaluation_config.max_failed_base_runs,
        max_unexpected_variant_failures=evaluation_config.max_unexpected_variant_failures,
        max_unexpected_bad_cases=evaluation_config.max_unexpected_bad_cases,
    )


def _artifact_safety_passed(configured_artifact_dir: str, artifact_dir: Path) -> bool:
    normalized = configured_artifact_dir.replace("\\", "/").strip("/")
    return (
        (normalized.startswith("runs/") or "runs" in artifact_dir.parts)
        and "reports" not in artifact_dir.parts
        and artifact_dir.name != ""
    )


def _seed_family(entry: RunMatrixEntry) -> str:
    return entry.seed_family or entry.seed.replace("\\", "/").rstrip("/").split("/")[-1]
