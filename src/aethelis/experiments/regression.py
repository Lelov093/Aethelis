from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from aethelis.agents.action_proposal import ProposalSourceMode, ProviderProposalFailureCode
from aethelis.agents.activation import AgentActivationBuilder
from aethelis.agents.context import build_agent_context
from aethelis.evaluation import evaluate_formal_run, evaluate_formal_trace_preview
from aethelis.events.conversion import action_proposal_to_event_candidate
from aethelis.experiments.matrix import (
    inspect_matrix_summary,
    load_aggregate_evaluation_config,
    load_matrix_summary,
    load_run_matrix_config,
    render_matrix_review_markdown,
    run_matrix,
)
from aethelis.experiments.runner import load_formal_experiment_config, run_formal_experiment
from aethelis.experiments.variants import run_experiment_comparison
from aethelis.llm.base import LLMResult
from aethelis.llm.structured import generate_structured
from aethelis.providers import ProviderAttempt
from aethelis.runtime.scenario_matrix import (
    RUNTIME_SCENARIO_MATRIX,
    get_player_input_fixture_contract,
    get_proposal_fixture_contract,
    get_state_diff_contract,
    get_verifier_rule_pack,
)
from aethelis.runtime.single_step import build_committed_event, run_single_step
from aethelis.runtime.state_apply import ControlledStateDiffApplier
from aethelis.runtime.world_run import load_run_config, run_world
from aethelis.schemas.activation import ActivationStatus, AgentActivationConfig
from aethelis.schemas.common import AethelisModel, Identifier, RecordStatus
from aethelis.schemas.events import (
    ActionIntent,
    ActionProposal,
    PatchOperation,
    PatchTargetType,
    StateDiff,
    StatePatch,
    VerificationDecision,
)
from aethelis.schemas.run import RunStepPlanItem
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator
from aethelis.trace.formal import load_formal_trace, validate_formal_trace_file
from aethelis.utils.redaction import redact_text
from aethelis.verification.deterministic import DeterministicVerifier


class RegressionTargetKind(StrEnum):
    RUN_WORLD_PREVIEW = "run_world_preview"
    RUN_WORLD_APPLY = "run_world_apply"
    EXPERIMENT_RUN = "experiment_run"
    TRACE_VALIDATE = "trace_validate"
    EVALUATION_CHECK = "evaluation_check"
    EXPERIMENT_EVALUATE = "experiment_evaluate"
    EXPERIMENT_COMPARE = "experiment_compare"
    STATE_APPLY_CONTRACTS = "state_apply_contracts"
    SCHEDULER_EVOLUTION_CONTRACTS = "scheduler_evolution_contracts"
    MATRIX_CONFIG_LOAD = "matrix_config_load"
    MATRIX_RUN = "matrix_run"
    AGGREGATE_SUMMARY = "aggregate_summary"
    ARTIFACT_SAFETY = "artifact_safety"
    MATRIX_INSPECT_LOAD = "matrix_inspect_load"
    MATRIX_INSPECT_REVIEW_PROTOCOL = "matrix_inspect_review_protocol"
    MATRIX_INSPECT_SAFETY = "matrix_inspect_safety"
    SCENARIO_CONTRACT_LOAD = "scenario_contract_load"
    SCENARIO_CONTRACT_FIXTURE_COVERAGE = "scenario_contract_fixture_coverage"
    SCENARIO_CONTRACT_VERIFIER_RULE_PACK_PARITY = "scenario_contract_verifier_rule_pack_parity"
    SCENARIO_CONTRACT_MATRIX_COMPATIBILITY = "scenario_contract_matrix_compatibility"
    SCENARIO_CONTRACT_NO_PROVIDER = "scenario_contract_no_provider"
    PROVIDER_PROPOSAL_MISSING_SETTINGS_STOPS_BEFORE_CHAIN = (
        "provider_proposal_missing_settings_stops_before_chain"
    )
    PROVIDER_PROPOSAL_SCHEMA_FAILURE_NO_EVENT_CANDIDATE = (
        "provider_proposal_schema_failure_no_event_candidate"
    )
    PROVIDER_PROPOSAL_GOVERNANCE_CHAIN = "provider_proposal_governance_chain"
    PROVIDER_PROPOSAL_NO_RAW_TEXT_ARTIFACTS = "provider_proposal_no_raw_text_artifacts"
    DETERMINISTIC_REGRESSIONS_STILL_NO_PROVIDER = "deterministic_regressions_still_no_provider"


class RegressionTargetConfig(AethelisModel):
    id: Identifier
    kind: RegressionTargetKind
    seed: str | None = None
    config: str | None = None
    run: str | None = None
    trace: str | None = None
    matrix: str | None = None
    evaluation: str | None = None
    summary: str | None = None


class RuntimeRegressionConfig(AethelisModel):
    regression_id: Identifier
    artifact_dir: str = "runs/v02_runtime_regression"
    targets: tuple[RegressionTargetConfig, ...] = Field(min_length=1)


class RegressionTargetSummary(AethelisModel):
    target_id: Identifier
    kind: RegressionTargetKind
    status: str
    step_count: int | None = None
    formal_experiment_result: bool | None = None
    provider_called: bool | None = None
    state_diff_applied_count: int | None = None
    causal_node_count: int | None = None
    causal_edge_count: int | None = None
    pressure_update_count: int | None = None
    belief_update_count: int | None = None
    memory_signal_count: int | None = None
    relationship_signal_count: int | None = None
    metric_count: int | None = None
    failed_metric_count: int | None = None
    bad_case_count: int | None = None
    variant_count: int | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    failure_reason: str | None = None

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class RuntimeRegressionSummary(AethelisModel):
    regression_id: Identifier
    target_count: int
    passed_count: int
    failed_count: int
    provider_called_any: bool
    summary_path: str
    targets: tuple[RegressionTargetSummary, ...]

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class RuntimeRegressionConfigurationError(ValueError):
    """Safe regression config error."""


def load_runtime_regression_config(path: Path) -> RuntimeRegressionConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeRegressionConfigurationError(
            f"{exc.__class__.__name__}: regression config could not be read"
        ) from None
    if payload is None:
        raise RuntimeRegressionConfigurationError("Regression config is empty.")
    try:
        return RuntimeRegressionConfig.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeRegressionConfigurationError(
            "ValidationError: regression config schema invalid"
        ) from exc


def run_runtime_regression(
    *,
    config: RuntimeRegressionConfig,
) -> RuntimeRegressionSummary:
    target_summaries = tuple(_run_target(target) for target in config.targets)
    provider_called_any = any(target.provider_called is True for target in target_summaries)
    passed_count = sum(1 for target in target_summaries if target.status == "passed")
    artifact_dir = Path(config.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "regression_summary.json"
    summary = RuntimeRegressionSummary(
        regression_id=config.regression_id,
        target_count=len(target_summaries),
        passed_count=passed_count,
        failed_count=len(target_summaries) - passed_count,
        provider_called_any=provider_called_any,
        summary_path=str(summary_path),
        targets=target_summaries,
    )
    summary_path.write_text(
        json.dumps(summary.safe_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _run_target(target: RegressionTargetConfig) -> RegressionTargetSummary:
    try:
        summary = _execute_target(target)
    except Exception as exc:
        return RegressionTargetSummary(
            target_id=target.id,
            kind=target.kind,
            status="failed",
            failure_reason=redact_text(f"{exc.__class__.__name__}: {exc}"),
        )
    failure_reason = _failure_reason(summary)
    if failure_reason:
        return summary.model_copy(update={"status": "failed", "failure_reason": failure_reason})
    return summary.model_copy(update={"status": "passed"})


def _execute_target(target: RegressionTargetConfig) -> RegressionTargetSummary:
    if target.kind == RegressionTargetKind.RUN_WORLD_PREVIEW:
        result = run_world(
            seed_path=_required_path(target.seed),
            config=load_run_config(_required_path(target.config)),
        )
        return _world_target_summary(target, result)
    if target.kind == RegressionTargetKind.RUN_WORLD_APPLY:
        result = run_world(
            seed_path=_required_path(target.seed),
            config=load_run_config(_required_path(target.config)),
            apply=True,
        )
        return _world_target_summary(target, result)
    if target.kind == RegressionTargetKind.EXPERIMENT_RUN:
        result = run_formal_experiment(
            seed_path=_required_path(target.seed),
            config=load_formal_experiment_config(_required_path(target.config)),
        )
        return RegressionTargetSummary(
            target_id=target.id,
            kind=target.kind,
            status="pending",
            step_count=result.plan.step_count,
            formal_experiment_result=result.formal_experiment_result,
            provider_called=result.provider_called,
            artifact_paths=result.artifacts.safe_dict(),
        )
    if target.kind == RegressionTargetKind.TRACE_VALIDATE:
        report = validate_formal_trace_file(_required_path(target.trace))
        return RegressionTargetSummary(
            target_id=target.id,
            kind=target.kind,
            status="pending",
            formal_experiment_result=report.formal_experiment_result,
            step_count=report.record_count,
            artifact_paths={"trace_path": str(report.path)},
            failure_reason=None if report.success else report.error,
        )
    if target.kind == RegressionTargetKind.EVALUATION_CHECK:
        trace = load_formal_trace(_required_path(target.trace))
        result = evaluate_formal_trace_preview(trace)
        return RegressionTargetSummary(
            target_id=target.id,
            kind=target.kind,
            status="pending",
            step_count=result.case_count,
            formal_experiment_result=result.formal_experiment_result,
            metric_count=result.case_count,
            failed_metric_count=result.failed_count,
            artifact_paths={"trace_path": str(_required_path(target.trace))},
        )
    if target.kind == RegressionTargetKind.EXPERIMENT_EVALUATE:
        result = evaluate_formal_run(_required_path(target.run))
        return RegressionTargetSummary(
            target_id=target.id,
            kind=target.kind,
            status="pending",
            formal_experiment_result=result.formal_experiment_result,
            provider_called=result.provider_called,
            metric_count=result.metric_count,
            failed_metric_count=result.failed_metric_count,
            bad_case_count=result.bad_case_count,
            artifact_paths={
                "metrics_summary_path": result.metrics_summary_path,
                "evaluation_summary_path": result.evaluation_summary_path,
                "bad_cases_path": result.bad_cases_path,
            },
        )
    if target.kind == RegressionTargetKind.EXPERIMENT_COMPARE:
        result = run_experiment_comparison(
            seed_path=_required_path(target.seed),
            config=load_formal_experiment_config(_required_path(target.config)),
        )
        return RegressionTargetSummary(
            target_id=target.id,
            kind=target.kind,
            status="pending",
            formal_experiment_result=result.formal_experiment_result,
            provider_called=result.provider_called,
            variant_count=result.variant_count,
            artifact_paths=result.artifacts.safe_dict(),
        )
    if target.kind == RegressionTargetKind.STATE_APPLY_CONTRACTS:
        return _state_apply_contract_summary(target)
    if target.kind == RegressionTargetKind.SCHEDULER_EVOLUTION_CONTRACTS:
        return _scheduler_evolution_contract_summary(target)
    if target.kind == RegressionTargetKind.MATRIX_CONFIG_LOAD:
        return _matrix_config_load_summary(target)
    if target.kind == RegressionTargetKind.MATRIX_RUN:
        return _matrix_run_summary(target)
    if target.kind == RegressionTargetKind.AGGREGATE_SUMMARY:
        return _aggregate_summary_target(target)
    if target.kind == RegressionTargetKind.ARTIFACT_SAFETY:
        return _artifact_safety_target(target)
    if target.kind == RegressionTargetKind.MATRIX_INSPECT_LOAD:
        return _matrix_inspect_load_target(target)
    if target.kind == RegressionTargetKind.MATRIX_INSPECT_REVIEW_PROTOCOL:
        return _matrix_inspect_review_protocol_target(target)
    if target.kind == RegressionTargetKind.MATRIX_INSPECT_SAFETY:
        return _matrix_inspect_safety_target(target)
    if target.kind == RegressionTargetKind.SCENARIO_CONTRACT_LOAD:
        return _scenario_contract_load_target(target)
    if target.kind == RegressionTargetKind.SCENARIO_CONTRACT_FIXTURE_COVERAGE:
        return _scenario_contract_fixture_coverage_target(target)
    if target.kind == RegressionTargetKind.SCENARIO_CONTRACT_VERIFIER_RULE_PACK_PARITY:
        return _scenario_contract_verifier_rule_pack_parity_target(target)
    if target.kind == RegressionTargetKind.SCENARIO_CONTRACT_MATRIX_COMPATIBILITY:
        return _scenario_contract_matrix_compatibility_target(target)
    if target.kind == RegressionTargetKind.SCENARIO_CONTRACT_NO_PROVIDER:
        return _scenario_contract_no_provider_target(target)
    if target.kind == RegressionTargetKind.PROVIDER_PROPOSAL_MISSING_SETTINGS_STOPS_BEFORE_CHAIN:
        return _provider_proposal_missing_settings_stops_before_chain_target(target)
    if target.kind == RegressionTargetKind.PROVIDER_PROPOSAL_SCHEMA_FAILURE_NO_EVENT_CANDIDATE:
        return _provider_proposal_schema_failure_no_event_candidate_target(target)
    if target.kind == RegressionTargetKind.PROVIDER_PROPOSAL_GOVERNANCE_CHAIN:
        return _provider_proposal_governance_chain_target(target)
    if target.kind == RegressionTargetKind.PROVIDER_PROPOSAL_NO_RAW_TEXT_ARTIFACTS:
        return _provider_proposal_no_raw_text_artifacts_target(target)
    if target.kind == RegressionTargetKind.DETERMINISTIC_REGRESSIONS_STILL_NO_PROVIDER:
        return _deterministic_regressions_still_no_provider_target(target)
    raise ValueError(f"unsupported_regression_target_kind: {target.kind.value}")


def _state_apply_contract_summary(target: RegressionTargetConfig) -> RegressionTargetSummary:
    bundle, committed_event, verification = _committed_event_and_verification(
        _required_path(target.seed)
    )
    checks: list[tuple[str, bool]] = []

    entity_event = _committed_event_with_patches(
        committed_event,
        (
            StatePatch(
                operation=PatchOperation.MARK_STATUS,
                target_type=PatchTargetType.ENTITY,
                target_id="workshop_safe",
                path="/entity/workshop_safe/status",
                before="active",
                after="inactive",
                reason="regression entity status mark",
            ),
        ),
    )
    entity_world, entity_report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=entity_event,
        verification_result=verification,
    )
    entity = next(item for item in entity_world.entities if item.id == "workshop_safe")
    checks.append(("entity_status_mark_applies", entity_report.applied))
    checks.append(("entity_status_after_valid", entity.status == RecordStatus.INACTIVE))

    unsupported_event = _committed_event_with_patches(
        committed_event,
        (
            StatePatch(
                operation=PatchOperation.INCREMENT,
                target_type=PatchTargetType.RESOURCE,
                target_id="stabilizer_parts",
                path="/resource/stabilizer_parts/quantity",
                before=3,
                after=4,
                reason="regression staged quantity delta",
            ),
            StatePatch(
                operation=PatchOperation.MARK_STATUS,
                target_type=PatchTargetType.RESOURCE,
                target_id="stabilizer_parts",
                path="/resource/stabilizer_parts/status",
                before="active",
                after="inactive",
                reason="regression unsupported patch",
            ),
        ),
    )
    failed_world, failed_report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=unsupported_event,
        verification_result=verification,
    )
    resource = next(item for item in failed_world.resources if item.id == "stabilizer_parts")
    checks.append(("unsupported_patch_fails", not failed_report.applied))
    checks.append(("unsupported_patch_all_or_none", resource.quantity == 3))

    non_commit_verification = verification.model_copy(
        update={"decision": VerificationDecision.PENDING_GATE}
    )
    non_commit_world, non_commit_report = ControlledStateDiffApplier().apply(
        world_state=bundle.world,
        committed_event=entity_event,
        verification_result=non_commit_verification,
    )
    non_commit_entity = next(
        item for item in non_commit_world.entities if item.id == "workshop_safe"
    )
    checks.append(("non_commit_does_not_apply", not non_commit_report.applied))
    checks.append(("non_commit_world_unchanged", non_commit_entity.status == RecordStatus.ACTIVE))

    return _contract_target_summary(target, checks)


def _scheduler_evolution_contract_summary(
    target: RegressionTargetConfig,
) -> RegressionTargetSummary:
    seed_path = _required_path(target.seed)
    config_path = _required_path(target.config)
    result = run_world(seed_path=seed_path, config=load_run_config(config_path), apply=True)
    checks: list[tuple[str, bool]] = [
        ("provider_not_called", result.provider_called is False),
        (
            "scenario_matrix_plan_preserved",
            [step.agent_id for step in result.steps]
            == [
                "ivo",
                "mira",
                "selka",
                "selka",
                "ivo",
                "rowan",
                "player",
                "player",
            ],
        ),
    ]

    second_activation = result.steps[1].activation_result
    if second_activation is None:
        checks.append(("activation_present", False))
    else:
        reasons = {
            reason.reason_type: reason for reason in second_activation.selected_candidate.reasons
        }
        checks.extend(
            [
                (
                    "scheduler_version",
                    second_activation.scheduler_version == "deterministic_scheduler_v0",
                ),
                (
                    "recent_event_reason",
                    reasons.get("recent_committed_event_relevance") is not None
                    and reasons["recent_committed_event_relevance"].score == 1,
                ),
                (
                    "causal_open_thread_reason",
                    reasons.get("causal_open_thread_relevance") is not None
                    and reasons["causal_open_thread_relevance"].score == 1,
                ),
                (
                    "activation_safe",
                    "belief_ivo_key_in_safe" not in str(second_activation.safe_summary()),
                ),
            ]
        )

    bundle = _load_valid_bundle(seed_path)
    background_result = AgentActivationBuilder().build_for_step(
        bundle=bundle,
        run_id="v03_contract_scheduler",
        step=RunStepPlanItem(
            step_id="step_mira_wrong_key_reject",
            agent_id="mira",
            actor_type="agent",
            scenario_id="mira_search_archive_wrong_key",
        ),
        config=AgentActivationConfig(
            include_non_selected_candidates=True,
            top_k=1,
            selection_threshold=1,
        ),
    )
    checks.append(
        (
            "background_candidates",
            any(
                candidate.status == ActivationStatus.BACKGROUND
                for candidate in background_result.candidates
            ),
        )
    )

    first_update = result.steps[0].evolution_update
    reject_update = result.steps[1].evolution_update
    checks.extend(
        [
            (
                "pressure_governance_basis",
                first_update is not None
                and first_update.pressure_updates
                and first_update.pressure_updates[0].governance_basis
                == "commit_applied_state_diff",
            ),
            (
                "belief_non_canon",
                first_update is not None
                and first_update.belief_updates
                and first_update.belief_updates[0].canon_updated is False,
            ),
            (
                "memory_safe_summary",
                first_update is not None
                and "summary" not in first_update.safe_dict()["memory_updates"][0],
            ),
            (
                "relationship_bounded",
                first_update is not None
                and first_update.relationship_updates
                and -5 <= first_update.relationship_updates[0].trust_after <= 5,
            ),
            (
                "non_commit_trace_only_pressure",
                reject_update is not None
                and reject_update.pressure_updates
                and reject_update.pressure_updates[0].applied is False
                and reject_update.pressure_updates[0].governance_basis == "non_commit_trace_only",
            ),
        ]
    )
    return _contract_target_summary(target, checks, provider_called=result.provider_called)


def _contract_target_summary(
    target: RegressionTargetConfig,
    checks: list[tuple[str, bool]],
    *,
    provider_called: bool = False,
) -> RegressionTargetSummary:
    failures = tuple(name for name, passed in checks if not passed)
    return RegressionTargetSummary(
        target_id=target.id,
        kind=target.kind,
        status="pending",
        provider_called=provider_called,
        metric_count=len(checks),
        failed_metric_count=len(failures),
        failure_reason=", ".join(failures) if failures else None,
    )


def _matrix_config_load_summary(target: RegressionTargetConfig) -> RegressionTargetSummary:
    matrix_config = load_run_matrix_config(_required_path(target.matrix))
    evaluation_config = load_aggregate_evaluation_config(_required_path(target.evaluation))
    checks = [
        ("matrix_has_runs", len(matrix_config.runs) > 0),
        (
            "artifact_dir_under_runs",
            matrix_config.artifact_dir.replace("\\", "/").startswith("runs/"),
        ),
        ("evaluation_no_provider_default", evaluation_config.fail_on_provider_called),
        (
            "expected_ablation_failures_allowed",
            evaluation_config.allow_expected_ablation_failures,
        ),
    ]
    return _contract_target_summary(target, checks)


def _matrix_run_summary(target: RegressionTargetConfig) -> RegressionTargetSummary:
    matrix_config = load_run_matrix_config(_required_path(target.matrix))
    evaluation_config = load_aggregate_evaluation_config(_required_path(target.evaluation))
    summary = run_matrix(config=matrix_config, evaluation_config=evaluation_config)
    return RegressionTargetSummary(
        target_id=target.id,
        kind=target.kind,
        status="pending",
        provider_called=summary.overall.provider_called_any,
        metric_count=summary.overall.run_count + summary.overall.variant_count,
        failed_metric_count=(
            summary.overall.failed_run_count
            + summary.overall.unexpected_variant_failure_count
            + int(not summary.overall.artifact_safety_passed)
        ),
        variant_count=summary.overall.variant_count,
        artifact_paths={
            "matrix_summary_path": summary.matrix_summary_path,
            "aggregate_summary_path": summary.aggregate_summary_path,
        },
        failure_reason=None if summary.overall.passed else "matrix_overall_failed",
    )


def _aggregate_summary_target(target: RegressionTargetConfig) -> RegressionTargetSummary:
    summary = load_matrix_summary(_required_path(target.summary))
    checks = [
        ("has_per_run_summary", len(summary.runs) > 0),
        ("has_per_seed_summary", len(summary.seeds) > 0),
        ("has_per_family_summary", len(summary.families) > 0),
        ("has_per_variant_summary", len(summary.variants) > 0),
        ("has_bad_case_records", len(summary.bad_cases) > 0),
        ("has_bad_case_taxonomy", len(summary.overall.bad_case_taxonomy) > 0),
        ("family_count_matches", summary.overall.family_count == len(summary.families)),
        (
            "has_thresholds",
            summary.overall.thresholds.max_unexpected_bad_cases >= 0,
        ),
        (
            "unexpected_bad_cases_within_threshold",
            summary.overall.unexpected_bad_case_count
            <= summary.overall.thresholds.max_unexpected_bad_cases,
        ),
        ("overall_passed", summary.overall.passed),
        ("provider_not_called", not summary.overall.provider_called_any),
    ]
    return _contract_target_summary(
        target,
        checks,
        provider_called=summary.overall.provider_called_any,
    )


def _artifact_safety_target(target: RegressionTargetConfig) -> RegressionTargetSummary:
    summary = load_matrix_summary(_required_path(target.summary))
    checks = [
        (
            "artifact_dir_under_runs",
            summary.artifact_dir.replace("\\", "/").find("/runs/") >= 0,
        ),
        ("artifact_safety_passed", summary.overall.artifact_safety_passed),
        ("raw_text_not_saved", not summary.overall.raw_text_saved_any),
        ("provider_not_called", not summary.overall.provider_called_any),
    ]
    return _contract_target_summary(
        target, checks, provider_called=summary.overall.provider_called_any
    )


def _matrix_inspect_load_target(target: RegressionTargetConfig) -> RegressionTargetSummary:
    review = inspect_matrix_summary(_required_path(target.summary))
    checks = [
        ("has_matrix_id", bool(review.get("matrix_id"))),
        ("has_counts", review.get("run_count", 0) > 0 and review.get("variant_count", 0) > 0),
        ("has_families", bool(review.get("families"))),
        ("has_variant_status_by_type", bool(review.get("variant_status_by_type"))),
        ("has_bad_case_taxonomy", bool(review.get("bad_case_taxonomy"))),
        ("has_review_flags", bool(review.get("review_flags"))),
    ]
    return _contract_target_summary(
        target, checks, provider_called=bool(review.get("provider_called_any"))
    )


def _matrix_inspect_review_protocol_target(
    target: RegressionTargetConfig,
) -> RegressionTargetSummary:
    summary = load_matrix_summary(_required_path(target.summary))
    review = inspect_matrix_summary(_required_path(target.summary))
    thresholds = review["thresholds"]
    baseline_expected_ids = set(thresholds["expected_failure_variant_ids"])
    checks = [
        ("proposed_failures_visible", review["proposed_runtime_failure_count"] == 0),
        (
            "proposed_failures_not_expected",
            all(
                not variant.expected_failure
                for variant in summary.variants
                if variant.variant_type == "proposed"
            ),
        ),
        (
            "baseline_expected_has_rule",
            all(
                variant.variant_id in baseline_expected_ids
                for variant in summary.variants
                if variant.variant_type == "baseline" and variant.expected_failure
            ),
        ),
        (
            "ablation_expected_has_rule",
            bool(thresholds["allow_expected_ablation_failures"])
            or all(
                not variant.expected_failure
                for variant in summary.variants
                if variant.variant_type == "ablation"
            ),
        ),
        ("unexpected_bad_cases_visible", "unexpected_bad_case_count" in review),
    ]
    return _contract_target_summary(
        target, checks, provider_called=summary.overall.provider_called_any
    )


def _matrix_inspect_safety_target(target: RegressionTargetConfig) -> RegressionTargetSummary:
    review = inspect_matrix_summary(_required_path(target.summary))
    output = json.dumps(review, ensure_ascii=False) + render_matrix_review_markdown(review)
    forbidden_markers = (
        "raw_llm_text",
        "raw_text_content",
        "full_raw_text",
        "authorization:",
        "bearer ",
        "sk-",
        '"records"',
    )
    checks = [
        ("keeps_provider_flag", "provider_called_any" in review),
        ("keeps_raw_text_flag", "raw_text_saved_any" in review),
        ("keeps_artifact_safety_flag", "artifact_safety_passed" in review),
        (
            "no_raw_trace_or_secret_markers",
            not any(marker in output.lower() for marker in forbidden_markers),
        ),
    ]
    return _contract_target_summary(
        target, checks, provider_called=bool(review.get("provider_called_any"))
    )


def _scenario_contract_load_target(target: RegressionTargetConfig) -> RegressionTargetSummary:
    checks = [
        ("scenario_count_preserved", len(RUNTIME_SCENARIO_MATRIX) == 16),
        ("seed_family_present", all(scenario.seed_family for scenario in RUNTIME_SCENARIO_MATRIX)),
        (
            "rule_pack_present",
            all(_has_rule_pack(scenario.scenario_id) for scenario in RUNTIME_SCENARIO_MATRIX),
        ),
        (
            "state_diff_contracts_present",
            all(
                _has_state_diff_contract(scenario.scenario_id)
                for scenario in RUNTIME_SCENARIO_MATRIX
                if scenario.expects_state_diff
            ),
        ),
        (
            "real_llm_boundary_preserved",
            {
                scenario.scenario_id
                for scenario in RUNTIME_SCENARIO_MATRIX
                if scenario.allows_real_llm
            }
            == {"inspect_workshop_safe", "elin_inspect_cargo_manifest"},
        ),
    ]
    return _contract_target_summary(target, checks)


def _scenario_contract_fixture_coverage_target(
    target: RegressionTargetConfig,
) -> RegressionTargetSummary:
    checks: list[tuple[str, bool]] = []
    for scenario in RUNTIME_SCENARIO_MATRIX:
        if scenario.allows_real_llm:
            checks.append(
                (
                    f"{scenario.scenario_id}_has_no_fixture_contract",
                    scenario.fixture_contract_id is None,
                )
            )
        elif scenario.is_player_input:
            checks.append(
                (
                    f"{scenario.scenario_id}_player_input_contract",
                    _has_player_input_contract(scenario.scenario_id),
                )
            )
        else:
            checks.append(
                (
                    f"{scenario.scenario_id}_proposal_contract",
                    _has_proposal_fixture_contract(scenario.scenario_id),
                )
            )
    return _contract_target_summary(target, checks)


def _scenario_contract_verifier_rule_pack_parity_target(
    target: RegressionTargetConfig,
) -> RegressionTargetSummary:
    mistgate = run_world(
        seed_path=Path("seeds/mistgate_v01"),
        config=load_run_config(Path("configs/standard_run_deterministic_regression.yaml")),
    )
    harbor = run_world(
        seed_path=Path("seeds/harbor_lantern_v01"),
        config=load_run_config(
            Path("configs/harbor_lantern_standard_run_deterministic_regression.yaml")
        ),
    )
    checks = [
        ("mistgate_provider_not_called", mistgate.provider_called is False),
        ("harbor_provider_not_called", harbor.provider_called is False),
        (
            "mistgate_decisions_preserved",
            mistgate.decisions
            == (
                VerificationDecision.COMMIT,
                VerificationDecision.REJECT,
                VerificationDecision.COMMIT,
                VerificationDecision.COMMIT,
                VerificationDecision.REVISE,
                VerificationDecision.PENDING_GATE,
                VerificationDecision.REJECT,
                VerificationDecision.PENDING_GATE,
            ),
        ),
        (
            "harbor_decisions_preserved",
            harbor.decisions
            == (
                VerificationDecision.COMMIT,
                VerificationDecision.COMMIT,
                VerificationDecision.REJECT,
                VerificationDecision.PENDING_GATE,
                VerificationDecision.REJECT,
                VerificationDecision.PENDING_GATE,
            ),
        ),
        ("mistgate_commit_count_preserved", mistgate.committed_event_count == 3),
        ("harbor_commit_count_preserved", harbor.committed_event_count == 2),
        (
            "mistgate_state_diff_targets_preserved",
            _state_transition_targets(mistgate)
            == ["calibration_key", "stabilizer_parts", "market_credit"],
        ),
        (
            "harbor_state_diff_targets_preserved",
            _state_transition_targets(harbor) == ["harbor_pass", "relief_crates"],
        ),
    ]
    return _contract_target_summary(
        target,
        checks,
        provider_called=mistgate.provider_called or harbor.provider_called,
    )


def _scenario_contract_matrix_compatibility_target(
    target: RegressionTargetConfig,
) -> RegressionTargetSummary:
    review = inspect_matrix_summary(_required_path(target.summary))
    checks = [
        ("matrix_passed", review["passed"] is True),
        ("family_count_preserved", review["family_count"] == 2),
        ("variant_count_preserved", review["variant_count"] == 33),
        ("provider_not_called", review["provider_called_any"] is False),
        ("raw_text_not_saved", review["raw_text_saved_any"] is False),
        ("proposed_runtime_failure_count_preserved", review["proposed_runtime_failure_count"] == 0),
        ("unexpected_variant_failures_preserved", review["unexpected_variant_failure_count"] == 0),
        ("unexpected_bad_cases_preserved", review["unexpected_bad_case_count"] == 0),
    ]
    return _contract_target_summary(
        target,
        checks,
        provider_called=bool(review.get("provider_called_any")),
    )


def _scenario_contract_no_provider_target(
    target: RegressionTargetConfig,
) -> RegressionTargetSummary:
    mistgate = run_world(
        seed_path=Path("seeds/mistgate_v01"),
        config=load_run_config(Path("configs/standard_run_deterministic_regression.yaml")),
    )
    harbor = run_world(
        seed_path=Path("seeds/harbor_lantern_v01"),
        config=load_run_config(
            Path("configs/harbor_lantern_standard_run_deterministic_regression.yaml")
        ),
    )
    checks = [
        ("mistgate_provider_not_called", mistgate.provider_called is False),
        ("mistgate_raw_text_not_saved", mistgate.raw_text_saved is False),
        ("harbor_provider_not_called", harbor.provider_called is False),
        ("harbor_raw_text_not_saved", harbor.raw_text_saved is False),
    ]
    return _contract_target_summary(
        target,
        checks,
        provider_called=mistgate.provider_called or harbor.provider_called,
    )


def _provider_proposal_missing_settings_stops_before_chain_target(
    target: RegressionTargetConfig,
) -> RegressionTargetSummary:
    result = run_single_step(
        seed_path=Path("seeds/mistgate_v01"),
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
    )
    checks = [
        ("provider_not_called", result.provider_called is False),
        (
            "provider_unavailable_without_settings",
            result.failure_code == ProviderProposalFailureCode.PROVIDER_UNAVAILABLE,
        ),
        ("fallback_used", result.fallback_used is True),
        ("fallback_evidence_class", result.evidence_class == "fallback"),
        ("no_action_proposal", result.action_proposal is None),
        ("no_event_candidate", result.event_candidate is None),
        ("no_verification", result.verification_result is None),
        ("no_commit", result.committed_event is None),
    ]
    return _contract_target_summary(target, checks)


def _provider_proposal_schema_failure_no_event_candidate_target(
    target: RegressionTargetConfig,
) -> RegressionTargetSummary:
    result = _provider_structured_result("{not-json")
    checks = [
        ("fake_provider_called", result.provider_called is True),
        (
            "failure_malformed_output",
            result.failure_code == ProviderProposalFailureCode.MALFORMED_OUTPUT,
        ),
        ("no_action_proposal", result.action_proposal is None),
        ("no_event_candidate", result.event_candidate is None),
        ("no_verification", result.verification_result is None),
        ("no_commit", result.committed_event is None),
        ("no_state_diff", result.safe_summary()["state_diff_id"] is None),
    ]
    return _contract_target_summary(target, checks)


def _provider_proposal_governance_chain_target(
    target: RegressionTargetConfig,
) -> RegressionTargetSummary:
    result = _provider_structured_result(_valid_provider_action_json(), apply=True)
    state_diff = result.committed_event.state_diff if result.committed_event is not None else None
    checks = [
        ("fake_provider_called", result.provider_called is True),
        ("provider_source", result.proposal_source == "real_llm_structured_output"),
        ("action_proposal_created", result.action_proposal is not None),
        ("event_candidate_created_after_proposal", result.event_candidate is not None),
        (
            "verification_commit",
            result.verification_result is not None
            and result.verification_result.decision == VerificationDecision.COMMIT,
        ),
        ("committed_event_created", result.committed_event is not None),
        ("state_diff_created_after_commit", state_diff is not None),
        (
            "state_diff_not_sourced_from_action_proposal",
            state_diff is not None and state_diff.source_action_proposal_id is None,
        ),
        ("state_diff_applied_only_after_commit", result.state_diff_applied is True),
    ]
    return _contract_target_summary(target, checks)


def _provider_proposal_no_raw_text_artifacts_target(
    target: RegressionTargetConfig,
) -> RegressionTargetSummary:
    marker = "raw_provider_text_should_not_appear"
    result = _provider_structured_result(_valid_provider_action_json(rationale=marker))
    output = json.dumps(result.safe_summary(), ensure_ascii=False)
    checks = [
        ("has_raw_text_hash", bool(result.safe_summary()["raw_text_sha256"])),
        ("keeps_provider_called_flag", "provider_called" in result.safe_summary()),
        ("keeps_failure_code", "failure_code" in result.safe_summary()),
        ("raw_provider_text_not_saved", marker not in output.lower()),
        ("raw_prompt_not_saved", '"prompt"' not in output.lower()),
        (
            "secret_markers_absent",
            not any(secret in output.lower() for secret in ("sk-", "authorization", "api_key")),
        ),
    ]
    return _contract_target_summary(target, checks)


def _deterministic_regressions_still_no_provider_target(
    target: RegressionTargetConfig,
) -> RegressionTargetSummary:
    mistgate = run_world(
        seed_path=Path("seeds/mistgate_v01"),
        config=load_run_config(Path("configs/standard_run_deterministic_regression.yaml")),
    )
    harbor = run_world(
        seed_path=Path("seeds/harbor_lantern_v01"),
        config=load_run_config(
            Path("configs/harbor_lantern_standard_run_deterministic_regression.yaml")
        ),
    )
    review = inspect_matrix_summary(Path("runs/v05_run_matrix/matrix_summary.json"))
    checks = [
        ("mistgate_provider_not_called", mistgate.provider_called is False),
        ("harbor_provider_not_called", harbor.provider_called is False),
        ("matrix_provider_not_called", review["provider_called_any"] is False),
        ("matrix_raw_text_not_saved", review["raw_text_saved_any"] is False),
    ]
    return _contract_target_summary(target, checks)


def _provider_structured_result(content: str, *, apply: bool = False):
    return run_single_step(
        seed_path=Path("seeds/mistgate_v01"),
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
        provider=_FixtureProvider(content),
        proposal_source=ProposalSourceMode.PROVIDER_STRUCTURED,
        provider_proposals_enabled=True,
        allow_real_provider=True,
        apply=apply,
    )


class _FixtureProvider:
    provider_name = "fixture_test_provider"

    def __init__(self, content: str) -> None:
        self.content = content

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0):
        return LLMResult(
            content=self.content,
            model="fixture-test-model",
            latency_ms=1,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            attempts=(ProviderAttempt(model="fixture-test-model", success=True, latency_ms=1),),
        )

    def generate_structured(self, prompt, schema_type, *, max_tokens=512, temperature=0.0):
        return generate_structured(
            self,
            prompt,
            schema_type,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def _valid_provider_action_json(
    *,
    rationale: str = "Inspect through fixture provider.",
) -> str:
    return json.dumps(
        {
            "id": "proposal_inspect_workshop_safe_ivo",
            "proposer_agent_id": "ivo",
            "intent": "investigate",
            "rationale": rationale,
            "target_location_id": "workshop_lane",
            "target_entity_ids": ["workshop_safe"],
            "expected_outcome": "Inspect the workshop safe for the calibration key.",
        },
        separators=(",", ":"),
    )


def _has_rule_pack(scenario_id: str) -> bool:
    try:
        get_verifier_rule_pack(scenario_id)
        return True
    except ValueError:
        return False


def _has_state_diff_contract(scenario_id: str) -> bool:
    try:
        return get_state_diff_contract(scenario_id) is not None
    except ValueError:
        return False


def _has_proposal_fixture_contract(scenario_id: str) -> bool:
    try:
        get_proposal_fixture_contract(scenario_id)
        return True
    except ValueError:
        return False


def _has_player_input_contract(scenario_id: str) -> bool:
    try:
        get_player_input_fixture_contract(scenario_id)
        return True
    except ValueError:
        return False


def _state_transition_targets(result) -> list[str]:
    return [
        step.state_transition.patches[0].target_id
        for step in result.steps
        if step.state_transition is not None
    ]


def _committed_event_and_verification(seed_path: Path):
    bundle = _load_valid_bundle(seed_path)
    observation, cognition = build_agent_context(
        bundle,
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
    )
    proposal = ActionProposal(
        id="proposal_inspect_workshop_safe_ivo",
        proposer_agent_id="ivo",
        intent=ActionIntent.INVESTIGATE,
        rationale="Ivo has a lawful private reason to inspect his own workshop safe.",
        target_location_id="workshop_lane",
        target_entity_ids=("workshop_safe",),
        expected_outcome="Inspect the workshop safe for the calibration key.",
    )
    candidate = action_proposal_to_event_candidate(
        proposal,
        scenario_id="inspect_workshop_safe",
    )
    verification = DeterministicVerifier().verify(
        bundle=bundle,
        observation=observation,
        cognition=cognition,
        proposal=proposal,
        candidate=candidate,
        scenario_id="inspect_workshop_safe",
    )
    committed_event = build_committed_event(
        candidate=candidate,
        verification=verification,
        scenario_id="inspect_workshop_safe",
    )
    if committed_event is None:
        raise ValueError("state apply contract could not build committed event")
    return bundle, committed_event, verification


def _committed_event_with_patches(committed_event, patches):
    state_diff = StateDiff(
        id=f"{committed_event.state_diff.id}_contract",
        source_event_candidate_id=committed_event.event_candidate_id,
        committed_event_id=committed_event.id,
        patches=tuple(patches),
    )
    return committed_event.model_copy(update={"state_diff": state_diff})


def _load_valid_bundle(seed_path: Path):
    load_result = SeedLoader().load(seed_path)
    report = SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )
    if not report.success or load_result.bundle is None:
        raise ValueError(f"Seed validation failed: {report.safe_dict()}")
    return load_result.bundle


def _world_target_summary(target: RegressionTargetConfig, result) -> RegressionTargetSummary:
    cognitive = result.cognitive_runtime_summary or {}
    causal = result.causal_runtime_summary or {}
    pressure = result.pressure_runtime_summary or {}
    return RegressionTargetSummary(
        target_id=target.id,
        kind=target.kind,
        status="pending",
        step_count=result.step_count,
        formal_experiment_result=result.formal_experiment_result,
        provider_called=result.provider_called,
        state_diff_applied_count=result.state_diff_applied_count,
        causal_node_count=_int_or_none(causal.get("causal_node_count")),
        causal_edge_count=_int_or_none(causal.get("causal_edge_count")),
        pressure_update_count=_int_or_none(pressure.get("pressure_update_count")),
        belief_update_count=_int_or_none(cognitive.get("belief_update_count")),
        memory_signal_count=_int_or_none(cognitive.get("memory_signal_count")),
        relationship_signal_count=_int_or_none(cognitive.get("relationship_signal_count")),
    )


def _failure_reason(summary: RegressionTargetSummary) -> str | None:
    if summary.failure_reason:
        return summary.failure_reason
    if summary.provider_called:
        return "provider_called"
    if (
        summary.kind == RegressionTargetKind.RUN_WORLD_PREVIEW
        and summary.formal_experiment_result is not False
    ):
        return "preview_formal_experiment_result_not_false"
    if (
        summary.kind == RegressionTargetKind.RUN_WORLD_APPLY
        and (summary.state_diff_applied_count or 0) < 1
    ):
        return "apply_state_diff_applied_count_missing"
    if (
        summary.kind
        in {
            RegressionTargetKind.EXPERIMENT_RUN,
            RegressionTargetKind.TRACE_VALIDATE,
            RegressionTargetKind.EVALUATION_CHECK,
            RegressionTargetKind.EXPERIMENT_EVALUATE,
            RegressionTargetKind.EXPERIMENT_COMPARE,
        }
        and summary.formal_experiment_result is not True
    ):
        return "formal_experiment_result_not_true"
    if (
        summary.kind == RegressionTargetKind.EVALUATION_CHECK
        and (summary.failed_metric_count or 0) != 0
    ):
        return "evaluation_check_failed"
    if summary.kind == RegressionTargetKind.EXPERIMENT_EVALUATE:
        if (summary.failed_metric_count or 0) != 0:
            return "failed_metric_count_nonzero"
        if (summary.bad_case_count or 0) != 0:
            return "bad_case_count_nonzero"
    if summary.kind == RegressionTargetKind.EXPERIMENT_COMPARE and summary.variant_count != 11:
        return "variant_count_not_11"
    return None


def _required_path(value: str | None) -> Path:
    if value is None:
        raise ValueError("regression target missing required path")
    return Path(value).resolve()


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None
