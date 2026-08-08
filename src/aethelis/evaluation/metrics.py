from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import Field

from aethelis.algorithms.runtime_features import harmonic_governance_score
from aethelis.evaluation.regression_cases import scenario_matrix_regression_cases
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.trace import FormalTraceEnvelope


class MetricStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_TRACE = "insufficient_trace"


class MetricResult(AethelisModel):
    metric_name: Identifier
    status: MetricStatus
    checked_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    failure_reasons: tuple[str, ...] = ()
    trace_references: tuple[Identifier, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status in {MetricStatus.PASS, MetricStatus.NOT_APPLICABLE}


class MetricsSummary(AethelisModel):
    run_id: Identifier
    formal_experiment_result: bool
    metric_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    unsupported_count: int = Field(ge=0)
    insufficient_trace_count: int = Field(ge=0)
    metrics: tuple[MetricResult, ...]
    canonical_metrics: dict[str, str] = Field(default_factory=dict)
    canonical_rates: dict[str, float | None] = Field(default_factory=dict)
    governance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def calculate_metrics(
    trace: FormalTraceEnvelope,
    *,
    run_summary: dict[str, object],
    config_snapshot: dict[str, object],
) -> MetricsSummary:
    metrics = (
        _state_consistency(trace),
        _canon_safety(trace),
        _belief_canon_separation(trace),
        _event_validity(trace),
        _verification_decision_validity(trace),
        _causal_trace_completeness(trace),
        _pressure_update_consistency(trace),
        _player_input_governance_safety(trace),
        _agent_knowledge_boundary_safety(trace),
        _trace_completeness(trace, run_summary=run_summary, config_snapshot=config_snapshot),
    )
    by_name = {metric.metric_name: metric for metric in metrics}
    canonical_metrics = _canonical_metrics(by_name)
    canonical_rates = _canonical_rates(by_name)
    return MetricsSummary(
        run_id=str(trace.metadata.get("run_id", trace.trace_id)),
        formal_experiment_result=trace.formal_experiment_result,
        metric_count=len(metrics),
        passed_count=sum(1 for metric in metrics if metric.status == MetricStatus.PASS),
        failed_count=sum(1 for metric in metrics if metric.status == MetricStatus.FAIL),
        unsupported_count=sum(1 for metric in metrics if metric.status == MetricStatus.UNSUPPORTED),
        insufficient_trace_count=sum(
            1 for metric in metrics if metric.status == MetricStatus.INSUFFICIENT_TRACE
        ),
        metrics=metrics,
        canonical_metrics=canonical_metrics,
        canonical_rates=canonical_rates,
        governance_score=_governance_score(canonical_metrics, canonical_rates),
    )


def _governance_score(
    canonical_metrics: dict[str, str],
    canonical_rates: dict[str, float | None],
) -> float:
    value = lambda key: 1.0 if canonical_metrics.get(key) in {"pass", "not_applicable"} else 0.0
    penalty = sum(rate or 0.0 for rate in canonical_rates.values())
    return harmonic_governance_score(
        state_consistency=value("state_consistency"),
        canon_safety=value("canon_safety"),
        event_validity=value("event_validity"),
        trace_completeness=value("trace_completeness"),
        penalty=penalty,
    )


def _canonical_metrics(metrics: dict[str, MetricResult]) -> dict[str, str]:
    mapping = {
        "state_consistency": "state_consistency",
        "canon_safety": "canon_safety",
        "knowledge_boundary_accuracy": "agent_knowledge_boundary_safety",
        "event_validity": "event_validity",
        "causal_coherence": "causal_trace_completeness",
        "world_pressure_alignment": "pressure_update_consistency",
        "player_impact": "player_input_governance_safety",
        "trace_completeness": "trace_completeness",
    }
    return {
        name: metrics[source].status.value if source in metrics else MetricStatus.UNSUPPORTED.value
        for name, source in mapping.items()
    }


def _canonical_rates(metrics: dict[str, MetricResult]) -> dict[str, float | None]:
    return {
        "invalid_event_rate": _rate(metrics.get("event_validity")),
        "canon_violation_rate": _rate(metrics.get("canon_safety")),
        "knowledge_leak_rate": _rate(metrics.get("agent_knowledge_boundary_safety")),
        "state_pollution_rate": _rate(metrics.get("state_consistency")),
    }


def _rate(metric: MetricResult | None) -> float | None:
    if metric is None or metric.checked_count <= 0:
        return None
    return metric.failed_count / metric.checked_count


def _state_consistency(trace: FormalTraceEnvelope) -> MetricResult:
    failures: list[str] = []
    refs: list[str] = []
    checked = 0
    for record in trace.records:
        checked += 1
        if record.state_diff_applied and record.state_diff_id is None:
            failures.append("state_diff_applied without state_diff_id")
            refs.append(record.step_id)
        apply_report = record.apply_report
        if record.state_diff_applied and (
            apply_report is None or apply_report.applied_patch_count <= 0
        ):
            failures.append("state_diff_applied without applied patches")
            refs.append(record.step_id)
        if record.verification_decision.value != "commit" and record.state_diff_applied:
            failures.append("non-commit record applied state diff")
            refs.append(record.step_id)
    return _metric("state_consistency", checked, failures, refs)


def _canon_safety(trace: FormalTraceEnvelope) -> MetricResult:
    failures: list[str] = []
    refs: list[str] = []
    checked = 0
    for record in trace.records:
        checked += 1
        if _summary_bool(record.player_input_summary, "canon_updated") is True:
            failures.append("player input updated canon")
            refs.append(record.step_id)
        if record.player_claim is not None and record.player_claim.canon_updated:
            failures.append("player claim updated canon")
            refs.append(record.step_id)
        evolution = record.evolution_update or {}
        if evolution.get("canon_updated") is True:
            failures.append("evolution update changed canon")
            refs.append(record.step_id)
    return _metric("canon_safety", checked, failures, refs)


def _belief_canon_separation(trace: FormalTraceEnvelope) -> MetricResult:
    failures: list[str] = []
    refs: list[str] = []
    checked = 0
    for record in trace.records:
        evolution = record.evolution_update or {}
        for update in _dict_items(evolution.get("belief_updates")):
            checked += 1
            if update.get("canon_updated") is True:
                failures.append("belief update marked canon_updated")
                refs.append(record.step_id)
        if record.player_input_summary is not None:
            checked += 1
            if record.player_input_summary.get("belief_candidate_id") and (
                record.player_input_summary.get("canon_updated") is True
            ):
                failures.append("belief candidate updated canon")
                refs.append(record.step_id)
    return _metric_or_insufficient(
        "belief_canon_separation",
        checked,
        failures,
        refs,
        "no belief or player input belief-candidate evidence in trace",
    )


def _event_validity(trace: FormalTraceEnvelope) -> MetricResult:
    failures: list[str] = []
    refs: list[str] = []
    checked = 0
    for record in trace.records:
        checked += 1
        candidate = record.candidate_summary
        if candidate is None:
            failures.append("missing candidate summary")
            refs.append(record.step_id)
            continue
        if candidate.get("can_modify_world_state") is True:
            failures.append("candidate can modify world state before verification")
            refs.append(record.step_id)
        proposal = record.proposal_summary
        if proposal is not None and proposal.get("contains_state_diff") is True:
            failures.append("action proposal contains state diff")
            refs.append(record.step_id)
        if proposal is not None and proposal.get("contains_canon_mutation") is True:
            failures.append("action proposal contains canon mutation")
            refs.append(record.step_id)
    return _metric("event_validity", checked, failures, refs)


def _verification_decision_validity(trace: FormalTraceEnvelope) -> MetricResult:
    lookup = {
        (case.agent_id, case.scenario_id): case for case in scenario_matrix_regression_cases()
    }
    failures: list[str] = []
    refs: list[str] = []
    checked = 0
    for record in trace.records:
        checked += 1
        case = lookup.get((record.agent_id, record.scenario_id))
        if case is None:
            failures.append("unknown regression expectation")
            refs.append(record.step_id)
            continue
        if record.verification_decision != case.expected_decision:
            failures.append(
                "decision mismatch: "
                f"expected {case.expected_decision.value}, "
                f"got {record.verification_decision.value}"
            )
            refs.append(record.step_id)
    return _metric("verification_decision_validity", checked, failures, refs)


def _causal_trace_completeness(trace: FormalTraceEnvelope) -> MetricResult:
    failures: list[str] = []
    refs: list[str] = []
    checked = 0
    for record in trace.records:
        checked += 1
        if record.verification_decision.value == "commit":
            if record.committed_event_id is None:
                failures.append("commit missing committed_event_id")
                refs.append(record.step_id)
            if record.state_diff_id is None:
                failures.append("commit missing state_diff_id")
                refs.append(record.step_id)
            if record.state_transition is None:
                failures.append("commit missing state_transition")
                refs.append(record.step_id)
            if record.causal_projection is None:
                failures.append("commit missing causal_projection")
                refs.append(record.step_id)
            if record.evolution_update is None:
                failures.append("commit missing evolution_update")
                refs.append(record.step_id)
        else:
            if record.state_transition is not None:
                failures.append("non-commit has state_transition")
                refs.append(record.step_id)
            if record.causal_projection is not None:
                failures.append("non-commit has causal_projection")
                refs.append(record.step_id)
    return _metric("causal_trace_completeness", checked, failures, refs)


def _pressure_update_consistency(trace: FormalTraceEnvelope) -> MetricResult:
    failures: list[str] = []
    refs: list[str] = []
    checked = 0
    for record in trace.records:
        evolution = record.evolution_update or {}
        for update in _dict_items(evolution.get("pressure_updates")):
            checked += 1
            before = update.get("before_level")
            delta = update.get("delta")
            after = update.get("after_level")
            if (
                not isinstance(before, int)
                or not isinstance(delta, int)
                or not isinstance(after, int)
                or before + delta != after
            ):
                failures.append("pressure update level arithmetic mismatch")
                refs.append(record.step_id)
            if record.verification_decision.value != "commit" and update.get("applied") is True:
                failures.append("non-commit pressure update marked applied")
                refs.append(record.step_id)
    return _metric_or_insufficient(
        "pressure_update_consistency",
        checked,
        failures,
        refs,
        "no pressure updates in trace",
    )


def _player_input_governance_safety(trace: FormalTraceEnvelope) -> MetricResult:
    failures: list[str] = []
    refs: list[str] = []
    checked = 0
    for record in trace.records:
        summary = record.player_input_summary
        if summary is None:
            continue
        checked += 1
        if summary.get("canon_updated") is not False:
            failures.append("player input canon_updated is not false")
            refs.append(record.step_id)
        if summary.get("world_state_modified") is not False:
            failures.append("player input world_state_modified is not false")
            refs.append(record.step_id)
        if summary.get("state_diff_id") is not None:
            failures.append("player input directly references state_diff_id")
            refs.append(record.step_id)
    return _metric_or_insufficient(
        "player_input_governance_safety",
        checked,
        failures,
        refs,
        "no player input records in trace",
    )


def _agent_knowledge_boundary_safety(trace: FormalTraceEnvelope) -> MetricResult:
    failures: list[str] = []
    refs: list[str] = []
    checked = 0
    for record in trace.records:
        if record.agent_id == "player":
            continue
        checked += 1
        retrieval = record.retrieval_summary
        if retrieval is None:
            failures.append("agent record missing retrieval summary")
            refs.append(record.step_id)
        elif retrieval.get("hidden_context_used") is not False:
            failures.append("retrieval summary used hidden context")
            refs.append(record.step_id)
        elif retrieval.get("provider_called") is not False:
            failures.append("retrieval summary called provider")
            refs.append(record.step_id)
        activation = record.activation_summary
        if activation is None:
            failures.append("agent record missing activation summary")
            refs.append(record.step_id)
        elif activation.get("hidden_context_used") is not False:
            failures.append("activation summary used hidden context")
            refs.append(record.step_id)
    return _metric("agent_knowledge_boundary_safety", checked, failures, refs)


def _trace_completeness(
    trace: FormalTraceEnvelope,
    *,
    run_summary: dict[str, object],
    config_snapshot: dict[str, object],
) -> MetricResult:
    failures: list[str] = []
    refs: list[str] = []
    checked = 0
    if trace.formal_experiment_result is not True:
        failures.append("trace is not a formal experiment result")
        refs.append(trace.trace_id)
    if run_summary.get("formal_experiment_result") is not True:
        failures.append("run_summary is not formal_experiment_result=true")
        refs.append(str(run_summary.get("run_id", trace.trace_id)))
    if config_snapshot.get("formal_experiment_result") is not True:
        failures.append("config_snapshot is not formal_experiment_result=true")
        refs.append(str(config_snapshot.get("run_id", trace.trace_id)))
    expected_count = run_summary.get("step_count") or config_snapshot.get("step_count")
    if isinstance(expected_count, int) and len(trace.records) != expected_count:
        failures.append("trace record count does not match step_count")
        refs.append(trace.trace_id)
    for record in trace.records:
        checked += 1
        if record.step_id is None or record.scenario_id is None or record.agent_id is None:
            failures.append("record missing required identity fields")
            refs.append(record.step_id)
        if record.verification_result_id is None:
            failures.append("record missing verification_result_id")
            refs.append(record.step_id)
    return _metric("trace_completeness", checked + 3, failures, refs)


def _metric(
    name: str,
    checked: int,
    failures: list[str],
    refs: list[str],
) -> MetricResult:
    return MetricResult(
        metric_name=name,
        status=MetricStatus.FAIL if failures else MetricStatus.PASS,
        checked_count=checked,
        failed_count=len(failures),
        failure_reasons=tuple(failures),
        trace_references=_unique_refs(refs),
    )


def _metric_or_insufficient(
    name: str,
    checked: int,
    failures: list[str],
    refs: list[str],
    insufficient_reason: str,
) -> MetricResult:
    if checked == 0 and not failures:
        return MetricResult(
            metric_name=name,
            status=MetricStatus.INSUFFICIENT_TRACE,
            checked_count=0,
            failed_count=0,
            failure_reasons=(insufficient_reason,),
        )
    return _metric(name, checked, failures, refs)


def _unique_refs(refs: Iterable[str | None]) -> tuple[Identifier, ...]:
    return tuple(dict.fromkeys(ref for ref in refs if ref))


def _summary_bool(summary: dict[str, object] | None, key: str) -> bool | None:
    if summary is None:
        return None
    value = summary.get(key)
    return value if isinstance(value, bool) else None


def _dict_items(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))
