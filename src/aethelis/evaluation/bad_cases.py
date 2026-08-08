from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from aethelis.evaluation.metrics import MetricResult, MetricStatus
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.trace import FormalTraceEnvelope


class BadCaseSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BadCaseRecord(AethelisModel):
    case_id: Identifier
    failure_type: Identifier
    severity: BadCaseSeverity
    step_id: Identifier | None = None
    scenario_id: Identifier | None = None
    expected: str | None = None
    actual: str | None = None
    trace_reference: Identifier


class BadCaseSummary(AethelisModel):
    run_id: Identifier
    bad_case_count: int = Field(ge=0)
    bad_cases: tuple[BadCaseRecord, ...] = ()

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def collect_bad_cases(
    trace: FormalTraceEnvelope,
    *,
    metrics: tuple[MetricResult, ...],
) -> BadCaseSummary:
    cases: list[BadCaseRecord] = []
    for metric in metrics:
        if metric.status != MetricStatus.FAIL:
            continue
        refs = metric.trace_references or (trace.trace_id,)
        for index, reason in enumerate(metric.failure_reasons):
            ref = refs[min(index, len(refs) - 1)]
            record = _find_record(trace, ref)
            cases.append(
                BadCaseRecord(
                    case_id=f"bad_case_{len(cases) + 1:03d}_{metric.metric_name}",
                    failure_type=metric.metric_name,
                    severity=_severity(metric.metric_name),
                    step_id=record.step_id if record is not None else None,
                    scenario_id=record.scenario_id if record is not None else None,
                    expected=_expected(metric.metric_name),
                    actual=reason,
                    trace_reference=ref,
                )
            )
    return BadCaseSummary(
        run_id=str(trace.metadata.get("run_id", trace.trace_id)),
        bad_case_count=len(cases),
        bad_cases=tuple(cases),
    )


def _find_record(trace: FormalTraceEnvelope, ref: str):
    for record in trace.records:
        if record.step_id == ref:
            return record
    return None


def _severity(metric_name: str) -> BadCaseSeverity:
    if metric_name in {
        "canon_safety",
        "player_input_governance_safety",
        "agent_knowledge_boundary_safety",
        "state_consistency",
    }:
        return BadCaseSeverity.HIGH
    if metric_name in {
        "belief_canon_separation",
        "event_validity",
        "verification_decision_validity",
    }:
        return BadCaseSeverity.MEDIUM
    return BadCaseSeverity.LOW


def _expected(metric_name: str) -> str:
    return f"{metric_name} status=pass"
