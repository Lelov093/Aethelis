from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from aethelis.runtime.scenario_matrix import get_scenario_definition
from aethelis.runtime.single_step import SingleStepResult
from aethelis.schemas.events import ActionProposalSummary, EventCandidateSummary
from aethelis.schemas.run import WorldRunResult, WorldStepResult
from aethelis.schemas.trace import (
    ApplyTraceSummary,
    FormalTraceEnvelope,
    PlayerClaimTraceRecord,
    ProviderTraceMetadata,
    TraceStepTransaction,
)
from aethelis.utils.redaction import redact_text

SECRET_MARKERS = (
    "sk-",
    "authorization",
    "api_key",
    "openai_api_key",
    "embedding_api_key",
    "ark_api_key",
)
RAW_TEXT_KEY_MARKERS = ("raw_llm_text", "full_raw_text", "raw_text_content", "raw_text")


@dataclass(frozen=True)
class TraceValidationReport:
    path: Path
    success: bool
    trace_type: str | None
    formal_experiment_result: bool | None
    schema_version: str | None
    record_count: int
    decisions: tuple[str, ...]
    has_raw_text: bool
    has_secret_markers: bool
    error: str | None = None

    def safe_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "success": self.success,
            "trace_type": self.trace_type,
            "formal_experiment_result": self.formal_experiment_result,
            "schema_version": self.schema_version,
            "record_count": self.record_count,
            "decisions": list(self.decisions),
            "has_raw_text": self.has_raw_text,
            "has_secret_markers": self.has_secret_markers,
            "error": redact_text(self.error) if self.error is not None else None,
        }


def build_formal_trace_preview(
    result: SingleStepResult,
    *,
    seed_id: str,
    schema_version: str = "0.1",
) -> FormalTraceEnvelope:
    """Build a formal-shaped, non-experiment trace preview.

    Phase 1F defines and validates this contract but does not make it a formal
    experiment result.
    """

    decision = _require_decision(result)
    record = TraceStepTransaction(
        step_id=f"step_{result.scenario_id}_{result.agent_id}",
        scenario_id=result.scenario_id,
        agent_id=result.agent_id,
        action_proposal_id=(
            result.action_proposal.id if result.action_proposal is not None else None
        ),
        proposal_summary=(
            ActionProposalSummary.from_proposal(
                result.action_proposal,
                generated_by=result.proposal_source or "deterministic_fixture",
            ).model_dump(mode="json")
            if result.action_proposal is not None
            else None
        ),
        event_candidate_id=(
            result.event_candidate.id if result.event_candidate is not None else None
        ),
        candidate_summary=(
            EventCandidateSummary.from_candidate(
                result.event_candidate,
                candidate_kind=_candidate_kind(result.scenario_id),
            ).model_dump(mode="json")
            if result.event_candidate is not None
            else None
        ),
        verification_result_id=(
            result.verification_result.id if result.verification_result is not None else None
        ),
        verification_decision=decision,
        verification_checks=_verification_checks(result),
        verification_reasons=(
            result.verification_result.reasons if result.verification_result is not None else ()
        ),
        verification_risk_flags=(
            result.verification_result.risk_flags if result.verification_result is not None else ()
        ),
        committed_event_id=(
            result.committed_event.id if result.committed_event is not None else None
        ),
        state_diff_id=(
            result.committed_event.state_diff.id if result.committed_event is not None else None
        ),
        state_diff_applied=result.state_diff_applied,
        apply_report=_apply_summary(result),
        state_transition=None,
        causal_projection=None,
        evolution_update=None,
        player_input_summary=result.player_input_summary,
        retrieval_summary=result.retrieval_summary,
        proposal_source=result.proposal_source,
        provider_metadata=_provider_metadata(result),
        safety_flags=_safety_flags(result),
        player_claim=(
            PlayerClaimTraceRecord(
                claim_id=result.player_claim.claim_id,
                player_id=result.player_claim.player_id,
                claim_summary=_summarize_text(result.player_claim.claim),
                verification_decision=result.player_claim.verification_result.decision,
                rejected_claim_ids=result.player_claim.verification_result.rejected_claim_ids,
                canon_updated=result.player_claim.canon_updated,
                state_diff_id=result.player_claim.state_diff_id,
            )
            if result.player_claim is not None
            else None
        ),
        notes=_notes(result),
    )
    return FormalTraceEnvelope(
        trace_id=f"trace_{result.scenario_id}_{result.agent_id}",
        formal_experiment_result=False,
        schema_version=schema_version,
        seed_id=seed_id,
        scenario_id=result.scenario_id,
        agent_id=result.agent_id,
        records=(record,),
    )


def write_formal_trace_preview(
    result: SingleStepResult,
    path: Path,
    *,
    seed_id: str,
    schema_version: str = "0.1",
) -> Path:
    trace = build_formal_trace_preview(
        result,
        seed_id=seed_id,
        schema_version=schema_version,
    )
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def build_world_run_trace_preview(
    result: WorldRunResult,
    *,
    seed_id: str,
    schema_version: str = "0.1",
) -> FormalTraceEnvelope:
    """Build a run-level formal-shaped preview, not a formal experiment result."""

    return FormalTraceEnvelope(
        trace_id=f"trace_{result.run_id}",
        formal_experiment_result=False,
        schema_version=schema_version,
        seed_id=seed_id,
        scenario_id="world_run_preview",
        agent_id="multi_actor",
        runtime_phase="runtime_foundation_preview",
        records=tuple(_run_step_transaction(step) for step in result.steps),
        metadata={
            "run_id": result.run_id,
            "mode": result.mode.value,
            "step_count": result.step_count,
            "wrote_runs": False,
            "wrote_reports": False,
            "raw_text_saved": False,
            "provider_called": result.provider_called,
            "activation_trace_included": any(
                step.activation_result is not None for step in result.steps
            ),
            "activation_mode": (
                result.steps[0].activation_result.mode.value
                if result.steps and result.steps[0].activation_result is not None
                else None
            ),
            "activation_provider_called": any(
                step.activation_result.provider_called
                for step in result.steps
                if step.activation_result is not None
            ),
            "final_evolution_state_summary": result.final_evolution_state_summary,
        },
    )


def build_world_run_formal_experiment_trace(
    result: WorldRunResult,
    *,
    seed_id: str,
    metadata: dict[str, object],
    schema_version: str = "0.1",
) -> FormalTraceEnvelope:
    """Build a formal experiment trace from an explicit experiment-run result."""

    return FormalTraceEnvelope(
        trace_id=f"trace_{result.run_id}",
        formal_experiment_result=True,
        schema_version=schema_version,
        seed_id=seed_id,
        scenario_id="formal_experiment_run",
        agent_id="multi_actor",
        runtime_phase="phase_7_formal_experiment",
        records=tuple(_formal_run_step_transaction(step) for step in result.steps),
        metadata={
            **metadata,
            "formal_experiment_result": True,
            "trace_source": "experiment-run",
        },
    )


def write_world_run_trace_preview(
    result: WorldRunResult,
    path: Path,
    *,
    seed_id: str,
    schema_version: str = "0.1",
) -> Path:
    trace = build_world_run_trace_preview(
        result,
        seed_id=seed_id,
        schema_version=schema_version,
    )
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_formal_trace(path: Path) -> FormalTraceEnvelope:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FormalTraceEnvelope.model_validate(payload)


def validate_formal_trace_file(path: Path) -> TraceValidationReport:
    path = path.resolve()
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        trace = FormalTraceEnvelope.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raw = "" if not path.exists() else path.read_text(encoding="utf-8", errors="ignore")
        payload = _safe_json_loads(raw)
        return TraceValidationReport(
            path=path,
            success=False,
            trace_type=None,
            formal_experiment_result=None,
            schema_version=None,
            record_count=0,
            decisions=(),
            has_raw_text=_contains_raw_text_key(payload),
            has_secret_markers=_contains_any(raw, SECRET_MARKERS),
            error=_safe_error_summary(exc),
        )

    has_raw_text = _contains_raw_text_key(payload)
    has_secret_markers = _contains_any(raw, SECRET_MARKERS)
    decisions = tuple(record.verification_decision.value for record in trace.records)
    success = trace.trace_type.value == "formal" and not has_raw_text and not has_secret_markers
    return TraceValidationReport(
        path=path,
        success=success,
        trace_type=trace.trace_type.value,
        formal_experiment_result=trace.formal_experiment_result,
        schema_version=trace.schema_version,
        record_count=len(trace.records),
        decisions=decisions,
        has_raw_text=has_raw_text,
        has_secret_markers=has_secret_markers,
        error=None if success else "Trace failed safety validation.",
    )


def inspect_formal_trace_file(path: Path) -> dict[str, object]:
    return validate_formal_trace_file(path).safe_dict()


def _require_decision(result: SingleStepResult):
    if result.verification_result is None:
        raise ValueError("Cannot build formal trace without verification_result")
    return result.verification_result.decision


def _run_step_transaction(step: WorldStepResult) -> TraceStepTransaction:
    return TraceStepTransaction(
        step_id=step.step_id,
        scenario_id=step.scenario_id,
        agent_id=step.agent_id,
        action_proposal_id=step.action_proposal_id,
        proposal_summary=step.proposal_summary_dict(),
        event_candidate_id=step.event_candidate_id,
        candidate_summary=step.candidate_summary_dict(),
        verification_result_id=step.verification_result_id,
        verification_decision=step.decision,
        verification_checks=step.verification_checks,
        verification_reasons=step.verification_reasons,
        verification_risk_flags=step.verification_risk_flags,
        committed_event_id=step.committed_event_id,
        state_diff_id=step.state_diff_id,
        state_diff_applied=step.state_diff_applied,
        apply_report=_run_apply_summary(step),
        state_transition=(
            step.state_transition.model_dump(mode="json")
            if step.state_transition is not None
            else None
        ),
        causal_projection=(
            step.causal_projection.model_dump(mode="json")
            if step.causal_projection is not None
            else None
        ),
        evolution_update=(
            step.evolution_update.safe_dict() if step.evolution_update is not None else None
        ),
        player_input_summary=step.player_input_summary,
        retrieval_summary=step.retrieval_summary,
        proposal_source=step.proposal_source,
        provider_metadata=ProviderTraceMetadata(
            provider_mode=step.provider_mode,
            fallback_used=step.fallback_used,
            fallback_reason=step.fallback_reason,
            evidence_class=step.evidence_class,
        ),
        activation_summary=step.activation_summary_dict(),
        safety_flags=step.safety_flags,
        player_claim=_run_player_claim(step),
        notes=step.notes,
    )


def _formal_run_step_transaction(step: WorldStepResult) -> TraceStepTransaction:
    record = _run_step_transaction(step)
    return record.model_copy(
        update={
            "safety_flags": _replace_formal_experiment_flag(
                record.safety_flags,
                formal_experiment_result=True,
            )
        }
    )


def _replace_formal_experiment_flag(
    flags: tuple[str, ...],
    *,
    formal_experiment_result: bool,
) -> tuple[str, ...]:
    replacement = (
        "formal_experiment_result_true"
        if formal_experiment_result
        else "formal_experiment_result_false"
    )
    filtered = tuple(
        flag
        for flag in flags
        if flag not in {"formal_experiment_result_false", "formal_experiment_result_true"}
    )
    return (*filtered, replacement)


def _run_apply_summary(step: WorldStepResult) -> ApplyTraceSummary | None:
    if step.apply_report is None:
        return None
    return ApplyTraceSummary(
        applied=bool(step.apply_report.get("applied", False)),
        applied_patch_count=int(step.apply_report.get("applied_patch_count", 0)),
        skipped_patch_count=int(step.apply_report.get("skipped_patch_count", 0)),
        errors=tuple(str(error) for error in step.apply_report.get("errors", ())),
    )


def _candidate_kind(scenario_id: str) -> str | None:
    try:
        return get_scenario_definition(scenario_id).candidate_kind
    except ValueError:
        return None


def _run_player_claim(step: WorldStepResult) -> PlayerClaimTraceRecord | None:
    if step.player_claim_id is None or step.player_claim_summary is None:
        return None
    return PlayerClaimTraceRecord(
        claim_id=step.player_claim_id,
        player_id=step.agent_id,
        claim_summary=step.player_claim_summary,
        verification_decision=step.decision,
        rejected_claim_ids=step.player_claim_rejected_claim_ids,
        canon_updated=step.player_claim_canon_updated,
        state_diff_id=step.player_claim_state_diff_id,
    )


def _provider_metadata(result: SingleStepResult) -> ProviderTraceMetadata | None:
    if result.structured_output is None:
        return None
    return ProviderTraceMetadata(
        provider_mode=result.provider_mode,
        fallback_used=result.fallback_used,
        fallback_reason=result.fallback_reason,
        evidence_class=result.evidence_class,
        provider_name=result.structured_output.provider_name,
        model_name=result.structured_output.model_name,
        latency_ms=result.structured_output.latency_ms,
        usage=result.structured_output.usage,
        raw_text_sha256=result.structured_output.raw_text_sha256,
    )


def _verification_checks(result: SingleStepResult) -> tuple[dict[str, object], ...]:
    if result.verification_result is None:
        return ()
    return tuple(
        {
            "name": check.name,
            "passed": check.passed,
            "message": check.message,
        }
        for check in result.verification_result.checks
    )


def _apply_summary(result: SingleStepResult) -> ApplyTraceSummary | None:
    if result.apply_report is None:
        return None
    return ApplyTraceSummary(
        applied=result.apply_report.applied,
        applied_patch_count=result.apply_report.applied_patch_count,
        skipped_patch_count=result.apply_report.skipped_patch_count,
        errors=result.apply_report.errors,
    )


def _safety_flags(result: SingleStepResult) -> tuple[str, ...]:
    flags = [
        result.provider_mode or "provider_mode_unknown",
        result.evidence_class or "evidence_class_unknown",
        "llm_output_hash_only",
        "formal_experiment_result_false",
    ]
    if result.fallback_used:
        flags.append("fallback_used")
    if result.fallback_reason is not None:
        flags.append(f"fallback_reason:{result.fallback_reason}")
    if result.committed_event is None:
        flags.append("non_commit_no_committed_event")
    if not result.state_diff_applied:
        flags.append("state_diff_not_applied")
    return tuple(flags)


def _notes(result: SingleStepResult) -> tuple[str, ...]:
    if result.verification_result is None:
        return ()
    return result.verification_result.reasons


def _summarize_text(value: str, limit: int = 120) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
    normalized = value.lower()
    return any(marker in normalized for marker in markers)


def _safe_json_loads(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _contains_raw_text_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).lower()
            if normalized_key in RAW_TEXT_KEY_MARKERS:
                return True
            if _contains_raw_text_key(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_text_key(item) for item in value)
    return False


def _safe_error_summary(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "JSONDecodeError: invalid trace JSON"
    if isinstance(exc, ValidationError):
        return "ValidationError: trace schema validation failed"
    if isinstance(exc, OSError):
        return f"{exc.__class__.__name__}: trace file could not be read"
    return f"{exc.__class__.__name__}: trace validation failed"
