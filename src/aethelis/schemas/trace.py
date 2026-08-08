from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.events import VerificationDecision


class TraceType(StrEnum):
    DEBUG = "debug"
    FORMAL = "formal"


class ProviderTraceMetadata(AethelisModel):
    provider_mode: Identifier | None = None
    fallback_used: bool = False
    fallback_reason: Identifier | None = None
    evidence_class: Identifier | None = None
    provider_name: str | None = None
    model_name: str | None = None
    latency_ms: int | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    raw_text_sha256: str | None = None


class ApplyTraceSummary(AethelisModel):
    applied: bool
    applied_patch_count: int = 0
    skipped_patch_count: int = 0
    errors: tuple[str, ...] = ()


class PlayerClaimTraceRecord(AethelisModel):
    claim_id: Identifier
    player_id: Identifier
    claim_summary: str = Field(min_length=1)
    verification_decision: VerificationDecision
    rejected_claim_ids: tuple[Identifier, ...] = ()
    canon_updated: bool = False
    state_diff_id: Identifier | None = None


class TraceStepTransaction(AethelisModel):
    step_id: Identifier
    scenario_id: Identifier
    agent_id: Identifier
    action_proposal_id: Identifier | None = None
    proposal_summary: dict[str, Any] | None = None
    event_candidate_id: Identifier | None = None
    candidate_summary: dict[str, Any] | None = None
    verification_result_id: Identifier | None = None
    verification_decision: VerificationDecision
    verification_checks: tuple[dict[str, Any], ...] = ()
    verification_reasons: tuple[str, ...] = ()
    verification_risk_flags: tuple[Identifier, ...] = ()
    committed_event_id: Identifier | None = None
    state_diff_id: Identifier | None = None
    state_diff_applied: bool = False
    apply_report: ApplyTraceSummary | None = None
    state_transition: dict[str, Any] | None = None
    causal_projection: dict[str, Any] | None = None
    evolution_update: dict[str, Any] | None = None
    player_input_summary: dict[str, Any] | None = None
    retrieval_summary: dict[str, Any] | None = None
    proposal_source: Identifier | None = None
    provider_metadata: ProviderTraceMetadata | None = None
    activation_summary: dict[str, Any] | None = None
    safety_flags: tuple[str, ...] = ()
    player_claim: PlayerClaimTraceRecord | None = None
    notes: tuple[str, ...] = ()


class FormalTraceEnvelope(AethelisModel):
    trace_id: Identifier
    trace_type: TraceType = TraceType.FORMAL
    formal_experiment_result: bool = False
    schema_version: str = Field(min_length=1)
    seed_id: Identifier
    scenario_id: Identifier
    agent_id: Identifier
    runtime_phase: str = Field(default="phase_1f", min_length=1)
    records: tuple[TraceStepTransaction, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimulationTraceHeader(AethelisModel):
    trace_id: Identifier
    run_id: Identifier
    seed_id: Identifier
    schema_version: str = Field(min_length=1)
    created_by: str = Field(default="aethelis", min_length=1)


class TraceStepRecord(AethelisModel):
    step_id: Identifier
    step_index: int = Field(ge=0)
    action_proposal_ids: tuple[Identifier, ...] = ()
    event_candidate_ids: tuple[Identifier, ...] = ()
    verification_result_ids: tuple[Identifier, ...] = ()
    committed_event_ids: tuple[Identifier, ...] = ()
    notes: tuple[str, ...] = ()
