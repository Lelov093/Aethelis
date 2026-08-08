from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import UserDefinedType


class Base(DeclarativeBase):
    pass


class Vector1024(UserDefinedType):
    cache_ok = True

    def get_col_spec(self, **_: object) -> str:
        return "vector(1024)"


def utcnow() -> datetime:
    return datetime.now(UTC)


def jsonb(default: Any | None = None) -> Mapped[dict[str, Any]]:
    return mapped_column(JSONB, nullable=False, default=lambda: dict(default or {}))


class RunRecord(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    seed_id: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    algorithm_mode: Mapped[str] = mapped_column(String(80), nullable=False)
    provider_called: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    db_persisted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    metadata_json: Mapped[dict[str, Any]] = jsonb()


class RunStepRecord(Base):
    __tablename__ = "run_steps"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    scenario_id: Mapped[str] = mapped_column(String(120), nullable=False)
    agent_id: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    provider_called: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    structured_validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    state_diff_applied: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    metadata_json: Mapped[dict[str, Any]] = jsonb()


class ProviderCallRecord(Base):
    __tablename__ = "provider_call_records"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    provider_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_name: Mapped[str] = mapped_column(String(240), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    usage_json: Mapped[dict[str, Any]] = jsonb()
    attempts_json: Mapped[dict[str, Any]] = jsonb()


class LLMInputRecord(Base):
    __tablename__ = "llm_input_records"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    provider_call_id: Mapped[str] = mapped_column(
        ForeignKey("provider_call_records.id", ondelete="CASCADE")
    )
    prompt_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    redaction_status: Mapped[str] = mapped_column(String(80), nullable=False)
    schema_name: Mapped[str] = mapped_column(String(120), nullable=False)


class LLMOutputRecord(Base):
    __tablename__ = "llm_output_records"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    provider_call_id: Mapped[str] = mapped_column(
        ForeignKey("provider_call_records.id", ondelete="CASCADE")
    )
    raw_text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_output_saved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    redaction_status: Mapped[str] = mapped_column(String(80), nullable=False)
    structured_json: Mapped[dict[str, Any]] = jsonb()


class StructuredOutputValidationRecord(Base):
    __tablename__ = "structured_output_validations"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    provider_call_id: Mapped[str] = mapped_column(
        ForeignKey("provider_call_records.id", ondelete="CASCADE")
    )
    schema_name: Mapped[str] = mapped_column(String(120), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    json_parse_error: Mapped[str | None] = mapped_column(Text)
    validation_error: Mapped[str | None] = mapped_column(Text)


class ActionProposalRecord(Base):
    __tablename__ = "action_proposals"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    proposal_json: Mapped[dict[str, Any]] = jsonb()


class EventCandidateRecord(Base):
    __tablename__ = "event_candidates"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    action_proposal_id: Mapped[str | None] = mapped_column(ForeignKey("action_proposals.id"))
    candidate_json: Mapped[dict[str, Any]] = jsonb()


class VerificationResultRecord(Base):
    __tablename__ = "verification_results"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    event_candidate_id: Mapped[str] = mapped_column(ForeignKey("event_candidates.id"))
    decision: Mapped[str] = mapped_column(String(40), nullable=False)
    verification_json: Mapped[dict[str, Any]] = jsonb()


class VerifierCheckResultRecord(Base):
    __tablename__ = "verifier_check_results"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    verification_result_id: Mapped[str] = mapped_column(
        ForeignKey("verification_results.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)


class CommittedEventRecord(Base):
    __tablename__ = "committed_events"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    event_candidate_id: Mapped[str] = mapped_column(ForeignKey("event_candidates.id"))
    verification_result_id: Mapped[str] = mapped_column(ForeignKey("verification_results.id"))
    committed_event_json: Mapped[dict[str, Any]] = jsonb()


class StateDiffRecord(Base):
    __tablename__ = "state_diffs"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    committed_event_id: Mapped[str] = mapped_column(ForeignKey("committed_events.id"))
    event_candidate_id: Mapped[str] = mapped_column(ForeignKey("event_candidates.id"))
    state_diff_json: Mapped[dict[str, Any]] = jsonb()


class StatePatchRecord(Base):
    __tablename__ = "state_patches"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    state_diff_id: Mapped[str] = mapped_column(ForeignKey("state_diffs.id", ondelete="CASCADE"))
    patch_index: Mapped[int] = mapped_column(Integer, nullable=False)
    operation: Mapped[str] = mapped_column(String(40), nullable=False)
    target_type: Mapped[str] = mapped_column(String(80), nullable=False)
    target_id: Mapped[str] = mapped_column(String(120), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    patch_json: Mapped[dict[str, Any]] = jsonb()


class WorldStateSnapshotRecord(Base):
    __tablename__ = "world_state_snapshots"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    snapshot_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    world_state_json: Mapped[dict[str, Any]] = jsonb()


class AlgorithmDecisionRecord(Base):
    __tablename__ = "algorithm_decisions"

    id: Mapped[str] = mapped_column(String(180), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    mechanism_id: Mapped[str] = mapped_column(String(120), nullable=False)
    model_family: Mapped[str] = mapped_column(String(160), nullable=False)
    formula: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(120), nullable=False)
    runtime_object_type: Mapped[str] = mapped_column(String(80), nullable=False)
    runtime_object_id: Mapped[str] = mapped_column(String(160), nullable=False)
    input_features_json: Mapped[dict[str, Any]] = jsonb()


class AlgorithmScoreBreakdownRecord(Base):
    __tablename__ = "algorithm_score_breakdowns"

    id: Mapped[str] = mapped_column(String(220), primary_key=True)
    algorithm_decision_id: Mapped[str] = mapped_column(
        ForeignKey("algorithm_decisions.id", ondelete="CASCADE")
    )
    score_name: Mapped[str] = mapped_column(String(120), nullable=False)
    score_value: Mapped[float | None] = mapped_column(Float)
    detail_json: Mapped[dict[str, Any]] = jsonb()


class TraceEventRecord(Base):
    __tablename__ = "trace_events"

    id: Mapped[str] = mapped_column(String(180), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(120), nullable=False)
    event_json: Mapped[dict[str, Any]] = jsonb()


class EvidenceArtifactRecord(Base):
    __tablename__ = "evidence_artifacts"

    id: Mapped[str] = mapped_column(String(180), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    artifact_type: Mapped[str] = mapped_column(String(120), nullable=False)
    redaction_status: Mapped[str] = mapped_column(String(120), nullable=False)
    artifact_json: Mapped[dict[str, Any]] = jsonb()


class MetricResultRecord(Base):
    __tablename__ = "metric_results"

    id: Mapped[str] = mapped_column(String(180), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    metric_name: Mapped[str] = mapped_column(String(120), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_json: Mapped[dict[str, Any]] = jsonb()


class EmbeddingRecord(Base):
    __tablename__ = "embedding_records"

    embedding_id: Mapped[str] = mapped_column(String(180), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    step_id: Mapped[str] = mapped_column(ForeignKey("run_steps.id", ondelete="CASCADE"))
    provider_call_id: Mapped[str] = mapped_column(ForeignKey("provider_call_records.id"))
    source_type: Mapped[str] = mapped_column(String(120), nullable=False)
    source_object_id: Mapped[str] = mapped_column(String(180), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(240), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_norm: Mapped[float] = mapped_column(Float, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    redaction_status: Mapped[str] = mapped_column(String(120), nullable=False)
    provider_called: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    metadata_json: Mapped[dict[str, Any]] = jsonb()


class EmbeddingChunkRecord(Base):
    __tablename__ = "embedding_chunks"

    chunk_id: Mapped[str] = mapped_column(String(220), primary_key=True)
    embedding_id: Mapped[str] = mapped_column(
        ForeignKey("embedding_records.embedding_id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    source_type: Mapped[str] = mapped_column(String(120), nullable=False)
    source_object_id: Mapped[str] = mapped_column(String(180), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_vector: Mapped[str] = mapped_column(Vector1024(), nullable=False)
    vector_norm: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    metadata_json: Mapped[dict[str, Any]] = jsonb()
