from __future__ import annotations

import json
import math
from collections.abc import Iterable
from hashlib import sha256

from sqlalchemy import func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import aethelis.db.models as models
from aethelis.algorithms.mechanisms import MechanismKind
from aethelis.algorithms.runtime_wiring import RuntimeAlgorithmDecision
from aethelis.embedding.base import EmbeddingResult
from aethelis.runtime.single_step import SingleStepResult, build_step_context
from aethelis.schemas.seed import SeedBundle
from aethelis.schemas.world import WorldState
from aethelis.utils.redaction import redact_text


class RuntimeDBRepository:
    def __init__(self, engine: Engine) -> None:
        self._session = sessionmaker(engine, expire_on_commit=False)

    def persist_real_provider_step(
        self,
        *,
        run_id: str,
        step_id: str,
        seed_id: str,
        seed_bundle: SeedBundle,
        result: SingleStepResult,
        algorithm_mode: str,
        algorithm_decisions: tuple[RuntimeAlgorithmDecision, ...],
        embedding_result: EmbeddingResult | None = None,
        step_index: int = 1,
        before_world_state: WorldState | None = None,
    ) -> None:
        if result.structured_output is None or result.action_proposal is None:
            raise ValueError(
                "Cannot persist DB-backed real-provider run without structured output."
            )
        if result.event_candidate is None or result.verification_result is None:
            raise ValueError("Cannot persist incomplete governed runtime chain.")
        if result.committed_event is None or result.applied_world_state is None:
            raise ValueError(
                "DB-backed real-provider run requires committed event and applied copy."
            )

        effective_seed_bundle = seed_bundle.model_copy(
            update={"world": before_world_state or seed_bundle.world}
        )
        with self._session.begin() as session:
            session.merge(
                models.RunRecord(
                    id=run_id,
                    seed_id=seed_id,
                    status="completed",
                    algorithm_mode=algorithm_mode,
                    provider_called=result.provider_called,
                    fallback_used=result.fallback_used,
                    db_persisted=True,
                    metadata_json={
                        "evidence_class": result.evidence_class,
                        "provider_mode": result.provider_mode,
                    },
                )
            )
            session.add(
                models.RunStepRecord(
                    id=step_id,
                    run_id=run_id,
                    step_index=step_index,
                    scenario_id=result.scenario_id,
                    agent_id=result.agent_id,
                    status="completed",
                    provider_called=result.provider_called,
                    fallback_used=result.fallback_used,
                    structured_validation_passed=result.structured_output.success,
                    state_diff_applied=result.state_diff_applied,
                    metadata_json={"dry_run": result.dry_run},
                )
            )
            session.flush()
            self._persist_provider(session, run_id, step_id, result, effective_seed_bundle)
            session.flush()
            self._persist_governance(session, run_id, step_id, effective_seed_bundle, result)
            session.flush()
            self._persist_algorithms(session, run_id, step_id, algorithm_decisions)
            self._persist_trace_evidence_metrics(session, run_id, step_id, result)
            if embedding_result is not None:
                self._persist_embedding(session, run_id, step_id, result, embedding_result)

    def readback_summary(self, *, run_id: str) -> dict[str, object]:
        with self._session() as session:
            provider_ids = tuple(
                session.scalars(
                    select(models.ProviderCallRecord.id).where(
                        models.ProviderCallRecord.run_id == run_id
                    )
                )
            )
            verification_ids = tuple(
                session.scalars(
                    select(models.VerificationResultRecord.id).where(
                        models.VerificationResultRecord.run_id == run_id
                    )
                )
            )
            state_diff_ids = tuple(
                session.scalars(
                    select(models.StateDiffRecord.id).where(models.StateDiffRecord.run_id == run_id)
                )
            )
            algorithm_ids = tuple(
                session.scalars(
                    select(models.AlgorithmDecisionRecord.id).where(
                        models.AlgorithmDecisionRecord.run_id == run_id
                    )
                )
            )
            mechanism_ids = tuple(
                session.scalars(
                    select(models.AlgorithmDecisionRecord.mechanism_id)
                    .where(models.AlgorithmDecisionRecord.run_id == run_id)
                    .order_by(models.AlgorithmDecisionRecord.mechanism_id)
                )
            )
            embedding_ids = tuple(
                session.scalars(
                    select(models.EmbeddingRecord.embedding_id).where(
                        models.EmbeddingRecord.run_id == run_id
                    )
                )
            )
            counts = {
                "run": session.scalar(
                    select(func.count())
                    .select_from(models.RunRecord)
                    .where(models.RunRecord.id == run_id)
                ),
                "run_step": _count_run(session, models.RunStepRecord, run_id),
                "provider_call_record": _count_run(session, models.ProviderCallRecord, run_id),
                "llm_input_record": _count_in(
                    session, models.LLMInputRecord.provider_call_id, provider_ids
                ),
                "llm_output_record": _count_in(
                    session, models.LLMOutputRecord.provider_call_id, provider_ids
                ),
                "structured_output_validation": _count_in(
                    session, models.StructuredOutputValidationRecord.provider_call_id, provider_ids
                ),
                "action_proposal": _count_run(session, models.ActionProposalRecord, run_id),
                "event_candidate": _count_run(session, models.EventCandidateRecord, run_id),
                "verification_result": len(verification_ids),
                "verifier_check_result": _count_in(
                    session,
                    models.VerifierCheckResultRecord.verification_result_id,
                    verification_ids,
                ),
                "committed_event": _count_run(session, models.CommittedEventRecord, run_id),
                "state_diff": len(state_diff_ids),
                "state_patch": _count_in(
                    session, models.StatePatchRecord.state_diff_id, state_diff_ids
                ),
                "world_state_snapshot": _count_run(
                    session, models.WorldStateSnapshotRecord, run_id
                ),
                "algorithm_decision": len(algorithm_ids),
                "algorithm_score_breakdown": _count_in(
                    session,
                    models.AlgorithmScoreBreakdownRecord.algorithm_decision_id,
                    algorithm_ids,
                ),
                "trace_event": _count_run(session, models.TraceEventRecord, run_id),
                "evidence_artifact": _count_run(session, models.EvidenceArtifactRecord, run_id),
                "metric_result": _count_run(session, models.MetricResultRecord, run_id),
                "embedding_record": len(embedding_ids),
                "embedding_chunk": _count_in(
                    session, models.EmbeddingChunkRecord.embedding_id, embedding_ids
                ),
            }
            run = session.get(models.RunRecord, run_id)
            step = session.scalar(
                select(models.RunStepRecord).where(models.RunStepRecord.run_id == run_id)
            )
            embedding = _embedding_readback(session, embedding_ids)
        required = (
            "run",
            "run_step",
            "provider_call_record",
            "llm_input_record",
            "llm_output_record",
            "structured_output_validation",
            "action_proposal",
            "event_candidate",
            "verification_result",
            "verifier_check_result",
            "committed_event",
            "state_diff",
            "state_patch",
            "world_state_snapshot",
            "algorithm_decision",
            "algorithm_score_breakdown",
            "trace_event",
            "evidence_artifact",
            "metric_result",
            "embedding_record",
            "embedding_chunk",
        )
        expected_mechanisms = {kind.value for kind in MechanismKind}
        observed_mechanisms = set(mechanism_ids)
        mechanism_coverage_passed = observed_mechanisms == expected_mechanisms
        return {
            "run_id": run_id,
            "counts": counts,
            "mechanism_ids": mechanism_ids,
            "mechanism_coverage_count": len(observed_mechanisms),
            "mechanism_coverage_expected": len(expected_mechanisms),
            "mechanism_coverage_passed": mechanism_coverage_passed,
            "provider_called": bool(run.provider_called) if run else False,
            "fallback_used": bool(run.fallback_used) if run else True,
            "structured_validation_passed": (
                bool(step.structured_validation_passed) if step else False
            ),
            "embedding_db_readback": embedding,
            "embedding_db_readback_passed": bool(embedding.get("passed")),
            "db_readback_passed": all((counts.get(name) or 0) > 0 for name in required)
            and bool(embedding.get("passed"))
            and mechanism_coverage_passed,
        }

    def long_horizon_readback_summary(
        self,
        *,
        run_id: str,
        expected_steps: int,
    ) -> dict[str, object]:
        base = self.readback_summary(run_id=run_id)
        with self._session() as session:
            steps = tuple(
                session.execute(
                    select(
                        models.RunStepRecord.id,
                        models.RunStepRecord.step_index,
                        models.RunStepRecord.provider_called,
                        models.RunStepRecord.fallback_used,
                        models.RunStepRecord.structured_validation_passed,
                        models.RunStepRecord.state_diff_applied,
                    )
                    .where(models.RunStepRecord.run_id == run_id)
                    .order_by(models.RunStepRecord.step_index)
                ).mappings()
            )
            per_step = tuple(_per_step_readback(session, step) for step in steps)
        return base | {
            "expected_step_count": expected_steps,
            "completed_step_count": len(steps),
            "per_step": per_step,
            "long_horizon_db_readback_passed": (
                bool(base["db_readback_passed"])
                and len(steps) == expected_steps
                and all(step["per_step_readback_passed"] for step in per_step)
            ),
        }

    def _persist_provider(
        self,
        session: Session,
        run_id: str,
        step_id: str,
        result: SingleStepResult,
        seed_bundle: SeedBundle,
    ) -> None:
        structured = result.structured_output
        assert structured is not None
        provider_call_id = f"provider:{step_id}"
        prompt_hash = sha256(
            build_step_context(
                seed_bundle,
                agent_id=result.agent_id,
                scenario_id=result.scenario_id,
            ).prompt.encode("utf-8")
        ).hexdigest()
        session.add(
            models.ProviderCallRecord(
                id=provider_call_id,
                run_id=run_id,
                step_id=step_id,
                provider_name=structured.provider_name,
                model_name=structured.model_name,
                latency_ms=structured.latency_ms,
                success=structured.success,
                fallback_used=result.fallback_used,
                usage_json=structured.usage,
                attempts_json={
                    "attempts": [attempt.safe_dict() for attempt in structured.attempts]
                },
            )
        )
        session.flush()
        session.add(
            models.LLMInputRecord(
                id=f"llm-input:{step_id}",
                provider_call_id=provider_call_id,
                prompt_sha256=prompt_hash,
                redaction_status="hash_only_not_raw_saved",
                schema_name="ActionProposal",
            )
        )
        session.add(
            models.LLMOutputRecord(
                id=f"llm-output:{step_id}",
                provider_call_id=provider_call_id,
                raw_text_sha256=structured.raw_text_sha256,
                raw_output_saved=False,
                redaction_status="hash_only_validated_structured_output_saved",
                structured_json=(
                    structured.data.model_dump(mode="json") if structured.data is not None else {}
                ),
            )
        )
        session.add(
            models.StructuredOutputValidationRecord(
                id=f"structured-validation:{step_id}",
                provider_call_id=provider_call_id,
                schema_name="ActionProposal",
                passed=structured.success,
                json_parse_error=redact_text(structured.json_parse_error or "")
                if structured.json_parse_error
                else None,
                validation_error=redact_text(structured.validation_error or "")
                if structured.validation_error
                else None,
            )
        )

    def _persist_governance(
        self,
        session: Session,
        run_id: str,
        step_id: str,
        seed_bundle: SeedBundle,
        result: SingleStepResult,
    ) -> None:
        proposal = result.action_proposal
        candidate = result.event_candidate
        verification = result.verification_result
        committed = result.committed_event
        assert proposal is not None and candidate is not None
        assert verification is not None and committed is not None
        proposal_db_id = _db_row_id(step_id, proposal.id)
        candidate_db_id = _db_row_id(step_id, candidate.id)
        verification_db_id = _db_row_id(step_id, verification.id)
        committed_db_id = _db_row_id(step_id, committed.id)
        state_diff_db_id = _db_row_id(step_id, committed.state_diff.id)
        session.add(
            models.ActionProposalRecord(
                id=proposal_db_id,
                run_id=run_id,
                step_id=step_id,
                proposal_json=proposal.model_dump(mode="json"),
            )
        )
        session.flush()
        session.add(
            models.EventCandidateRecord(
                id=candidate_db_id,
                run_id=run_id,
                step_id=step_id,
                action_proposal_id=proposal_db_id,
                candidate_json=candidate.model_dump(mode="json"),
            )
        )
        session.flush()
        session.add(
            models.VerificationResultRecord(
                id=verification_db_id,
                run_id=run_id,
                step_id=step_id,
                event_candidate_id=candidate_db_id,
                decision=verification.decision.value,
                verification_json=verification.model_dump(mode="json"),
            )
        )
        session.flush()
        for index, check in enumerate(verification.checks):
            session.add(
                models.VerifierCheckResultRecord(
                    id=f"verifier-check:{step_id}:{index}",
                    verification_result_id=verification_db_id,
                    name=check.name,
                    passed=check.passed,
                    message=check.message,
                )
            )
        session.add(
            models.CommittedEventRecord(
                id=committed_db_id,
                run_id=run_id,
                step_id=step_id,
                event_candidate_id=candidate_db_id,
                verification_result_id=verification_db_id,
                committed_event_json=committed.model_dump(mode="json"),
            )
        )
        session.flush()
        session.add(
            models.StateDiffRecord(
                id=state_diff_db_id,
                run_id=run_id,
                step_id=step_id,
                committed_event_id=committed_db_id,
                event_candidate_id=candidate_db_id,
                state_diff_json=committed.state_diff.model_dump(mode="json"),
            )
        )
        session.flush()
        for index, patch in enumerate(committed.state_diff.patches):
            session.add(
                models.StatePatchRecord(
                    id=f"state-patch:{step_id}:{index}",
                    state_diff_id=state_diff_db_id,
                    patch_index=index,
                    operation=patch.operation.value,
                    target_type=patch.target_type.value,
                    target_id=patch.target_id,
                    path=patch.path,
                    patch_json=patch.model_dump(mode="json"),
                )
            )
        session.add_all(
            (
                models.WorldStateSnapshotRecord(
                    id=f"snapshot:{step_id}:before",
                    run_id=run_id,
                    step_id=step_id,
                    snapshot_kind="before",
                    world_state_json=seed_bundle.world.model_dump(mode="json"),
                ),
                models.WorldStateSnapshotRecord(
                    id=f"snapshot:{step_id}:after",
                    run_id=run_id,
                    step_id=step_id,
                    snapshot_kind="after",
                    world_state_json=result.applied_world_state.model_dump(mode="json"),
                ),
            )
        )

    def _persist_algorithms(
        self,
        session: Session,
        run_id: str,
        step_id: str,
        decisions: Iterable[RuntimeAlgorithmDecision],
    ) -> None:
        for index, decision in enumerate(decisions):
            decision_id = f"algorithm:{step_id}:{index}:{decision.mechanism_id}"
            session.add(
                models.AlgorithmDecisionRecord(
                    id=decision_id,
                    run_id=run_id,
                    step_id=step_id,
                    mechanism_id=decision.mechanism_id,
                    model_family=decision.model_family,
                    formula=decision.formula,
                    score=decision.score,
                    decision=decision.decision,
                    runtime_object_type=decision.runtime_object_type,
                    runtime_object_id=decision.runtime_object_id,
                    input_features_json=decision.input_features,
                )
            )
            session.flush()
            for score_name, value in decision.score_breakdown.items():
                numeric = value if isinstance(value, int | float) else None
                session.add(
                    models.AlgorithmScoreBreakdownRecord(
                        id=f"algorithm-score:{step_id}:{index}:{score_name}",
                        algorithm_decision_id=decision_id,
                        score_name=score_name,
                        score_value=float(numeric) if numeric is not None else None,
                        detail_json={"value": value},
                    )
                )

    def _persist_trace_evidence_metrics(
        self,
        session: Session,
        run_id: str,
        step_id: str,
        result: SingleStepResult,
    ) -> None:
        summary = result.safe_summary()
        session.add(
            models.TraceEventRecord(
                id=f"trace:{step_id}:governed-chain",
                run_id=run_id,
                step_id=step_id,
                event_type="governed_runtime_step",
                event_json=summary,
            )
        )

    def _persist_embedding(
        self,
        session: Session,
        run_id: str,
        step_id: str,
        result: SingleStepResult,
        embedding: EmbeddingResult,
    ) -> None:
        if embedding.dimensions != 1024:
            raise ValueError(
                f"Embedding dimension mismatch for vector(1024): {embedding.dimensions}"
            )
        provider_call_id = f"embedding-provider:{step_id}"
        embedding_id = f"embedding:{step_id}:governed-summary"
        chunk_id = f"embedding-chunk:{step_id}:0"
        source_text = _embedding_source_text(result)
        content_hash = sha256(source_text.encode("utf-8")).hexdigest()
        vector_norm = _vector_norm(embedding.embedding)
        session.add(
            models.ProviderCallRecord(
                id=provider_call_id,
                run_id=run_id,
                step_id=step_id,
                provider_name="volcengine_ark",
                model_name=embedding.model,
                latency_ms=embedding.latency_ms,
                success=True,
                fallback_used=False,
                usage_json=embedding.usage,
                attempts_json={
                    "attempts": [
                        {
                            "model": embedding.model,
                            "success": True,
                            "latency_ms": embedding.latency_ms,
                        }
                    ]
                },
            )
        )
        session.flush()
        session.add(
            models.EmbeddingRecord(
                embedding_id=embedding_id,
                run_id=run_id,
                step_id=step_id,
                provider_call_id=provider_call_id,
                source_type="governed_step_evidence_summary",
                source_object_id=step_id,
                embedding_model=embedding.model,
                dimensions=embedding.dimensions,
                vector_norm=vector_norm,
                content_hash=content_hash,
                redaction_status="safe_summary_hash_source",
                provider_called=True,
                fallback_used=False,
                metadata_json={
                    "source_objects": [
                        result.action_proposal.id if result.action_proposal else None,
                        result.event_candidate.id if result.event_candidate else None,
                        result.verification_result.id if result.verification_result else None,
                        result.committed_event.id if result.committed_event else None,
                    ]
                },
            )
        )
        session.flush()
        session.execute(
            text(
                """
                insert into embedding_chunks (
                    chunk_id, embedding_id, chunk_index, source_type, source_object_id,
                    content_hash, dimensions, embedding_vector, vector_norm, metadata_json
                )
                values (
                    :chunk_id, :embedding_id, 0, :source_type, :source_object_id,
                    :content_hash, :dimensions, cast(:embedding_vector as vector),
                    :vector_norm, cast(:metadata_json as jsonb)
                )
                """
            ),
            {
                "chunk_id": chunk_id,
                "embedding_id": embedding_id,
                "source_type": "governed_step_evidence_summary",
                "source_object_id": step_id,
                "content_hash": content_hash,
                "dimensions": embedding.dimensions,
                "embedding_vector": _vector_literal(embedding.embedding),
                "vector_norm": vector_norm,
                "metadata_json": json.dumps(
                    {"raw_source_text_saved": False}, separators=(",", ":")
                ),
            },
        )
        session.add(
            models.EvidenceArtifactRecord(
                id=f"evidence:{step_id}:embedding",
                run_id=run_id,
                step_id=step_id,
                artifact_type="real_embedding_pgvector_evidence",
                redaction_status="source_text_not_saved_vector_in_pgvector",
                artifact_json={
                    "embedding_provider_called": True,
                    "embedding_fallback_used": False,
                    "embedding_model": embedding.model,
                    "embedding_dimensions": embedding.dimensions,
                    "embedding_vector_persisted": True,
                    "embedding_id": embedding_id,
                    "content_hash": content_hash,
                    "vector_norm": vector_norm,
                },
            )
        )
        session.add(
            models.EvidenceArtifactRecord(
                id=f"evidence:{step_id}:real-provider",
                run_id=run_id,
                step_id=step_id,
                artifact_type="real_provider_governed_step",
                redaction_status="raw_prompt_and_raw_output_not_saved",
                artifact_json={
                    "provider_called": result.provider_called,
                    "fallback_used": result.fallback_used,
                    "evidence_class": result.evidence_class,
                    "raw_text_sha256": (
                        result.structured_output.raw_text_sha256
                        if result.structured_output is not None
                        else None
                    ),
                    "state_diff_applied": result.state_diff_applied,
                },
            )
        )
        session.add(
            models.MetricResultRecord(
                id=f"metric:{step_id}:governed_chain_complete",
                run_id=run_id,
                step_id=step_id,
                metric_name="governed_chain_complete",
                metric_value=1.0,
                metric_json={
                    "action_proposal": result.action_proposal is not None,
                    "event_candidate": result.event_candidate is not None,
                    "verification_result": result.verification_result is not None,
                    "committed_event": result.committed_event is not None,
                    "state_diff": (
                        result.committed_event.state_diff.id
                        if result.committed_event is not None
                        else None
                    ),
                    "state_diff_applied": result.state_diff_applied,
                },
            )
        )


def _count_run(session: Session, table, run_id: str) -> int:
    return int(
        session.scalar(select(func.count()).select_from(table).where(table.run_id == run_id))
    )


def _count_in(session: Session, column, values: tuple[str, ...]) -> int:
    if not values:
        return 0
    return int(session.scalar(select(func.count()).where(column.in_(values))))


def _per_step_readback(session: Session, step) -> dict[str, object]:
    step_id = str(step["id"])
    provider_ids = tuple(
        session.scalars(
            select(models.ProviderCallRecord.id).where(models.ProviderCallRecord.step_id == step_id)
        )
    )
    verification_ids = tuple(
        session.scalars(
            select(models.VerificationResultRecord.id).where(
                models.VerificationResultRecord.step_id == step_id
            )
        )
    )
    state_diff_ids = tuple(
        session.scalars(
            select(models.StateDiffRecord.id).where(models.StateDiffRecord.step_id == step_id)
        )
    )
    algorithm_ids = tuple(
        session.scalars(
            select(models.AlgorithmDecisionRecord.id).where(
                models.AlgorithmDecisionRecord.step_id == step_id
            )
        )
    )
    embedding_ids = tuple(
        session.scalars(
            select(models.EmbeddingRecord.embedding_id).where(
                models.EmbeddingRecord.step_id == step_id
            )
        )
    )
    counts = {
        "provider_call_record": len(provider_ids),
        "llm_input_record": _count_in(session, models.LLMInputRecord.provider_call_id, provider_ids),
        "llm_output_record": _count_in(
            session, models.LLMOutputRecord.provider_call_id, provider_ids
        ),
        "structured_output_validation": _count_in(
            session, models.StructuredOutputValidationRecord.provider_call_id, provider_ids
        ),
        "action_proposal": _count_step(session, models.ActionProposalRecord, step_id),
        "event_candidate": _count_step(session, models.EventCandidateRecord, step_id),
        "verification_result": len(verification_ids),
        "verifier_check_result": _count_in(
            session,
            models.VerifierCheckResultRecord.verification_result_id,
            verification_ids,
        ),
        "committed_event": _count_step(session, models.CommittedEventRecord, step_id),
        "state_diff": len(state_diff_ids),
        "state_patch": _count_in(session, models.StatePatchRecord.state_diff_id, state_diff_ids),
        "world_state_snapshot": _count_step(session, models.WorldStateSnapshotRecord, step_id),
        "algorithm_decision": len(algorithm_ids),
        "algorithm_score_breakdown": _count_in(
            session,
            models.AlgorithmScoreBreakdownRecord.algorithm_decision_id,
            algorithm_ids,
        ),
        "trace_event": _count_step(session, models.TraceEventRecord, step_id),
        "evidence_artifact": _count_step(session, models.EvidenceArtifactRecord, step_id),
        "metric_result": _count_step(session, models.MetricResultRecord, step_id),
        "embedding_record": len(embedding_ids),
        "embedding_chunk": _count_in(
            session,
            models.EmbeddingChunkRecord.embedding_id,
            embedding_ids,
        ),
    }
    required = (
        "provider_call_record",
        "llm_input_record",
        "llm_output_record",
        "structured_output_validation",
        "action_proposal",
        "event_candidate",
        "verification_result",
        "verifier_check_result",
        "committed_event",
        "state_diff",
        "state_patch",
        "world_state_snapshot",
        "algorithm_decision",
        "algorithm_score_breakdown",
        "trace_event",
        "evidence_artifact",
        "metric_result",
        "embedding_record",
        "embedding_chunk",
    )
    return {
        "step_id": step_id,
        "step_index": int(step["step_index"]),
        "provider_called": bool(step["provider_called"]),
        "fallback_used": bool(step["fallback_used"]),
        "structured_validation_passed": bool(step["structured_validation_passed"]),
        "state_diff_applied": bool(step["state_diff_applied"]),
        "counts": counts,
        "per_step_readback_passed": (
            bool(step["provider_called"])
            and not bool(step["fallback_used"])
            and bool(step["structured_validation_passed"])
            and bool(step["state_diff_applied"])
            and all((counts.get(name) or 0) > 0 for name in required)
            and counts["world_state_snapshot"] >= 2
        ),
    }


def _count_step(session: Session, table, step_id: str) -> int:
    return int(
        session.scalar(select(func.count()).select_from(table).where(table.step_id == step_id))
    )


def _db_row_id(step_id: str, domain_id: str) -> str:
    return f"{step_id}:{domain_id}"


def _embedding_readback(session: Session, embedding_ids: tuple[str, ...]) -> dict[str, object]:
    if not embedding_ids:
        return {"passed": False}
    row = (
        session.execute(
            text(
                """
            select r.embedding_model, r.dimensions, r.vector_norm,
                   c.embedding_vector::text as vector_text,
                   c.vector_norm as chunk_norm
            from embedding_records r
            join embedding_chunks c on c.embedding_id = r.embedding_id
            where r.embedding_id = :embedding_id
            limit 1
            """
            ),
            {"embedding_id": embedding_ids[0]},
        )
        .mappings()
        .first()
    )
    if row is None:
        return {"passed": False}
    return {
        "passed": str(row["vector_text"]).startswith("[") and float(row["chunk_norm"]) > 0,
        "embedding_model": row["embedding_model"],
        "dimensions": row["dimensions"],
        "vector_norm": row["vector_norm"],
    }


def _embedding_source_text(result: SingleStepResult) -> str:
    summary = result.safe_summary()
    safe_payload = {
        "scenario_id": summary["scenario_id"],
        "agent_id": summary["agent_id"],
        "action_proposal_summary": summary["action_proposal_summary"],
        "event_candidate_summary": summary["event_candidate_summary"],
        "verification": summary["verification"],
        "state_diff": summary["state_diff"],
        "state_diff_applied": summary["state_diff_applied"],
    }
    return json.dumps(safe_payload, ensure_ascii=False, sort_keys=True)


def _vector_norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _vector_literal(vector: tuple[float, ...]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"
