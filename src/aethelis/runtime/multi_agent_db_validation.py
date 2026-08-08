from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from pydantic import Field
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

import aethelis.db.models as models
from aethelis.agents.action_proposal import ActionProposalEngine, ActionProposalGenerationResult
from aethelis.config.settings import Settings
from aethelis.llm.openai_compatible import OpenAICompatibleLLMProvider
from aethelis.runtime.multi_agent_step import MultiAgentWorldStepResult, run_multi_agent_world_step
from aethelis.runtime.single_step import build_step_context
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.seed import SeedBundle
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator
from aethelis.utils.redaction import redact_text

MULTI_AGENT_B5_MODE = "r5_b5_multi_agent_real_provider_db"


class MultiAgentDBValidationResult(AethelisModel):
    run_id: Identifier
    step_id: Identifier
    seed_id: Identifier
    scenario_id: Identifier
    active_agent_ids: tuple[Identifier, ...]
    provider_called: bool
    fallback_used: bool
    structured_validation_passed: bool
    db_written: bool
    db_readback_passed: bool
    evidence_comparison_passed: bool
    state_diff_applied: bool
    provider_call_count: int = Field(ge=0)
    action_proposal_ids: tuple[Identifier, ...] = ()
    event_candidate_ids: tuple[Identifier, ...] = ()
    verification_result_ids: tuple[Identifier, ...] = ()
    committed_event_ids: tuple[Identifier, ...] = ()
    state_diff_ids: tuple[Identifier, ...] = ()
    route_summary: tuple[dict[str, object], ...] = ()
    unsupported_db_fields: tuple[str, ...] = ()
    readback: dict[str, object]

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def run_multi_agent_real_provider_db_validation(
    *,
    engine: Engine,
    seed_path: Path,
    scenario_id: str,
    active_agent_ids: tuple[str, ...],
    settings: Settings,
    apply: bool = True,
    run_id: str | None = None,
    context_budget_per_agent: int = 12,
) -> MultiAgentDBValidationResult:
    if len(active_agent_ids) < 2:
        raise ValueError("R5-B5 multi-agent validation requires at least two active agents.")

    bundle = _load_valid_seed(seed_path)
    run_id = run_id or f"r5_b5_multi_agent_db_run_{uuid4().hex}"
    step_id = f"{run_id}_step_1"
    proposals, prompt_hashes = _generate_provider_proposals(
        bundle=bundle,
        scenario_id=scenario_id,
        active_agent_ids=active_agent_ids,
        settings=settings,
    )
    result = run_multi_agent_world_step(
        seed_path=seed_path,
        step_id=step_id,
        scenario_id=scenario_id,
        active_agent_ids=active_agent_ids,
        apply=apply,
        context_budget_per_agent=context_budget_per_agent,
        proposal_results=proposals,
    )
    _assert_b5_runtime_result(result=result, generated=proposals)
    _persist_multi_agent_result(
        engine=engine,
        run_id=run_id,
        seed_id=seed_path.name,
        seed_bundle=bundle,
        result=result,
        generated=proposals,
        prompt_hashes=prompt_hashes,
    )
    readback = _readback_multi_agent_result(
        engine=engine,
        run_id=run_id,
        step_id=step_id,
        result=result,
        active_agent_ids=active_agent_ids,
        prompt_hashes=prompt_hashes,
    )
    return MultiAgentDBValidationResult(
        run_id=run_id,
        step_id=step_id,
        seed_id=seed_path.name,
        scenario_id=scenario_id,
        active_agent_ids=active_agent_ids,
        provider_called=result.provider_called,
        fallback_used=False,
        structured_validation_passed=all(
            generated.structured_output is not None and generated.structured_output.success
            for generated in proposals.values()
        ),
        db_written=True,
        db_readback_passed=bool(readback["db_readback_passed"]),
        evidence_comparison_passed=bool(readback["evidence_comparison_passed"]),
        state_diff_applied=result.state_diff_applied,
        provider_call_count=len(proposals),
        action_proposal_ids=tuple(proposal.id for proposal in result.action_proposals),
        event_candidate_ids=tuple(candidate.id for candidate in result.event_candidates),
        verification_result_ids=tuple(
            verification.id for verification in result.verification_results
        ),
        committed_event_ids=tuple(event.id for event in result.committed_events),
        state_diff_ids=tuple(diff.id for diff in result.state_diffs),
        route_summary=tuple(route.model_dump(mode="json") for route in result.routes),
        unsupported_db_fields=(
            "no dedicated multi_agent_context table; "
            "context summaries stored in evidence_artifacts",
            "no dedicated arbitration table; recommendation summary stored in evidence_artifacts",
            "no dedicated route table; "
            "route records stored in run step metadata and evidence_artifacts",
        ),
        readback=readback,
    )


def _generate_provider_proposals(
    *,
    bundle: SeedBundle,
    scenario_id: str,
    active_agent_ids: tuple[str, ...],
    settings: Settings,
) -> tuple[dict[str, ActionProposalGenerationResult], dict[str, str]]:
    engine = ActionProposalEngine()
    generated: dict[str, ActionProposalGenerationResult] = {}
    prompt_hashes: dict[str, str] = {}
    with OpenAICompatibleLLMProvider(settings) as provider:
        for agent_id in active_agent_ids:
            context = build_step_context(bundle, agent_id=agent_id, scenario_id=scenario_id)
            prompt_hashes[agent_id] = sha256(context.prompt.encode("utf-8")).hexdigest()
            result = engine.generate_structured(
                provider=provider,
                prompt=context.prompt,
                max_tokens=260,
                temperature=0.0,
            )
            if not result.provider_called:
                raise RuntimeError(f"Provider was not called for active agent {agent_id}.")
            if result.structured_output is None or not result.structured_output.success:
                detail = (
                    result.structured_output.json_parse_error
                    if result.structured_output is not None
                    else result.error
                )
                raise RuntimeError(
                    f"Structured ActionProposal validation failed for {agent_id}: "
                    f"{redact_text(detail or 'unknown_error')}"
                )
            if result.proposal is None:
                raise RuntimeError(f"Provider returned no ActionProposal for {agent_id}.")
            if result.proposal.proposer_agent_id != agent_id:
                raise ValueError(
                    "Provider proposal owner mismatch: "
                    f"{result.proposal.proposer_agent_id} returned for active agent {agent_id}."
                )
            generated[agent_id] = result
    return generated, prompt_hashes


def _assert_b5_runtime_result(
    *,
    result: MultiAgentWorldStepResult,
    generated: dict[str, ActionProposalGenerationResult],
) -> None:
    if not result.provider_called:
        raise RuntimeError("MultiAgentWorldStep did not preserve provider_called=true.")
    if not all(item.provider_called for item in generated.values()):
        raise RuntimeError("At least one active agent did not call the real provider.")
    if not result.event_candidates or not result.verification_results:
        raise RuntimeError("No routed proposal reached EventCandidate/VerificationResult.")
    if not any(
        route.event_candidate_id is None
        and route.route in {"blocked_by_hard_conflict", "requires_revision"}
        for route in result.routes
    ) and not result.joint_intent_candidates:
        raise RuntimeError("No non-routed proposal or joint-intent evidence was recorded.")
    if any(
        route.event_candidate_id for route in result.routes if route.route == "requires_revision"
    ):
        raise RuntimeError("requires_revision route produced governance objects.")
    if any(
        route.event_candidate_id
        for route in result.routes
        if route.route == "blocked_by_hard_conflict"
    ):
        raise RuntimeError("blocked_by_hard_conflict route produced governance objects.")
    if not result.verifier_retrieval_boundaries:
        raise RuntimeError("Verifier retrieval boundaries were not recorded.")
    if any(
        not boundary.matches_proposal_context
        for boundary in result.verifier_retrieval_boundaries
    ):
        raise RuntimeError("Verifier retrieval boundary did not match B2 context attribution.")


def _persist_multi_agent_result(
    *,
    engine: Engine,
    run_id: str,
    seed_id: str,
    seed_bundle: SeedBundle,
    result: MultiAgentWorldStepResult,
    generated: dict[str, ActionProposalGenerationResult],
    prompt_hashes: dict[str, str],
) -> None:
    session_factory = sessionmaker(engine, expire_on_commit=False)
    with session_factory.begin() as session:
        session.merge(
            models.RunRecord(
                id=run_id,
                seed_id=seed_id,
                status="completed",
                algorithm_mode=MULTI_AGENT_B5_MODE,
                provider_called=True,
                fallback_used=False,
                db_persisted=True,
                metadata_json={
                    "evidence_class": "real_provider_db_backed_multi_agent_world_step",
                    "phase": "R5-B5",
                    "active_agent_ids": list(result.active_agent_ids),
                },
            )
        )
        session.add(
            models.RunStepRecord(
                id=result.step_id,
                run_id=run_id,
                step_index=1,
                scenario_id=result.scenario_id,
                agent_id="multi_agent",
                status="completed",
                provider_called=True,
                fallback_used=False,
                structured_validation_passed=True,
                state_diff_applied=result.state_diff_applied,
                metadata_json={
                    "active_agent_ids": list(result.active_agent_ids),
                    "routed_proposal_ids": list(result.routed_proposal_ids),
                    "blocked_proposal_ids": list(result.blocked_proposal_ids),
                    "revision_required_proposal_ids": list(result.revision_required_proposal_ids),
                    "independent_proposal_ids": list(result.independent_proposal_ids),
                    "joint_intent_candidate_ids": [
                        candidate.id for candidate in result.joint_intent_candidates
                    ],
                },
            )
        )
        session.flush()
        for agent_id, generated_result in generated.items():
            _persist_provider_agent(
                session=session,
                run_id=run_id,
                step_id=result.step_id,
                agent_id=agent_id,
                generated=generated_result,
                prompt_hash=prompt_hashes[agent_id],
            )
        _persist_governance(session=session, run_id=run_id, seed_bundle=seed_bundle, result=result)
        _persist_multi_agent_trace_and_artifacts(session=session, run_id=run_id, result=result)


def _persist_provider_agent(
    *,
    session,
    run_id: str,
    step_id: str,
    agent_id: str,
    generated: ActionProposalGenerationResult,
    prompt_hash: str,
) -> None:
    structured = generated.structured_output
    assert structured is not None
    provider_call_id = f"provider:{step_id}:{agent_id}"
    session.add(
        models.ProviderCallRecord(
            id=provider_call_id,
            run_id=run_id,
            step_id=step_id,
            provider_name=structured.provider_name,
            model_name=structured.model_name,
            latency_ms=structured.latency_ms,
            success=structured.success,
            fallback_used=False,
            usage_json=structured.usage,
            attempts_json={"attempts": [attempt.safe_dict() for attempt in structured.attempts]},
        )
    )
    session.flush()
    session.add(
        models.LLMInputRecord(
            id=f"llm-input:{step_id}:{agent_id}",
            provider_call_id=provider_call_id,
            prompt_sha256=prompt_hash,
            redaction_status="hash_only_not_raw_saved",
            schema_name="ActionProposal",
        )
    )
    session.add(
        models.LLMOutputRecord(
            id=f"llm-output:{step_id}:{agent_id}",
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
            id=f"structured-validation:{step_id}:{agent_id}",
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
    *,
    session,
    run_id: str,
    seed_bundle: SeedBundle,
    result: MultiAgentWorldStepResult,
) -> None:
    proposal_db_ids: dict[str, str] = {}
    candidate_db_ids: dict[str, str] = {}
    verification_db_ids: dict[str, str] = {}
    for proposal in result.action_proposals:
        proposal_db_id = _db_row_id(result.step_id, proposal.id)
        proposal_db_ids[proposal.id] = proposal_db_id
        session.add(
            models.ActionProposalRecord(
                id=proposal_db_id,
                run_id=run_id,
                step_id=result.step_id,
                proposal_json=proposal.model_dump(mode="json"),
            )
        )
    session.flush()
    for candidate in result.event_candidates:
        candidate_db_id = _db_row_id(result.step_id, candidate.id)
        candidate_db_ids[candidate.id] = candidate_db_id
        session.add(
            models.EventCandidateRecord(
                id=candidate_db_id,
                run_id=run_id,
                step_id=result.step_id,
                action_proposal_id=proposal_db_ids.get(candidate.source_action_proposal_id),
                candidate_json=candidate.model_dump(mode="json"),
            )
        )
    session.flush()
    for verification in result.verification_results:
        verification_db_id = _db_row_id(result.step_id, verification.id)
        verification_db_ids[verification.id] = verification_db_id
        session.add(
            models.VerificationResultRecord(
                id=verification_db_id,
                run_id=run_id,
                step_id=result.step_id,
                event_candidate_id=candidate_db_ids[verification.event_candidate_id],
                decision=verification.decision.value,
                verification_json=verification.model_dump(mode="json"),
            )
        )
        session.flush()
        for index, check in enumerate(verification.checks):
            session.add(
                models.VerifierCheckResultRecord(
                    id=f"verifier-check:{result.step_id}:{verification.id}:{index}",
                    verification_result_id=verification_db_id,
                    name=check.name,
                    passed=check.passed,
                    message=check.message,
                )
            )
    session.flush()
    for committed in result.committed_events:
        committed_db_id = _db_row_id(result.step_id, committed.id)
        state_diff_db_id = _db_row_id(result.step_id, committed.state_diff.id)
        session.add(
            models.CommittedEventRecord(
                id=committed_db_id,
                run_id=run_id,
                step_id=result.step_id,
                event_candidate_id=candidate_db_ids[committed.event_candidate_id],
                verification_result_id=verification_db_ids[committed.verification_result_id],
                committed_event_json=committed.model_dump(mode="json"),
            )
        )
        session.flush()
        session.add(
            models.StateDiffRecord(
                id=state_diff_db_id,
                run_id=run_id,
                step_id=result.step_id,
                committed_event_id=committed_db_id,
                event_candidate_id=candidate_db_ids[committed.event_candidate_id],
                state_diff_json=committed.state_diff.model_dump(mode="json"),
            )
        )
        session.flush()
        for index, patch in enumerate(committed.state_diff.patches):
            session.add(
                models.StatePatchRecord(
                    id=f"state-patch:{result.step_id}:{committed.state_diff.id}:{index}",
                    state_diff_id=state_diff_db_id,
                    patch_index=index,
                    operation=patch.operation.value,
                    target_type=patch.target_type.value,
                    target_id=patch.target_id,
                    path=patch.path,
                    patch_json=patch.model_dump(mode="json"),
                )
            )
    snapshots = [
        models.WorldStateSnapshotRecord(
            id=f"snapshot:{result.step_id}:before",
            run_id=run_id,
            step_id=result.step_id,
            snapshot_kind="before",
            world_state_json=seed_bundle.world.model_dump(mode="json"),
        )
    ]
    if result.applied_world_state is not None:
        snapshots.append(
            models.WorldStateSnapshotRecord(
                id=f"snapshot:{result.step_id}:after",
                run_id=run_id,
                step_id=result.step_id,
                snapshot_kind="after",
                world_state_json=result.applied_world_state.model_dump(mode="json"),
            )
        )
    session.add_all(snapshots)


def _persist_multi_agent_trace_and_artifacts(
    *,
    session,
    run_id: str,
    result: MultiAgentWorldStepResult,
) -> None:
    summary = result.safe_summary()
    session.add(
        models.TraceEventRecord(
            id=f"trace:{result.step_id}:multi-agent-world-step",
            run_id=run_id,
            step_id=result.step_id,
            event_type="multi_agent_world_step_real_provider_db",
            event_json=summary,
        )
    )
    artifacts = {
        "multi_agent_context": result.context.safe_summary(),
        "multi_agent_proposal_bundle": result.proposal_bundle.model_dump(mode="json"),
        "multi_agent_dynamics": result.dynamics_summary.model_dump(mode="json"),
        "multi_agent_arbitration": result.arbitration_recommendation.model_dump(mode="json"),
        "multi_agent_routes": [route.model_dump(mode="json") for route in result.routes],
        "multi_agent_safe_summary": summary,
    }
    for artifact_type, artifact_json in artifacts.items():
        session.add(
            models.EvidenceArtifactRecord(
                id=f"evidence:{result.step_id}:{artifact_type}",
                run_id=run_id,
                step_id=result.step_id,
                artifact_type=artifact_type,
                redaction_status="safe_structured_summary_no_secrets",
                artifact_json=artifact_json,
            )
        )
    session.add(
        models.MetricResultRecord(
            id=f"metric:{result.step_id}:multi_agent_evidence_completeness",
            run_id=run_id,
            step_id=result.step_id,
            metric_name="multi_agent_evidence_completeness",
            metric_value=1.0,
            metric_json={
                "provider_called": result.provider_called,
                "event_candidate_count": len(result.event_candidates),
                "verification_result_count": len(result.verification_results),
                "route_count": len(result.routes),
                "state_diff_count": len(result.state_diffs),
                "db_schema": "existing_generic_runtime_tables",
            },
        )
    )


def _readback_multi_agent_result(
    *,
    engine: Engine,
    run_id: str,
    step_id: str,
    result: MultiAgentWorldStepResult,
    active_agent_ids: tuple[str, ...],
    prompt_hashes: dict[str, str],
) -> dict[str, object]:
    session_factory = sessionmaker(engine, expire_on_commit=False)
    with session_factory() as session:
        run = session.get(models.RunRecord, run_id)
        step = session.get(models.RunStepRecord, step_id)
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
        provider_rows = tuple(
            session.execute(
                select(
                    models.ProviderCallRecord.id,
                    models.ProviderCallRecord.success,
                    models.ProviderCallRecord.fallback_used,
                )
                .where(models.ProviderCallRecord.run_id == run_id)
                .order_by(models.ProviderCallRecord.id)
            ).mappings()
        )
        structured_rows = tuple(
            session.execute(
                select(
                    models.ProviderCallRecord.id.label("provider_call_id"),
                    models.StructuredOutputValidationRecord.passed,
                )
                .join(
                    models.ProviderCallRecord,
                    models.ProviderCallRecord.id
                    == models.StructuredOutputValidationRecord.provider_call_id,
                )
                .where(models.ProviderCallRecord.run_id == run_id)
                .order_by(models.ProviderCallRecord.id)
            ).mappings()
        )
        llm_input_hashes = {
            _agent_id_from_provider_row(row["provider_call_id"]): row["prompt_sha256"]
            for row in session.execute(
                select(
                    models.LLMInputRecord.provider_call_id,
                    models.LLMInputRecord.prompt_sha256,
                )
                .where(models.LLMInputRecord.provider_call_id.in_(provider_ids))
            ).mappings()
        }
        llm_output_saved_flags = tuple(
            session.scalars(
                select(models.LLMOutputRecord.raw_output_saved).where(
                    models.LLMOutputRecord.provider_call_id.in_(provider_ids)
                )
            )
        )
        counts = {
            "run": _count_run(session, models.RunRecord, run_id),
            "run_step": _count_run(session, models.RunStepRecord, run_id),
            "provider_call_record": _count_run(session, models.ProviderCallRecord, run_id),
            "llm_input_record": _count_in(
                session,
                models.LLMInputRecord.provider_call_id,
                provider_ids,
            ),
            "llm_output_record": _count_in(
                session,
                models.LLMOutputRecord.provider_call_id,
                provider_ids,
            ),
            "structured_output_validation": _count_in(
                session,
                models.StructuredOutputValidationRecord.provider_call_id,
                provider_ids,
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
                session,
                models.StatePatchRecord.state_diff_id,
                state_diff_ids,
            ),
            "world_state_snapshot": _count_run(session, models.WorldStateSnapshotRecord, run_id),
            "trace_event": _count_run(session, models.TraceEventRecord, run_id),
            "evidence_artifact": _count_run(session, models.EvidenceArtifactRecord, run_id),
            "metric_result": _count_run(session, models.MetricResultRecord, run_id),
        }
        proposal_rows = {
            row["proposal_json"]["id"]: row["proposal_json"]
            for row in session.execute(
                select(models.ActionProposalRecord.proposal_json).where(
                    models.ActionProposalRecord.run_id == run_id
                )
            ).mappings()
        }
        candidate_rows = {
            row["candidate_json"]["id"]: row["candidate_json"]
            for row in session.execute(
                select(models.EventCandidateRecord.candidate_json).where(
                    models.EventCandidateRecord.run_id == run_id
                )
            ).mappings()
        }
        verification_rows = {
            row["verification_json"]["id"]: {
                "decision": row["decision"],
                "json": row["verification_json"],
            }
            for row in session.execute(
                select(
                    models.VerificationResultRecord.decision,
                    models.VerificationResultRecord.verification_json,
                ).where(models.VerificationResultRecord.run_id == run_id)
            ).mappings()
        }
        committed_rows = {
            row["committed_event_json"]["id"]: row["committed_event_json"]
            for row in session.execute(
                select(models.CommittedEventRecord.committed_event_json).where(
                    models.CommittedEventRecord.run_id == run_id
                )
            ).mappings()
        }
        state_diff_rows = {
            row["state_diff_json"]["id"]: row["state_diff_json"]
            for row in session.execute(
                select(models.StateDiffRecord.state_diff_json).where(
                    models.StateDiffRecord.run_id == run_id
                )
            ).mappings()
        }
        artifact_rows = {
            row["artifact_type"]: row["artifact_json"]
            for row in session.execute(
                select(
                    models.EvidenceArtifactRecord.artifact_type,
                    models.EvidenceArtifactRecord.artifact_json,
                )
                .where(models.EvidenceArtifactRecord.run_id == run_id)
                .order_by(models.EvidenceArtifactRecord.artifact_type)
            ).mappings()
        }

    expected_counts = {
        "run": 1,
        "run_step": 1,
        "provider_call_record": len(active_agent_ids),
        "llm_input_record": len(active_agent_ids),
        "llm_output_record": len(active_agent_ids),
        "structured_output_validation": len(active_agent_ids),
        "action_proposal": len(result.action_proposals),
        "event_candidate": len(result.event_candidates),
        "verification_result": len(result.verification_results),
        "committed_event": len(result.committed_events),
        "state_diff": len(result.state_diffs),
        "trace_event": 1,
        "evidence_artifact": 6,
        "metric_result": 1,
    }
    comparisons = {
        key: counts[key] == expected
        for key, expected in expected_counts.items()
    }
    comparisons["run_provider_called"] = bool(run.provider_called) if run else False
    comparisons["run_fallback_used_false"] = not bool(run.fallback_used) if run else False
    comparisons["step_structured_validation_passed"] = (
        bool(step.structured_validation_passed) if step else False
    )
    comparisons["active_agent_ids_match"] = (
        tuple((step.metadata_json or {}).get("active_agent_ids", ())) == active_agent_ids
        if step
        else False
    )
    expected_proposal_ids = tuple(proposal.id for proposal in result.action_proposals)
    expected_candidate_ids = tuple(candidate.id for candidate in result.event_candidates)
    expected_verification_ids = tuple(
        verification.id for verification in result.verification_results
    )
    expected_verification_decisions = {
        verification.id: verification.decision.value for verification in result.verification_results
    }
    expected_committed_ids = tuple(event.id for event in result.committed_events)
    expected_state_diff_ids = tuple(diff.id for diff in result.state_diffs)
    expected_routes = [route.model_dump(mode="json") for route in result.routes]
    expected_artifacts = {
        "multi_agent_context": result.context.safe_summary(),
        "multi_agent_proposal_bundle": result.proposal_bundle.model_dump(mode="json"),
        "multi_agent_dynamics": result.dynamics_summary.model_dump(mode="json"),
        "multi_agent_arbitration": result.arbitration_recommendation.model_dump(mode="json"),
        "multi_agent_routes": expected_routes,
        "multi_agent_safe_summary": result.safe_summary(),
    }
    persisted_verification_decisions = {
        verification_id: row["decision"] for verification_id, row in verification_rows.items()
    }
    persisted_proposers = tuple(row["proposer_agent_id"] for row in proposal_rows.values())
    comparisons["persisted_action_proposal_ids_match"] = (
        _keys_in_expected_order(proposal_rows, expected_proposal_ids) == expected_proposal_ids
        and len(proposal_rows) == len(expected_proposal_ids)
    )
    comparisons["persisted_proposer_agent_id_set_match"] = set(persisted_proposers) == set(
        active_agent_ids
    )
    comparisons["persisted_event_candidate_ids_match"] = (
        _keys_in_expected_order(candidate_rows, expected_candidate_ids) == expected_candidate_ids
        and len(candidate_rows) == len(expected_candidate_ids)
    )
    comparisons["persisted_verification_result_ids_match"] = (
        _keys_in_expected_order(verification_rows, expected_verification_ids)
        == expected_verification_ids
        and len(verification_rows) == len(expected_verification_ids)
    )
    comparisons["persisted_verification_decisions_match"] = (
        persisted_verification_decisions == expected_verification_decisions
    )
    comparisons["persisted_committed_event_ids_match"] = (
        _keys_in_expected_order(committed_rows, expected_committed_ids) == expected_committed_ids
        and len(committed_rows) == len(expected_committed_ids)
    )
    comparisons["persisted_state_diff_ids_match"] = (
        _keys_in_expected_order(state_diff_rows, expected_state_diff_ids) == expected_state_diff_ids
        and len(state_diff_rows) == len(expected_state_diff_ids)
    )
    comparisons["provider_call_rows_match_active_agent_count"] = (
        len(provider_rows) == len(active_agent_ids)
    )
    comparisons["provider_call_rows_all_success"] = all(row["success"] for row in provider_rows)
    comparisons["provider_call_rows_no_fallback"] = all(
        not row["fallback_used"] for row in provider_rows
    )
    comparisons["structured_validation_rows_all_passed"] = all(
        row["passed"] for row in structured_rows
    )
    comparisons["llm_input_prompt_hashes_match"] = llm_input_hashes == prompt_hashes
    comparisons["llm_output_rows_raw_output_not_saved"] = all(
        flag is False for flag in llm_output_saved_flags
    )
    comparisons["world_state_snapshot_count_match"] = counts["world_state_snapshot"] == (
        1 + int(result.applied_world_state is not None)
    )
    comparisons["state_patch_count_match"] = counts["state_patch"] == sum(
        len(diff.patches) for diff in result.state_diffs
    )
    comparisons["verifier_check_result_count_match"] = counts["verifier_check_result"] == sum(
        len(verification.checks) for verification in result.verification_results
    )
    comparisons["evidence_artifact_types_match"] = set(artifact_rows) == set(expected_artifacts)
    comparisons["evidence_artifacts_match_runtime"] = artifact_rows == expected_artifacts
    comparisons["route_artifact_content_match"] = artifact_rows.get(
        "multi_agent_routes"
    ) == expected_routes
    safe_summary_artifact = artifact_rows.get("multi_agent_safe_summary", {})
    comparisons["safe_summary_key_ids_match"] = isinstance(
        safe_summary_artifact,
        dict,
    ) and (
        safe_summary_artifact.get("proposal_ids") == list(expected_proposal_ids)
        and safe_summary_artifact.get("event_candidate_ids") == list(expected_candidate_ids)
        and safe_summary_artifact.get("verification_result_ids")
        == list(expected_verification_ids)
        and safe_summary_artifact.get("committed_event_ids") == list(expected_committed_ids)
        and safe_summary_artifact.get("state_diff_ids") == list(expected_state_diff_ids)
    )
    proposal_bundle_artifact = artifact_rows.get("multi_agent_proposal_bundle", {})
    comparisons["proposal_bundle_artifact_ids_match"] = isinstance(
        proposal_bundle_artifact,
        dict,
    ) and (
        proposal_bundle_artifact.get("active_agent_ids") == list(active_agent_ids)
        and [
            frame.get("proposal_id")
            for frame in proposal_bundle_artifact.get("frames", [])
        ]
        == list(expected_proposal_ids)
    )
    return {
        "run_id": run_id,
        "step_id": step_id,
        "counts": counts,
        "expected_counts": expected_counts,
        "comparisons": comparisons,
        "provider_ids": provider_ids,
        "db_readback_passed": all(comparisons.values()),
        "evidence_comparison_passed": all(comparisons.values()),
        "artifact_types": tuple(sorted(artifact_rows)),
        "persisted_action_proposal_ids": tuple(proposal_rows),
        "persisted_event_candidate_ids": tuple(candidate_rows),
        "persisted_verification_result_ids": tuple(verification_rows),
        "persisted_committed_event_ids": tuple(committed_rows),
        "persisted_state_diff_ids": tuple(state_diff_rows),
        "route_artifact": artifact_rows.get("multi_agent_routes"),
    }


def _count_run(session, model, run_id: str) -> int:
    if model is models.RunRecord:
        return int(
            session.scalar(select(func.count()).select_from(model).where(model.id == run_id)) or 0
        )
    return int(
        session.scalar(select(func.count()).select_from(model).where(model.run_id == run_id)) or 0
    )


def _count_in(session, column, values: tuple[str, ...]) -> int:
    if not values:
        return 0
    return int(session.scalar(select(func.count()).where(column.in_(values))) or 0)


def _db_row_id(step_id: str, object_id: str) -> str:
    value = f"{step_id}:{object_id}"
    if len(value) <= 120:
        return value
    return f"{step_id}:{sha256(object_id.encode('utf-8')).hexdigest()[:16]}"


def _keys_in_expected_order(
    rows: dict[str, object],
    expected_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(object_id for object_id in expected_ids if object_id in rows)


def _agent_id_from_provider_row(provider_call_id: str) -> str:
    return provider_call_id.rsplit(":", 1)[-1]


def _load_valid_seed(seed_path: Path) -> SeedBundle:
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
