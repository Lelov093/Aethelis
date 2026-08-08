from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from pydantic import Field
from sqlalchemy.engine import Engine

from aethelis.algorithms.runtime_wiring import (
    ALGORITHM_MODE_V11,
    build_runtime_algorithm_decisions,
)
from aethelis.config.settings import Settings
from aethelis.db.repository import RuntimeDBRepository
from aethelis.embedding.volcengine_ark import VolcengineArkEmbeddingProvider
from aethelis.evolution import DeterministicEvolutionBuilder
from aethelis.providers import ProviderError
from aethelis.runtime.single_step import run_single_step
from aethelis.runtime.state_store import RuntimeStateStore
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator
from aethelis.utils.redaction import redact_text


class RealProviderDBRunResult(AethelisModel):
    run_id: Identifier
    step_id: Identifier
    seed_id: Identifier
    scenario_id: Identifier
    agent_id: Identifier
    algorithm_mode: Identifier
    provider_called: bool
    fallback_used: bool
    structured_validation_passed: bool
    state_diff_applied: bool
    db_persisted: bool
    db_readback_passed: bool
    provider_name: str | None = None
    model_name: str | None = None
    action_proposal_id: str | None = None
    event_candidate_id: str | None = None
    verification_result_id: str | None = None
    committed_event_id: str | None = None
    state_diff_id: str | None = None
    algorithm_decision_count: int = Field(ge=0)
    mechanism_coverage_count: int = Field(ge=0)
    mechanism_coverage_passed: bool = False
    embedding_provider_called: bool = False
    embedding_fallback_used: bool = False
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    embedding_vector_norm: float | None = None
    embedding_db_readback_passed: bool = False
    readback: dict[str, object]

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class LongHorizonDBRunResult(AethelisModel):
    run_id: Identifier
    seed_id: Identifier
    scenario_id: Identifier
    agent_id: Identifier
    algorithm_mode: Identifier
    requested_step_count: int = Field(ge=1)
    completed_step_count: int = Field(ge=0)
    status: str
    provider_called_count: int = Field(ge=0)
    fallback_used_count: int = Field(ge=0)
    structured_validation_passed_count: int = Field(ge=0)
    state_diff_applied_count: int = Field(ge=0)
    algorithm_decision_count: int = Field(ge=0)
    embedding_provider_called_count: int = Field(ge=0)
    embedding_fallback_used_count: int = Field(ge=0)
    db_persisted_step_count: int = Field(ge=0)
    db_readback_passed: bool
    mechanism_coverage_passed: bool
    long_horizon_db_readback_passed: bool
    step_ids: tuple[str, ...] = ()
    failure_step_index: int | None = None
    failure_summary: str | None = None
    failure_code: str | None = None
    failure_provider_called: bool | None = None
    failure_raw_text_sha256: str | None = None
    failure_json_parse_error: str | None = None
    failure_validation_error: str | None = None
    readback: dict[str, object]

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def run_real_provider_db_step(
    *,
    engine: Engine,
    seed_path: Path,
    agent_id: str,
    scenario_id: str,
    settings: Settings,
    algorithm_config_path: Path | None,
    algorithm_mode: str = ALGORITHM_MODE_V11,
) -> RealProviderDBRunResult:
    seed_bundle = _load_valid_seed(seed_path)
    run_id = f"db_run_{uuid4().hex}"
    step_id = f"{run_id}_step_1"
    result = run_single_step(
        seed_path=seed_path,
        agent_id=agent_id,
        scenario_id=scenario_id,
        settings=settings,
        apply=True,
        allow_real_provider=True,
    )
    if result.error is not None:
        raise RuntimeError(result.error)
    if not result.provider_called:
        raise RuntimeError("Real provider was not called.")
    if result.fallback_used:
        raise RuntimeError("Fallback path was used; refusing DB-backed success.")
    if result.structured_output is None or not result.structured_output.success:
        raise RuntimeError("Structured output validation failed.")
    if not result.state_diff_applied:
        raise RuntimeError("Committed StateDiff was not applied to the WorldState copy.")
    if result.committed_event is None:
        raise RuntimeError("Verification did not produce a CommittedEvent.")

    result = _with_evolution_update(seed_bundle, result, step_id=step_id)
    embedding = VolcengineArkEmbeddingProvider(settings).embed_text(
        json.dumps(result.safe_summary(), ensure_ascii=False, sort_keys=True)
    )
    if embedding.dimensions != settings.embedding_dimensions:
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {settings.embedding_dimensions}, "
            f"actual {embedding.dimensions}"
        )

    decisions = build_runtime_algorithm_decisions(
        seed_path=seed_path,
        result=result,
        config_path=algorithm_config_path,
        evolution_update=result.evolution_update,
    )
    repository = RuntimeDBRepository(engine)
    repository.persist_real_provider_step(
        run_id=run_id,
        step_id=step_id,
        seed_id=seed_path.name,
        seed_bundle=seed_bundle,
        result=result,
        algorithm_mode=algorithm_mode,
        algorithm_decisions=decisions,
        embedding_result=embedding,
    )
    readback = repository.readback_summary(run_id=run_id)
    return RealProviderDBRunResult(
        run_id=run_id,
        step_id=step_id,
        seed_id=seed_path.name,
        scenario_id=result.scenario_id,
        agent_id=result.agent_id,
        algorithm_mode=algorithm_mode,
        provider_called=result.provider_called,
        fallback_used=result.fallback_used,
        structured_validation_passed=result.structured_output.success,
        state_diff_applied=result.state_diff_applied,
        db_persisted=True,
        db_readback_passed=bool(readback["db_readback_passed"]),
        provider_name=result.structured_output.provider_name,
        model_name=result.structured_output.model_name,
        action_proposal_id=result.action_proposal.id if result.action_proposal else None,
        event_candidate_id=result.event_candidate.id if result.event_candidate else None,
        verification_result_id=(
            result.verification_result.id if result.verification_result else None
        ),
        committed_event_id=result.committed_event.id if result.committed_event else None,
        state_diff_id=result.committed_event.state_diff.id if result.committed_event else None,
        algorithm_decision_count=len(decisions),
        mechanism_coverage_count=int(readback["mechanism_coverage_count"]),
        mechanism_coverage_passed=bool(readback["mechanism_coverage_passed"]),
        embedding_provider_called=True,
        embedding_fallback_used=False,
        embedding_model=embedding.model,
        embedding_dimensions=embedding.dimensions,
        embedding_vector_norm=readback["embedding_db_readback"].get("vector_norm"),
        embedding_db_readback_passed=bool(readback["embedding_db_readback_passed"]),
        readback=readback,
    )


def run_mistgate_long_horizon_db(
    *,
    engine: Engine,
    seed_path: Path,
    agent_id: str,
    scenario_id: str,
    settings: Settings,
    algorithm_config_path: Path | None,
    step_count: int = 20,
    run_id: str | None = None,
    algorithm_mode: str = ALGORITHM_MODE_V11,
) -> LongHorizonDBRunResult:
    if step_count < 20 or step_count > 50:
        raise ValueError("R2 long-horizon step_count must be between 20 and 50.")

    seed_bundle = _load_valid_seed(seed_path)
    run_id = run_id or f"r2_mistgate_db_long_horizon_{uuid4().hex}"
    repository = RuntimeDBRepository(engine)
    runtime_store = RuntimeStateStore(world_state=seed_bundle.world)
    embedding_provider = VolcengineArkEmbeddingProvider(settings)
    step_ids: list[str] = []
    failure_step_index: int | None = None
    failure_summary: str | None = None
    failure_result = None

    for step_index in range(1, step_count + 1):
        step_id = f"{run_id}_step_{step_index:02d}"
        before_world_state = runtime_store.world_state
        result = run_single_step(
            seed_path=seed_path,
            agent_id=agent_id,
            scenario_id=scenario_id,
            settings=settings,
            apply=True,
            allow_real_provider=True,
            world_state_override=before_world_state,
        )
        failure_summary = _step_failure_summary(result)
        if failure_summary is not None:
            failure_step_index = step_index
            failure_result = result
            break

        result = _with_evolution_update(seed_bundle, result, step_id=step_id)
        try:
            embedding = embedding_provider.embed_text(
                json.dumps(result.safe_summary(), ensure_ascii=False, sort_keys=True)
            )
        except ProviderError as exc:
            failure_step_index = step_index
            failure_summary = f"embedding_provider_error: {redact_text(str(exc))}"
            failure_result = result
            break
        if embedding.dimensions != settings.embedding_dimensions:
            failure_step_index = step_index
            failure_summary = (
                f"embedding_dimension_mismatch: expected {settings.embedding_dimensions}, "
                f"actual {embedding.dimensions}"
            )
            failure_result = result
            break

        decisions = build_runtime_algorithm_decisions(
            seed_path=seed_path,
            result=result,
            config_path=algorithm_config_path,
            evolution_update=result.evolution_update,
        )
        repository.persist_real_provider_step(
            run_id=run_id,
            step_id=step_id,
            seed_id=seed_path.name,
            seed_bundle=seed_bundle,
            result=result,
            algorithm_mode=algorithm_mode,
            algorithm_decisions=decisions,
            embedding_result=embedding,
            step_index=step_index,
            before_world_state=before_world_state,
        )
        step_ids.append(step_id)
        runtime_store, _ = runtime_store.record_apply_result(
            world_state=result.applied_world_state,
            report=result.apply_report,
            verification_result_id=result.verification_result.id,
        )

    readback = repository.long_horizon_readback_summary(
        run_id=run_id,
        expected_steps=step_count,
    )
    counts = readback["counts"]
    completed_step_count = len(step_ids)
    status = "completed" if completed_step_count == step_count and failure_summary is None else "partial"
    if completed_step_count == 0 and failure_summary is not None:
        status = "failed"
    return LongHorizonDBRunResult(
        run_id=run_id,
        seed_id=seed_path.name,
        scenario_id=scenario_id,
        agent_id=agent_id,
        algorithm_mode=algorithm_mode,
        requested_step_count=step_count,
        completed_step_count=completed_step_count,
        status=status,
        provider_called_count=completed_step_count,
        fallback_used_count=0,
        structured_validation_passed_count=completed_step_count,
        state_diff_applied_count=int(counts["state_diff"]),
        algorithm_decision_count=int(counts["algorithm_decision"]),
        embedding_provider_called_count=int(counts["embedding_record"]),
        embedding_fallback_used_count=0,
        db_persisted_step_count=int(counts["run_step"]),
        db_readback_passed=bool(readback["db_readback_passed"]),
        mechanism_coverage_passed=bool(readback["mechanism_coverage_passed"]),
        long_horizon_db_readback_passed=bool(readback["long_horizon_db_readback_passed"]),
        step_ids=tuple(step_ids),
        failure_step_index=failure_step_index,
        failure_summary=failure_summary,
        failure_code=(
            failure_result.failure_code.value
            if failure_result is not None and failure_result.failure_code is not None
            else None
        ),
        failure_provider_called=(
            failure_result.provider_called if failure_result is not None else None
        ),
        failure_raw_text_sha256=(
            failure_result.structured_output.raw_text_sha256
            if failure_result is not None and failure_result.structured_output is not None
            else None
        ),
        failure_json_parse_error=(
            failure_result.structured_output.json_parse_error
            if failure_result is not None and failure_result.structured_output is not None
            else None
        ),
        failure_validation_error=(
            failure_result.structured_output.validation_error
            if failure_result is not None and failure_result.structured_output is not None
            else None
        ),
        readback=readback,
    )


def _with_evolution_update(seed_bundle, result, *, step_id: str):
    if result.verification_result is None:
        return result
    update = DeterministicEvolutionBuilder().build_for_step(
        bundle=seed_bundle,
        step_id=step_id,
        scenario_id=result.scenario_id,
        agent_id=result.agent_id,
        decision=result.verification_result.decision,
        committed_event_id=(result.committed_event.id if result.committed_event else None),
        state_diff_id=(result.committed_event.state_diff.id if result.committed_event else None),
        verification_result_id=result.verification_result.id,
        event_candidate_id=(result.event_candidate.id if result.event_candidate else None),
        state_diff_applied=result.state_diff_applied,
        verification_result=result.verification_result,
        event_candidate=result.event_candidate,
    )
    return result.model_copy(update={"evolution_update": update})


def _step_failure_summary(result) -> str | None:
    if result.structured_output is not None and not result.structured_output.success:
        detail = (
            result.structured_output.json_parse_error
            or result.structured_output.validation_error
            or result.error
            or "unknown_structured_output_error"
        )
        code = result.failure_code.value if result.failure_code is not None else "unknown"
        return f"structured_output_failed: failure_code={code}; {redact_text(detail)}"
    if result.error is not None:
        return f"runtime_error: {redact_text(result.error)}"
    if not result.provider_called:
        return "real_provider_not_called"
    if result.fallback_used:
        return "fallback_path_used"
    if result.structured_output is None or not result.structured_output.success:
        return "structured_output_validation_failed"
    if result.committed_event is None:
        return "verification_did_not_produce_committed_event"
    if result.apply_report is None or result.applied_world_state is None:
        return "state_diff_apply_report_missing"
    if not result.state_diff_applied:
        return f"state_diff_not_applied: {redact_text('; '.join(result.apply_report.errors))}"
    return None


def _load_valid_seed(seed_path: Path):
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
