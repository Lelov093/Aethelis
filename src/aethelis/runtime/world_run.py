from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from aethelis.agents.action_proposal import ProposalSourceMode
from aethelis.agents.activation import AgentActivationBuilder
from aethelis.evolution import DeterministicEvolutionBuilder, append_applied_evolution_update
from aethelis.runtime.scenario_matrix import get_scenario_definition
from aethelis.runtime.single_step import SingleStepResult, run_single_step
from aethelis.runtime.state_store import RuntimeStateStore
from aethelis.schemas.activation import ActivationResult
from aethelis.schemas.events import (
    ActionProposalSummary,
    EventCandidateSummary,
    VerificationDecision,
)
from aethelis.schemas.run import (
    CausalTraceProjection,
    RunConfig,
    RunMode,
    RunStepPlanItem,
    StateTransitionPatchSummary,
    StateTransitionSummary,
    WorldRunResult,
    WorldRunState,
    WorldStepResult,
)
from aethelis.schemas.seed import SeedBundle
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator
from aethelis.utils.redaction import redact_text


class WorldRunConfigurationError(ValueError):
    """Safe run-level configuration error."""


def load_run_config(path: Path) -> RunConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorldRunConfigurationError(
            f"{exc.__class__.__name__}: run config could not be read"
        ) from None
    if payload is None:
        raise WorldRunConfigurationError("Run config is empty.")
    try:
        config = RunConfig.model_validate(payload)
    except ValidationError as exc:
        raise WorldRunConfigurationError("ValidationError: run config schema invalid") from exc
    _validate_deterministic_preview_config(config)
    return config


def run_world(
    *,
    seed_path: Path,
    config: RunConfig,
    apply: bool = False,
    settings: Any | None = None,
) -> WorldRunResult:
    _validate_deterministic_preview_config(config)
    bundle = _load_valid_seed(seed_path)
    seed_id = seed_path.resolve().name
    state = WorldRunState(
        run_id=config.run_id,
        seed_id=seed_id,
        dry_run=not apply,
        world_state=bundle.world,
        safety_flags=(
            config.mode.value,
            "provider_enabled" if config.allow_real_llm else "provider_not_called",
        ),
    )
    runtime_store = RuntimeStateStore(world_state=bundle.world)

    activation_builder = AgentActivationBuilder()
    evolution_builder = DeterministicEvolutionBuilder()
    steps: list[WorldStepResult] = []
    for index, planned_step in enumerate(config.step_plan):
        _validate_step_against_matrix(planned_step, config)
        activation_result = activation_builder.build_for_step(
            bundle=bundle,
            run_id=config.run_id,
            step=planned_step,
            config=config.activation,
            evolution_context=runtime_store.evolution_state.safe_summary(),
        )
        single_step = run_single_step(
            seed_path=seed_path,
            agent_id=planned_step.agent_id,
            scenario_id=planned_step.scenario_id,
            settings=settings if planned_step.allow_real_llm else None,
            proposal_source=(
                ProposalSourceMode.PROVIDER_STRUCTURED
                if planned_step.allow_real_llm
                else ProposalSourceMode.DETERMINISTIC
            ),
            provider_proposals_enabled=planned_step.allow_real_llm,
            allow_real_provider=bool(config.allow_real_llm and planned_step.allow_real_llm),
            apply=bool(apply and planned_step.apply),
            world_state_override=runtime_store.world_state,
            pressure_context=_pressure_context_summary(bundle),
            evolution_context=runtime_store.evolution_state.safe_summary(),
        )
        step = _world_step_result(
            bundle=bundle,
            planned_step=planned_step,
            step_index=index,
            result=single_step,
            activation_result=activation_result,
            evolution_builder=evolution_builder,
        )
        steps.append(step)
        runtime_store = _advance_runtime_store(runtime_store, step, single_step)
        state = _advance_state(state, step, runtime_store)

    decisions = tuple(step.decision for step in steps)
    return WorldRunResult(
        run_id=config.run_id,
        seed_id=seed_id,
        mode=config.mode,
        dry_run=not apply,
        apply_requested=apply,
        provider_called=any(step.provider_called for step in steps),
        step_count=len(steps),
        decisions=decisions,
        steps=tuple(steps),
        state_diff_applied=any(step.state_diff_applied for step in steps),
        state_diff_applied_count=sum(1 for step in steps if step.state_diff_applied),
        committed_event_count=sum(
            1 for step in steps if step.decision == VerificationDecision.COMMIT
        ),
        rejected_count=sum(1 for step in steps if step.decision == VerificationDecision.REJECT),
        revise_count=sum(1 for step in steps if step.decision == VerificationDecision.REVISE),
        pending_gate_count=sum(
            1 for step in steps if step.decision == VerificationDecision.PENDING_GATE
        ),
        final_state_summary=_world_state_summary(state.world_state),
        final_evolution_state_summary=state.evolution_state.safe_summary(),
        causal_runtime_summary=state.evolution_state.causal_runtime_summary(),
        pressure_runtime_summary=state.evolution_state.pressure_runtime_summary(),
        cognitive_runtime_summary=state.evolution_state.cognitive_runtime_summary(),
        apply_journal_count=state.apply_journal_count,
        replay_journal_count=state.replay_journal_count,
    )


def _validate_deterministic_preview_config(config: RunConfig) -> None:
    if config.mode not in {RunMode.DETERMINISTIC_PREVIEW, RunMode.REAL_PROVIDER_PREVIEW}:
        raise WorldRunConfigurationError("unsupported_run_mode")
    if config.formal_experiment_result:
        raise WorldRunConfigurationError("formal_experiment_result_not_allowed")
    if config.mode == RunMode.DETERMINISTIC_PREVIEW and config.allow_real_llm:
        raise WorldRunConfigurationError("real_llm_not_allowed_in_deterministic_preview")
    if config.mode == RunMode.REAL_PROVIDER_PREVIEW and not config.allow_real_llm:
        raise WorldRunConfigurationError("real_provider_preview_requires_allow_real_llm")
    if not config.dry_run:
        raise WorldRunConfigurationError("run_config_must_default_to_dry_run")
    if config.apply:
        raise WorldRunConfigurationError("run_config_apply_must_be_false")
    if config.activation.allow_real_llm:
        raise WorldRunConfigurationError("activation_real_llm_not_allowed_in_deterministic_preview")
    if config.activation.allow_private_belief_scoring:
        raise WorldRunConfigurationError("activation_private_belief_scoring_not_allowed")
    for step in config.step_plan:
        scenario = get_scenario_definition(step.scenario_id)
        if config.mode == RunMode.DETERMINISTIC_PREVIEW and step.allow_real_llm:
            raise WorldRunConfigurationError(f"step_real_llm_not_allowed: {step.step_id}")
        if (
            config.mode == RunMode.REAL_PROVIDER_PREVIEW
            and step.allow_real_llm
            and not scenario.allows_real_llm
        ):
            raise WorldRunConfigurationError(f"step_real_llm_not_supported: {step.step_id}")


def _validate_step_against_matrix(step: RunStepPlanItem, config: RunConfig) -> None:
    scenario = get_scenario_definition(step.scenario_id)
    if scenario.allows_real_llm and not (
        config.mode == RunMode.REAL_PROVIDER_PREVIEW and step.allow_real_llm
    ):
        raise WorldRunConfigurationError(f"scenario_requires_real_llm: {step.scenario_id}")
    if step.agent_id != scenario.actor_id:
        raise WorldRunConfigurationError(
            f"scenario_actor_mismatch: {step.scenario_id} expected {scenario.actor_id}"
        )
    if step.actor_type != scenario.actor_type:
        raise WorldRunConfigurationError(
            f"scenario_actor_type_mismatch: {step.scenario_id} expected {scenario.actor_type}"
        )


def _world_step_result(
    *,
    bundle: SeedBundle,
    planned_step: RunStepPlanItem,
    step_index: int,
    result: SingleStepResult,
    activation_result: ActivationResult,
    evolution_builder: DeterministicEvolutionBuilder,
) -> WorldStepResult:
    if result.verification_result is None:
        raise WorldRunConfigurationError(f"missing_verification_result: {planned_step.step_id}")
    player_claim_id = result.player_claim.claim_id if result.player_claim is not None else None
    player_claim_summary = (
        _summarize(result.player_claim.claim) if result.player_claim is not None else None
    )
    player_claim_rejected_claim_ids = (
        result.player_claim.verification_result.rejected_claim_ids
        if result.player_claim is not None
        else ()
    )
    return WorldStepResult(
        step_id=planned_step.step_id,
        step_index=step_index,
        agent_id=planned_step.agent_id,
        actor_type=planned_step.actor_type,
        scenario_id=planned_step.scenario_id,
        decision=result.verification_result.decision,
        action_proposal_id=(
            result.action_proposal.id if result.action_proposal is not None else None
        ),
        proposal_summary=(
            ActionProposalSummary.from_proposal(
                result.action_proposal,
                generated_by=result.proposal_source or "deterministic_fixture",
            )
            if result.action_proposal is not None
            else None
        ),
        event_candidate_id=(
            result.event_candidate.id if result.event_candidate is not None else None
        ),
        candidate_summary=(
            EventCandidateSummary.from_candidate(
                result.event_candidate,
                candidate_kind=get_scenario_definition(planned_step.scenario_id).candidate_kind,
            )
            if result.event_candidate is not None
            else None
        ),
        verification_result_id=result.verification_result.id,
        verification_checks=tuple(
            {
                "name": check.name,
                "passed": check.passed,
                "message": check.message,
            }
            for check in result.verification_result.checks
        ),
        verification_reasons=result.verification_result.reasons,
        verification_risk_flags=result.verification_result.risk_flags,
        committed_event_id=(
            result.committed_event.id if result.committed_event is not None else None
        ),
        state_diff_id=(
            result.committed_event.state_diff.id if result.committed_event is not None else None
        ),
        state_diff_applied=result.state_diff_applied,
        apply_report=(result.apply_report.safe_dict() if result.apply_report is not None else None),
        state_transition=_state_transition_summary(planned_step, result),
        causal_projection=_causal_projection(result),
        evolution_update=evolution_builder.build_for_step(
            bundle=bundle,
            step_id=planned_step.step_id,
            scenario_id=planned_step.scenario_id,
            agent_id=planned_step.agent_id,
            decision=result.verification_result.decision,
            committed_event_id=(
                result.committed_event.id if result.committed_event is not None else None
            ),
            state_diff_id=(
                result.committed_event.state_diff.id if result.committed_event is not None else None
            ),
            verification_result_id=result.verification_result.id,
            event_candidate_id=(
                result.event_candidate.id if result.event_candidate is not None else None
            ),
            state_diff_applied=result.state_diff_applied,
            verification_result=result.verification_result,
            event_candidate=result.event_candidate,
        ),
        player_input_summary=result.player_input_summary,
        retrieval_summary=result.retrieval_summary,
        proposal_source=result.proposal_source,
        provider_mode=result.provider_mode,
        fallback_used=result.fallback_used,
        fallback_reason=result.fallback_reason,
        evidence_class=result.evidence_class,
        provider_called=result.provider_called,
        player_claim_id=player_claim_id,
        player_claim_summary=player_claim_summary,
        player_claim_canon_updated=(
            result.player_claim.canon_updated if result.player_claim is not None else False
        ),
        player_claim_state_diff_id=(
            result.player_claim.state_diff_id if result.player_claim is not None else None
        ),
        player_claim_rejected_claim_ids=player_claim_rejected_claim_ids,
        activation_result=activation_result,
        safety_flags=_step_safety_flags(result),
        notes=(
            tuple(redact_text(reason) for reason in result.verification_result.reasons)
            if result.verification_result is not None
            else ()
        ),
    )


def _advance_state(
    state: WorldRunState,
    step: WorldStepResult,
    runtime_store: RuntimeStateStore,
) -> WorldRunState:
    return WorldRunState(
        run_id=state.run_id,
        seed_id=state.seed_id,
        dry_run=state.dry_run,
        current_step_index=step.step_index + 1,
        world_state=runtime_store.world_state,
        state_diff_applied_count=state.state_diff_applied_count + int(step.state_diff_applied),
        committed_event_count=state.committed_event_count
        + int(step.decision == VerificationDecision.COMMIT),
        rejected_count=state.rejected_count + int(step.decision == VerificationDecision.REJECT),
        revise_count=state.revise_count + int(step.decision == VerificationDecision.REVISE),
        pending_gate_count=state.pending_gate_count
        + int(step.decision == VerificationDecision.PENDING_GATE),
        provider_called=state.provider_called or step.provider_called,
        evolution_state=runtime_store.evolution_state,
        apply_journal_count=len(runtime_store.apply_journal),
        replay_journal_count=len(runtime_store.replay_journal),
        safety_flags=state.safety_flags,
    )


def _advance_runtime_store(
    store: RuntimeStateStore,
    step: WorldStepResult,
    result: SingleStepResult,
) -> RuntimeStateStore:
    next_store = store
    if result.apply_report is not None and result.applied_world_state is not None:
        next_store, _ = next_store.record_apply_result(
            world_state=result.applied_world_state,
            report=result.apply_report,
            verification_result_id=step.verification_result_id,
        )
    next_evolution_state = append_applied_evolution_update(
        next_store.evolution_state,
        step.evolution_update,
    )
    return next_store.with_evolution_state(next_evolution_state)


def _load_valid_seed(seed_path: Path) -> SeedBundle:
    load_result = SeedLoader().load(seed_path)
    report = SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )
    if not report.success or load_result.bundle is None:
        raise WorldRunConfigurationError(f"Seed validation failed: {report.safe_dict()}")
    return load_result.bundle


def _step_safety_flags(result: SingleStepResult) -> tuple[str, ...]:
    flags = [
        result.provider_mode or "provider_mode_unknown",
        result.evidence_class or "evidence_class_unknown",
        "provider_called" if result.provider_called else "provider_not_called",
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
    if result.player_claim is not None:
        flags.append("player_claim_not_canon")
    return tuple(flags)


def _state_transition_summary(
    planned_step: RunStepPlanItem,
    result: SingleStepResult,
) -> StateTransitionSummary | None:
    if result.committed_event is None:
        return None
    state_diff = result.committed_event.state_diff
    report = result.apply_report.safe_dict() if result.apply_report is not None else None
    report_patch_results = (
        {int(item["patch_index"]): item for item in report.get("patch_results", [])}
        if report is not None
        else {}
    )
    patches: list[StateTransitionPatchSummary] = []
    for index, patch in enumerate(state_diff.patches):
        applied_patch = report_patch_results.get(index)
        patches.append(
            StateTransitionPatchSummary(
                patch_index=index,
                applied=bool(applied_patch.get("applied", False)) if applied_patch else False,
                target_type=patch.target_type.value,
                target_id=patch.target_id,
                path=patch.path,
                before_summary=_safe_value_summary(
                    applied_patch.get("before") if applied_patch else patch.before
                ),
                after_summary=_safe_value_summary(
                    applied_patch.get("after") if applied_patch else patch.after
                ),
                error=(
                    str(applied_patch.get("error"))
                    if applied_patch and applied_patch.get("error")
                    else None
                ),
            )
        )
    return StateTransitionSummary(
        step_id=planned_step.step_id,
        committed_event_id=result.committed_event.id,
        state_diff_id=state_diff.id,
        applied=bool(report and report.get("applied", False)),
        applied_patch_count=int(report.get("applied_patch_count", 0)) if report else 0,
        skipped_patch_count=int(report.get("skipped_patch_count", 0)) if report else 0,
        patches=tuple(patches),
    )


def _causal_projection(result: SingleStepResult) -> CausalTraceProjection | None:
    if result.committed_event is None or result.verification_result is None:
        return None
    state_diff = result.committed_event.state_diff
    affected = tuple(f"{patch.target_type.value}:{patch.target_id}" for patch in state_diff.patches)
    return CausalTraceProjection(
        committed_event_node_id=f"event:{result.committed_event.id}",
        state_diff_node_id=f"state_diff:{state_diff.id}",
        verification_result_id=result.verification_result.id,
        affected_target_node_ids=affected,
        caused_state_diff_edge_id=f"edge:{result.committed_event.id}:caused:{state_diff.id}",
    )


def _safe_value_summary(value):
    if isinstance(value, str):
        return _summarize(value, limit=80)
    if isinstance(value, list | tuple):
        items = list(value)
        return items[:8] + (["..."] if len(items) > 8 else [])
    if isinstance(value, dict):
        return {"keys": sorted(str(key) for key in value)[:8]}
    return value


def _world_state_summary(world_state) -> dict[str, object] | None:
    if world_state is None:
        return None
    discovered_resources = [
        {
            "resource_id": resource.id,
            "discovered_by_agent_ids": list(resource.discovery_state.discovered_by_agent_ids),
        }
        for resource in world_state.resources
        if resource.discovery_state.discovered_by_agent_ids
    ]
    return {
        "location_count": len(world_state.locations),
        "entity_count": len(world_state.entities),
        "resource_count": len(world_state.resources),
        "canon_fact_count": len(world_state.canon_facts),
        "discovered_resources": discovered_resources,
    }


def _pressure_context_summary(bundle: SeedBundle) -> dict[str, object] | None:
    if bundle.metadata is None:
        return None
    return {
        "pressure_seed_count": len(bundle.metadata.pressure_seeds),
        "pressure_types": sorted(
            {pressure.pressure_type for pressure in bundle.metadata.pressure_seeds}
        ),
        "max_pressure_level": (
            max((pressure.level for pressure in bundle.metadata.pressure_seeds), default=0)
        ),
    }


def _summarize(value: str, limit: int = 120) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."
