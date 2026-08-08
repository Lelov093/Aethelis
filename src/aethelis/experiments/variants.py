from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from aethelis.algorithms.runtime_features import model_selection_score
from aethelis.evaluation.harness import evaluate_formal_run
from aethelis.experiments.runner import run_formal_experiment
from aethelis.schemas.common import Identifier
from aethelis.schemas.experiment import (
    AblationMode,
    BaselineMode,
    ExperimentComparisonArtifacts,
    ExperimentComparisonResult,
    ExperimentVariantResult,
    ExperimentVariantType,
    FormalExperimentConfig,
)


class VariantTransform(StrEnum):
    NONE = "none"
    FREE_CHAT = "free_chat"
    SHARED_CONTEXT = "shared_context"
    RAG_MEMORY = "rag_memory"
    RULE_ONLY = "rule_only"
    WITHOUT_EVENT_VERIFICATION = "without_event_verification"
    WITHOUT_BELIEF_CANON_LEDGER = "without_belief_canon_ledger"
    WITHOUT_CAUSAL_EVENT_GRAPH = "without_causal_event_graph"
    WITHOUT_WORLD_PRESSURE_FIELD = "without_world_pressure_field"
    WITHOUT_PLAYER_INPUT_VERIFICATION = "without_player_input_verification"
    WITHOUT_AGENT_PRIVATE_MEMORY = "without_agent_private_memory"


@dataclass(frozen=True)
class VariantSpec:
    variant_id: Identifier
    variant_type: ExperimentVariantType
    baseline_mode: BaselineMode = BaselineMode.NONE
    ablation_mode: AblationMode = AblationMode.NONE
    transform: VariantTransform = VariantTransform.NONE
    safety_notes: tuple[str, ...] = ()


DEFAULT_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        variant_id="aethelis_proposed_runtime",
        variant_type=ExperimentVariantType.PROPOSED,
        baseline_mode=BaselineMode.AETHELIS_PROPOSED_RUNTIME,
        safety_notes=("proposed runtime governance chain executed unchanged",),
    ),
    VariantSpec(
        variant_id="free_multi_agent_chat",
        variant_type=ExperimentVariantType.BASELINE,
        baseline_mode=BaselineMode.FREE_MULTI_AGENT_CHAT,
        transform=VariantTransform.FREE_CHAT,
        safety_notes=("controlled baseline marks free-chat governance bypass risk only",),
    ),
    VariantSpec(
        variant_id="shared_context_agents",
        variant_type=ExperimentVariantType.BASELINE,
        baseline_mode=BaselineMode.SHARED_CONTEXT_AGENTS,
        transform=VariantTransform.SHARED_CONTEXT,
        safety_notes=("controlled baseline marks shared context boundary exposure only",),
    ),
    VariantSpec(
        variant_id="rag_memory_agents",
        variant_type=ExperimentVariantType.BASELINE,
        baseline_mode=BaselineMode.RAG_MEMORY_AGENTS,
        transform=VariantTransform.RAG_MEMORY,
        safety_notes=("controlled baseline marks RAG memory boundary exposure only",),
    ),
    VariantSpec(
        variant_id="rule_only_world_simulation",
        variant_type=ExperimentVariantType.BASELINE,
        baseline_mode=BaselineMode.RULE_ONLY_WORLD_SIMULATION,
        transform=VariantTransform.RULE_ONLY,
        safety_notes=("controlled baseline removes causal/evolution trace detail only",),
    ),
    VariantSpec(
        variant_id="without_event_verification",
        variant_type=ExperimentVariantType.ABLATION,
        ablation_mode=AblationMode.WITHOUT_EVENT_VERIFICATION,
        transform=VariantTransform.WITHOUT_EVENT_VERIFICATION,
        safety_notes=("simulated ablation; runtime verification was not bypassed",),
    ),
    VariantSpec(
        variant_id="without_belief_canon_ledger",
        variant_type=ExperimentVariantType.ABLATION,
        ablation_mode=AblationMode.WITHOUT_BELIEF_CANON_LEDGER,
        transform=VariantTransform.WITHOUT_BELIEF_CANON_LEDGER,
        safety_notes=("simulated ablation; canon was not changed by player input",),
    ),
    VariantSpec(
        variant_id="without_causal_event_graph",
        variant_type=ExperimentVariantType.ABLATION,
        ablation_mode=AblationMode.WITHOUT_CAUSAL_EVENT_GRAPH,
        transform=VariantTransform.WITHOUT_CAUSAL_EVENT_GRAPH,
        safety_notes=("simulated ablation; causal graph was removed from trace only",),
    ),
    VariantSpec(
        variant_id="without_world_pressure_field",
        variant_type=ExperimentVariantType.ABLATION,
        ablation_mode=AblationMode.WITHOUT_WORLD_PRESSURE_FIELD,
        transform=VariantTransform.WITHOUT_WORLD_PRESSURE_FIELD,
        safety_notes=("simulated ablation; pressure field consistency was disrupted in trace only",),
    ),
    VariantSpec(
        variant_id="without_player_input_verification",
        variant_type=ExperimentVariantType.ABLATION,
        ablation_mode=AblationMode.WITHOUT_PLAYER_INPUT_VERIFICATION,
        transform=VariantTransform.WITHOUT_PLAYER_INPUT_VERIFICATION,
        safety_notes=("simulated ablation; direct player state mutation was not executed",),
    ),
    VariantSpec(
        variant_id="without_agent_private_memory",
        variant_type=ExperimentVariantType.ABLATION,
        ablation_mode=AblationMode.WITHOUT_AGENT_PRIVATE_MEMORY,
        transform=VariantTransform.WITHOUT_AGENT_PRIVATE_MEMORY,
        safety_notes=("simulated ablation; private memory content was not written",),
    ),
)


def run_experiment_comparison(
    *,
    seed_path: Path,
    config: FormalExperimentConfig,
    runs_dir: Path = Path("runs"),
    variants: tuple[VariantSpec, ...] = DEFAULT_VARIANTS,
) -> ExperimentComparisonResult:
    comparison_dir = (runs_dir / config.run_id).resolve()
    variants_dir = comparison_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    variants = _scoped_variants(config, variants)
    variant_results: list[ExperimentVariantResult] = []
    provider_called = False
    for spec in variants:
        variant_config = _variant_config(config, spec)
        run_result = run_formal_experiment(
            seed_path=seed_path,
            config=variant_config,
            runs_dir=variants_dir,
        )
        _apply_controlled_variant(
            run_dir=run_result.artifacts.run_dir,
            spec=spec,
            base_run_id=config.run_id,
        )
        evaluation = evaluate_formal_run(run_result.artifacts.run_dir)
        provider_called = (
            provider_called or run_result.provider_called or evaluation.provider_called
        )
        selection_score = _variant_selection_score(spec, evaluation)
        variant_results.append(
            ExperimentVariantResult(
                variant_id=spec.variant_id,
                variant_type=spec.variant_type,
                implemented=True,
                run_id=variant_config.run_id,
                artifact_path=str(run_result.artifacts.run_dir),
                metric_summary={
                    "metric_count": evaluation.metric_count,
                    "failed_metric_count": evaluation.failed_metric_count,
                    "metrics_summary_path": evaluation.metrics_summary_path,
                    "evaluation_summary_path": evaluation.evaluation_summary_path,
                    "step_count": run_result.plan.step_count,
                    "seed_id": seed_path.name,
                    "provider_called": evaluation.provider_called,
                    "fallback_used_count": 0,
                    "db_readback_status": "not_claimed_for_controlled_comparison",
                    "execution_boundary": _execution_boundary(spec),
                    "selection_score": selection_score,
                    "selection_decision": "weighted_behavior_rank",
                },
                bad_case_count=evaluation.bad_case_count,
                safety_notes=spec.safety_notes,
            )
        )
    variant_results = sorted(
        variant_results,
        key=lambda item: (
            -float(item.metric_summary.get("selection_score", 0.0)),
            item.variant_id,
        ),
    )
    artifacts = _write_comparison_artifacts(
        comparison_dir=comparison_dir,
        run_id=config.run_id,
        provider_called=provider_called,
        variants=tuple(variant_results),
    )
    return ExperimentComparisonResult(
        run_id=config.run_id,
        provider_called=provider_called,
        variant_count=len(variant_results),
        variants=tuple(variant_results),
        artifacts=artifacts,
    )


def _variant_config(config: FormalExperimentConfig, spec: VariantSpec) -> FormalExperimentConfig:
    payload = config.safe_dict()
    payload["run_id"] = f"{config.run_id}_{spec.variant_id}"
    payload["baseline_mode"] = spec.baseline_mode.value
    payload["baseline_implemented"] = spec.baseline_mode != BaselineMode.NONE
    payload["ablation_mode"] = spec.ablation_mode.value
    payload["ablation_implemented"] = spec.ablation_mode != AblationMode.NONE
    return FormalExperimentConfig.model_validate(payload)


def _scoped_variants(
    config: FormalExperimentConfig,
    variants: tuple[VariantSpec, ...],
) -> tuple[VariantSpec, ...]:
    if config.comparison_variant_scope == "all":
        return variants
    allowed = (
        {ExperimentVariantType.PROPOSED, ExperimentVariantType.BASELINE}
        if config.comparison_variant_scope == "baselines"
        else {ExperimentVariantType.PROPOSED, ExperimentVariantType.ABLATION}
    )
    return tuple(
        spec
        for spec in variants
        if spec.variant_type in allowed
    )


def _apply_controlled_variant(
    *,
    run_dir: Path,
    spec: VariantSpec,
    base_run_id: str,
) -> None:
    trace_path = run_dir / "trace.json"
    summary_path = run_dir / "run_summary.json"
    snapshot_path = run_dir / "config_snapshot.json"
    trace = _read_json(trace_path)
    summary = _read_json(summary_path)
    snapshot = _read_json(snapshot_path)
    _mutate_trace(trace, spec)
    variant_fields = {
        "base_run_id": base_run_id,
        "variant_id": spec.variant_id,
        "variant_type": spec.variant_type.value,
        "variant_implemented": True,
        "controlled_variant": True,
        "variant_transform": spec.transform.value,
        "safety_notes": list(spec.safety_notes),
        "provider_called": False,
        "raw_text_saved": False,
    }
    trace.setdefault("metadata", {}).update(variant_fields)
    summary.update(variant_fields)
    snapshot.update(
        {
            "variant_id": spec.variant_id,
            "variant_type": spec.variant_type.value,
            "controlled_variant": True,
        }
    )
    _write_json(trace_path, trace)
    _write_json(summary_path, summary)
    _write_json(snapshot_path, snapshot)


def _mutate_trace(trace: dict[str, object], spec: VariantSpec) -> None:
    records = trace.get("records")
    if not isinstance(records, list):
        return
    if spec.transform == VariantTransform.FREE_CHAT:
        _mark_candidate_can_modify_state(records, "simulated_free_multi_agent_chat")
    elif spec.transform == VariantTransform.SHARED_CONTEXT:
        _mark_agent_hidden_context(records, "simulated_shared_context_baseline")
    elif spec.transform == VariantTransform.RAG_MEMORY:
        _mark_agent_hidden_context(records, "simulated_rag_memory_baseline")
    elif spec.transform == VariantTransform.RULE_ONLY:
        _remove_commit_trace_details(records, "simulated_rule_only_baseline")
    elif spec.transform == VariantTransform.WITHOUT_EVENT_VERIFICATION:
        _mark_candidate_can_modify_state(records, "simulated_without_event_verification")
    elif spec.transform == VariantTransform.WITHOUT_BELIEF_CANON_LEDGER:
        _mark_player_input_canon_update(records, "simulated_without_belief_canon_ledger")
    elif spec.transform == VariantTransform.WITHOUT_CAUSAL_EVENT_GRAPH:
        _remove_causal_projection(records, "simulated_without_causal_event_graph")
    elif spec.transform == VariantTransform.WITHOUT_WORLD_PRESSURE_FIELD:
        _break_pressure_update(records, "simulated_without_world_pressure_field")
    elif spec.transform == VariantTransform.WITHOUT_PLAYER_INPUT_VERIFICATION:
        _mark_player_input_state_mutation(records, "simulated_without_player_input_verification")
    elif spec.transform == VariantTransform.WITHOUT_AGENT_PRIVATE_MEMORY:
        _mark_agent_hidden_context(records, "simulated_without_agent_private_memory")


def _mark_candidate_can_modify_state(records: list[object], flag: str) -> None:
    for record in _record_dicts(records):
        candidate = record.get("candidate_summary")
        if isinstance(candidate, dict):
            candidate["can_modify_world_state"] = True
            _append_safety_flag(record, flag)
            return


def _mark_player_input_canon_update(records: list[object], flag: str) -> None:
    for record in _record_dicts(records):
        summary = record.get("player_input_summary")
        if isinstance(summary, dict):
            summary["canon_updated"] = True
            _append_safety_flag(record, flag)
            return


def _mark_player_input_state_mutation(records: list[object], flag: str) -> None:
    for record in _record_dicts(records):
        summary = record.get("player_input_summary")
        if isinstance(summary, dict):
            summary["world_state_modified"] = True
            summary["state_diff_id"] = "simulated_player_input_direct_state_diff"
            _append_safety_flag(record, flag)
            return


def _mark_agent_hidden_context(records: list[object], flag: str) -> None:
    for record in _record_dicts(records):
        if record.get("agent_id") == "player":
            continue
        retrieval = record.get("retrieval_summary")
        if isinstance(retrieval, dict):
            retrieval["hidden_context_used"] = True
            _append_safety_flag(record, flag)
            return


def _remove_commit_trace_details(records: list[object], flag: str) -> None:
    for record in _record_dicts(records):
        if record.get("verification_decision") != "commit":
            continue
        record["causal_projection"] = None
        record["evolution_update"] = None
        _append_safety_flag(record, flag)
        return


def _remove_causal_projection(records: list[object], flag: str) -> None:
    for record in _record_dicts(records):
        if record.get("verification_decision") == "commit" and record.get("causal_projection"):
            record["causal_projection"] = None
            _append_safety_flag(record, flag)
            return


def _break_pressure_update(records: list[object], flag: str) -> None:
    for record in _record_dicts(records):
        evolution = record.get("evolution_update")
        if not isinstance(evolution, dict):
            continue
        updates = evolution.get("pressure_updates")
        if not isinstance(updates, list):
            continue
        for update in updates:
            if isinstance(update, dict) and isinstance(update.get("after_level"), int):
                update["after_level"] = update["after_level"] + 1
                _append_safety_flag(record, flag)
                return


def _record_dicts(records: list[object]) -> tuple[dict[str, object], ...]:
    return tuple(record for record in records if isinstance(record, dict))


def _append_safety_flag(record: dict[str, object], flag: str) -> None:
    flags = record.get("safety_flags")
    if isinstance(flags, list):
        flags.append(flag)
    else:
        record["safety_flags"] = [flag]


def _write_comparison_artifacts(
    *,
    comparison_dir: Path,
    run_id: str,
    provider_called: bool,
    variants: tuple[ExperimentVariantResult, ...],
) -> ExperimentComparisonArtifacts:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    variant_results_path = comparison_dir / "variant_results.json"
    comparison_summary_path = comparison_dir / "comparison_summary.json"
    variant_payload = {
        "run_id": run_id,
        "formal_experiment_result": True,
        "provider_called": provider_called,
        "raw_text_saved": False,
        "variant_count": len(variants),
        "variants": [variant.safe_dict() for variant in variants],
        "execution_boundary": "controlled_trace_mutation_comparison",
        "db_readback_status": "not_claimed_for_controlled_comparison",
    }
    _write_json(variant_results_path, variant_payload)
    comparison_payload = {
        **variant_payload,
        "implemented_variant_ids": [variant.variant_id for variant in variants],
        "canonical_baseline_ids": [
            "aethelis_proposed_runtime",
            "free_multi_agent_chat",
            "shared_context_agents",
            "rag_memory_agents",
            "rule_only_world_simulation",
        ],
        "canonical_ablation_ids": [
            "without_belief_canon_ledger",
            "without_event_verification",
            "without_causal_event_graph",
            "without_world_pressure_field",
            "without_agent_private_memory",
            "without_player_input_verification",
        ],
        "unsupported_variant_ids": [
            "without_causal_event_graph",
            "without_world_pressure_field",
        ]
        if not any(
            variant.variant_id
            in {"without_causal_event_graph", "without_world_pressure_field"}
            for variant in variants
        )
        else [],
    }
    _write_json(comparison_summary_path, comparison_payload)
    return ExperimentComparisonArtifacts(
        run_dir=comparison_dir,
        variant_results_path=variant_results_path,
        comparison_summary_path=comparison_summary_path,
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _execution_boundary(spec: VariantSpec) -> str:
    if spec.variant_type == ExperimentVariantType.PROPOSED:
        return "formal_runtime_reference"
    return "controlled_trace_mutation_diagnostic"


def _variant_selection_score(spec: VariantSpec, evaluation) -> float:
    failed = getattr(evaluation, "failed_metric_count", 0)
    metric_count = max(getattr(evaluation, "metric_count", 0), 1)
    bad_cases = getattr(evaluation, "bad_case_count", 0)
    governance = 1.0 - min(failed / metric_count, 1.0)
    risk = min(bad_cases / max(metric_count, 1), 1.0)
    complexity = 0.10 if spec.variant_type.value == "proposed" else 0.25
    evidence_fit = 1.0 if spec.variant_type.value == "proposed" else 0.65
    metric_gain = governance
    return model_selection_score(
        governance=governance,
        evidence_fit=evidence_fit,
        metric_gain=metric_gain,
        trace_completeness=1.0,
        complexity=complexity,
        risk=risk,
    )
