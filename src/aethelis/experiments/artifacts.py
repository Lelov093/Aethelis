from __future__ import annotations

import json
from pathlib import Path

from aethelis.schemas.experiment import (
    FormalExperimentArtifacts,
    FormalExperimentConfig,
    GeneratedExperimentPlan,
)
from aethelis.schemas.run import WorldRunResult
from aethelis.trace.formal import build_world_run_formal_experiment_trace


def write_formal_experiment_artifacts(
    *,
    result: WorldRunResult,
    config: FormalExperimentConfig,
    plan: GeneratedExperimentPlan,
    seed_id: str,
    runs_dir: Path = Path("runs"),
) -> FormalExperimentArtifacts:
    run_dir = (runs_dir / config.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.json"
    run_summary_path = run_dir / "run_summary.json"
    config_snapshot_path = run_dir / "config_snapshot.json"

    trace = build_world_run_formal_experiment_trace(
        result,
        seed_id=seed_id,
        metadata={
            "run_id": config.run_id,
            "mode": config.mode.value,
            "plan_source": plan.plan_source,
            "step_count": plan.step_count,
            "scenario_distribution": plan.scenario_distribution,
            "random_seed": config.random_seed,
            "provider_mode": config.provider_mode.value,
            "baseline_mode": config.baseline_mode.value,
            "baseline_implemented": config.baseline_implemented,
            "ablation_mode": config.ablation_mode.value,
            "ablation_implemented": config.ablation_implemented,
            "wrote_runs": True,
            "wrote_reports": False,
            "raw_text_saved": False,
            "provider_called": result.provider_called,
        },
    )
    trace_path.write_text(
        json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_summary_path.write_text(
        json.dumps(
            _run_summary(result=result, config=config, plan=plan, artifacts_dir=run_dir),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    config_snapshot_path.write_text(
        json.dumps(config.safe_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return FormalExperimentArtifacts(
        run_dir=run_dir,
        trace_path=trace_path,
        run_summary_path=run_summary_path,
        config_snapshot_path=config_snapshot_path,
    )


def _run_summary(
    *,
    result: WorldRunResult,
    config: FormalExperimentConfig,
    plan: GeneratedExperimentPlan,
    artifacts_dir: Path,
) -> dict[str, object]:
    return {
        "run_id": config.run_id,
        "formal_experiment_result": True,
        "mode": config.mode.value,
        "seed_id": result.seed_id,
        "step_count": result.step_count,
        "plan_source": plan.plan_source,
        "scenario_distribution": plan.scenario_distribution,
        "random_seed": config.random_seed,
        "provider_mode": config.provider_mode.value,
        "provider_called": result.provider_called,
        "raw_text_saved": False,
        "wrote_runs": True,
        "wrote_reports": False,
        "baseline_mode": config.baseline_mode.value,
        "baseline_implemented": config.baseline_implemented,
        "ablation_mode": config.ablation_mode.value,
        "ablation_implemented": config.ablation_implemented,
        "state_diff_applied_count": result.state_diff_applied_count,
        "committed_event_count": result.committed_event_count,
        "rejected_count": result.rejected_count,
        "revise_count": result.revise_count,
        "pending_gate_count": result.pending_gate_count,
        "final_state_summary": result.final_state_summary,
        "final_evolution_state_summary": result.final_evolution_state_summary,
        "artifacts_dir": str(artifacts_dir),
    }
