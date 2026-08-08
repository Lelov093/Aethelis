from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from aethelis.experiments.artifacts import write_formal_experiment_artifacts
from aethelis.runtime.scenario_matrix import get_scenario_definition
from aethelis.runtime.world_run import run_world
from aethelis.schemas.experiment import (
    FormalExperimentConfig,
    FormalExperimentRunResult,
    GeneratedExperimentPlan,
    ProviderMode,
    scenario_distribution,
)
from aethelis.schemas.run import RunConfig, RunMode, RunStepPlanItem


class FormalExperimentConfigurationError(ValueError):
    """Safe formal experiment configuration error."""


def load_formal_experiment_config(path: Path) -> FormalExperimentConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FormalExperimentConfigurationError(
            f"{exc.__class__.__name__}: experiment config could not be read"
        ) from None
    if payload is None:
        raise FormalExperimentConfigurationError("Experiment config is empty.")
    try:
        return FormalExperimentConfig.model_validate(payload)
    except ValidationError as exc:
        raise FormalExperimentConfigurationError(
            "ValidationError: experiment config schema invalid"
        ) from exc


def run_formal_experiment(
    *,
    seed_path: Path,
    config: FormalExperimentConfig,
    runs_dir: Path = Path("runs"),
    settings=None,
) -> FormalExperimentRunResult:
    plan = build_experiment_plan(config)
    real_provider = config.provider_mode == ProviderMode.REAL_PROVIDER
    preview_config = RunConfig(
        run_id=config.run_id,
        mode=RunMode.REAL_PROVIDER_PREVIEW if real_provider else RunMode.DETERMINISTIC_PREVIEW,
        formal_experiment_result=False,
        allow_real_llm=real_provider,
        dry_run=True,
        apply=False,
        step_plan=plan.steps,
    )
    result = run_world(
        seed_path=seed_path,
        config=preview_config,
        apply=config.apply_state_diffs,
        settings=settings,
    )
    artifacts = write_formal_experiment_artifacts(
        result=result,
        config=config,
        plan=plan,
        seed_id=seed_path.resolve().name,
        runs_dir=runs_dir,
    )
    return FormalExperimentRunResult(
        run_id=config.run_id,
        provider_called=result.provider_called,
        plan=plan,
        artifacts=artifacts,
        baseline_mode=config.baseline_mode,
        baseline_implemented=config.baseline_implemented,
        ablation_mode=config.ablation_mode,
        ablation_implemented=config.ablation_implemented,
    )


def build_experiment_plan(config: FormalExperimentConfig) -> GeneratedExperimentPlan:
    if config.scenario_plan:
        steps = config.scenario_plan
        plan_source = "scenario_plan"
    else:
        steps = tuple(_cycle_step(config, index) for index in range(config.step_count))
        plan_source = "scenario_cycle"
    return GeneratedExperimentPlan(
        run_id=config.run_id,
        plan_source=plan_source,
        random_seed=config.random_seed,
        step_count=len(steps),
        scenario_distribution=scenario_distribution(steps),
        steps=steps,
    )


def _cycle_step(config: FormalExperimentConfig, index: int) -> RunStepPlanItem:
    scenario_id = config.scenario_cycle[index % len(config.scenario_cycle)]
    scenario = get_scenario_definition(scenario_id)
    return RunStepPlanItem(
        step_id=f"step_{index + 1:03d}_{scenario_id}",
        agent_id=scenario.actor_id,
        actor_type=scenario.actor_type,
        scenario_id=scenario_id,
        allow_real_llm=bool(
            config.provider_mode == ProviderMode.REAL_PROVIDER and scenario.allows_real_llm
        ),
        apply=scenario.allows_apply,
    )
