from __future__ import annotations

from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.run import RunStepPlanItem


class ExperimentMode(StrEnum):
    FORMAL_REAL_PROVIDER = "formal_real_provider"
    FORMAL_DETERMINISTIC = "formal_deterministic"


class ProviderMode(StrEnum):
    REAL_PROVIDER = "real_provider"
    NO_PROVIDER = "no_provider"
    DETERMINISTIC = "deterministic"


class BaselineMode(StrEnum):
    NONE = "none"
    AETHELIS_PROPOSED_RUNTIME = "aethelis_proposed_runtime"
    FREE_MULTI_AGENT_CHAT = "free_multi_agent_chat"
    SHARED_CONTEXT_AGENTS = "shared_context_agents"
    RAG_MEMORY_AGENTS = "rag_memory_agents"
    RULE_ONLY_WORLD_SIMULATION = "rule_only_world_simulation"


class AblationMode(StrEnum):
    NONE = "none"
    WITHOUT_EVENT_VERIFICATION = "without_event_verification"
    WITHOUT_BELIEF_CANON_LEDGER = "without_belief_canon_ledger"
    WITHOUT_CAUSAL_EVENT_GRAPH = "without_causal_event_graph"
    WITHOUT_WORLD_PRESSURE_FIELD = "without_world_pressure_field"
    WITHOUT_PLAYER_INPUT_VERIFICATION = "without_player_input_verification"
    WITHOUT_AGENT_PRIVATE_MEMORY = "without_agent_private_memory"


class ExperimentVariantType(StrEnum):
    PROPOSED = "proposed"
    BASELINE = "baseline"
    ABLATION = "ablation"


class FormalExperimentConfig(AethelisModel):
    run_id: Identifier
    seed_path: str = Field(min_length=1)
    mode: ExperimentMode = ExperimentMode.FORMAL_REAL_PROVIDER
    formal_experiment_result: Literal[True] = True
    random_seed: int = Field(default=0, ge=0)
    step_count: int = Field(default=20, ge=20, le=50)
    provider_mode: ProviderMode = ProviderMode.REAL_PROVIDER
    baseline_mode: BaselineMode = BaselineMode.NONE
    baseline_implemented: bool = False
    ablation_mode: AblationMode = AblationMode.NONE
    ablation_implemented: bool = False
    write_artifacts: Literal[True] = True
    apply_state_diffs: bool = False
    comparison_variant_scope: Literal["all", "baselines", "ablations"] = "all"
    scenario_cycle: tuple[Identifier, ...] = Field(default_factory=tuple)
    scenario_plan: tuple[RunStepPlanItem, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_plan_source(self) -> FormalExperimentConfig:
        from aethelis.runtime.scenario_matrix import get_scenario_definition

        if (
            self.mode == ExperimentMode.FORMAL_REAL_PROVIDER
            and self.provider_mode != ProviderMode.REAL_PROVIDER
        ):
            raise ValueError("formal_real_provider requires provider_mode=real_provider")
        if (
            self.mode == ExperimentMode.FORMAL_DETERMINISTIC
            and self.provider_mode == ProviderMode.REAL_PROVIDER
        ):
            raise ValueError("formal_deterministic cannot use provider_mode=real_provider")
        if self.scenario_cycle and self.scenario_plan:
            raise ValueError("Use scenario_cycle or scenario_plan, not both")
        if not self.scenario_cycle and not self.scenario_plan:
            raise ValueError("Formal experiment config requires scenario_cycle or scenario_plan")
        if self.scenario_plan and len(self.scenario_plan) != self.step_count:
            raise ValueError("scenario_plan length must match step_count")
        if self.baseline_implemented and self.baseline_mode == BaselineMode.NONE:
            raise ValueError("baseline_implemented requires a baseline_mode")
        if self.ablation_implemented and self.ablation_mode == AblationMode.NONE:
            raise ValueError("ablation_implemented requires an ablation_mode")
        if self.baseline_implemented and self.ablation_implemented:
            raise ValueError("baseline and ablation variants must be separate runs")
        for scenario_id in self.scenario_cycle:
            scenario = get_scenario_definition(scenario_id)
            if scenario.allows_real_llm and self.provider_mode != ProviderMode.REAL_PROVIDER:
                raise ValueError(
                    f"formal experiment scenario requires real provider: {scenario_id}"
                )
        for step in self.scenario_plan:
            scenario = get_scenario_definition(step.scenario_id)
            if (scenario.allows_real_llm or step.allow_real_llm) and (
                self.provider_mode != ProviderMode.REAL_PROVIDER
            ):
                raise ValueError(f"formal experiment step requires real provider: {step.step_id}")
        return self

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class GeneratedExperimentPlan(AethelisModel):
    run_id: Identifier
    plan_source: Literal["scenario_cycle", "scenario_plan"]
    random_seed: int
    step_count: int
    scenario_distribution: dict[str, int]
    steps: tuple[RunStepPlanItem, ...]

    def safe_summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "plan_source": self.plan_source,
            "random_seed": self.random_seed,
            "step_count": self.step_count,
            "scenario_distribution": self.scenario_distribution,
        }


class FormalExperimentArtifacts(AethelisModel):
    run_dir: Path
    trace_path: Path
    run_summary_path: Path
    config_snapshot_path: Path

    def safe_dict(self) -> dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "trace_path": str(self.trace_path),
            "run_summary_path": str(self.run_summary_path),
            "config_snapshot_path": str(self.config_snapshot_path),
        }


class FormalExperimentRunResult(AethelisModel):
    run_id: Identifier
    formal_experiment_result: Literal[True] = True
    provider_called: bool = False
    raw_text_saved: Literal[False] = False
    wrote_runs: Literal[True] = True
    wrote_reports: Literal[False] = False
    plan: GeneratedExperimentPlan
    artifacts: FormalExperimentArtifacts
    baseline_mode: BaselineMode
    baseline_implemented: bool = False
    ablation_mode: AblationMode
    ablation_implemented: bool = False

    def safe_summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "formal_experiment_result": self.formal_experiment_result,
            "provider_called": self.provider_called,
            "raw_text_saved": self.raw_text_saved,
            "wrote_runs": self.wrote_runs,
            "wrote_reports": self.wrote_reports,
            "plan": self.plan.safe_summary(),
            "artifacts": self.artifacts.safe_dict(),
            "baseline_mode": self.baseline_mode.value,
            "baseline_implemented": self.baseline_implemented,
            "ablation_mode": self.ablation_mode.value,
            "ablation_implemented": self.ablation_implemented,
        }


class ExperimentVariantResult(AethelisModel):
    variant_id: Identifier
    variant_type: ExperimentVariantType
    implemented: bool
    run_id: Identifier
    artifact_path: str
    metric_summary: dict[str, object]
    bad_case_count: int = Field(ge=0)
    safety_notes: tuple[str, ...] = ()

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class ExperimentComparisonArtifacts(AethelisModel):
    run_dir: Path
    variant_results_path: Path
    comparison_summary_path: Path

    def safe_dict(self) -> dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "variant_results_path": str(self.variant_results_path),
            "comparison_summary_path": str(self.comparison_summary_path),
        }


class ExperimentComparisonResult(AethelisModel):
    run_id: Identifier
    formal_experiment_result: Literal[True] = True
    provider_called: bool = False
    raw_text_saved: Literal[False] = False
    variant_count: int = Field(ge=0)
    variants: tuple[ExperimentVariantResult, ...]
    artifacts: ExperimentComparisonArtifacts

    def safe_summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "formal_experiment_result": self.formal_experiment_result,
            "provider_called": self.provider_called,
            "raw_text_saved": self.raw_text_saved,
            "variant_count": self.variant_count,
            "variants": [variant.safe_dict() for variant in self.variants],
            "artifacts": self.artifacts.safe_dict(),
        }


def scenario_distribution(steps: tuple[RunStepPlanItem, ...]) -> dict[str, int]:
    return dict(sorted(Counter(step.scenario_id for step in steps).items()))
