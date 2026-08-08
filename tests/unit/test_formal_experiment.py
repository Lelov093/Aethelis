from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aethelis.experiments.runner import build_experiment_plan, load_formal_experiment_config
from aethelis.schemas.experiment import FormalExperimentConfig

STANDARD_EXPERIMENT = Path("configs/standard_experiment_deterministic_regression.yaml")
HARBOR_EXPERIMENT = Path("configs/harbor_lantern_standard_experiment_deterministic_regression.yaml")
REAL_STANDARD_EXPERIMENT = Path("configs/standard_experiment.yaml")
REAL_HARBOR_EXPERIMENT = Path("configs/harbor_lantern_standard_experiment.yaml")


def test_standard_formal_experiment_config_generates_20_step_cycle() -> None:
    config = load_formal_experiment_config(STANDARD_EXPERIMENT)

    plan = build_experiment_plan(config)

    assert config.formal_experiment_result is True
    assert config.provider_mode.value == "no_provider"
    assert config.baseline_implemented is False
    assert config.ablation_implemented is False
    assert plan.plan_source == "scenario_cycle"
    assert plan.step_count == 20
    assert plan.steps[0].step_id == "step_001_ivo_inspect_workshop_safe_fixture"
    assert plan.steps[0].allow_real_llm is False
    assert plan.scenario_distribution["ivo_inspect_workshop_safe_fixture"] == 3
    assert plan.scenario_distribution["mira_search_archive_wrong_key"] == 3
    assert plan.scenario_distribution["selka_consume_stabilizer_part_fixture"] == 3
    assert plan.scenario_distribution["selka_restock_market_credit_fixture"] == 3
    assert plan.scenario_distribution["player_request_open_workshop_safe"] == 2


def test_standard_formal_experiment_config_is_real_provider_first() -> None:
    config = load_formal_experiment_config(REAL_STANDARD_EXPERIMENT)
    plan = build_experiment_plan(config)

    assert config.mode.value == "formal_real_provider"
    assert config.provider_mode.value == "real_provider"
    assert plan.step_count == 20
    assert set(plan.scenario_distribution) == {"inspect_workshop_safe"}
    assert all(step.allow_real_llm is True for step in plan.steps)


def test_harbor_formal_experiment_config_uses_harbor_only_scenarios() -> None:
    config = load_formal_experiment_config(HARBOR_EXPERIMENT)

    plan = build_experiment_plan(config)

    assert config.seed_path == "seeds/harbor_lantern_v01"
    assert config.provider_mode.value == "no_provider"
    assert plan.step_count == 20
    assert set(plan.scenario_distribution) == {
        "elin_inspect_cargo_manifest_fixture",
        "sora_release_relief_crates_fixture",
        "niven_search_lantern_wrong_pass",
        "niven_force_quay_lock",
        "player_claim_harbor_pass",
        "player_request_open_quay_gate",
    }
    assert "ivo_inspect_workshop_safe_fixture" not in plan.scenario_distribution
    assert all(step.allow_real_llm is False for step in plan.steps)


def test_harbor_formal_experiment_config_is_real_provider_first() -> None:
    config = load_formal_experiment_config(REAL_HARBOR_EXPERIMENT)
    plan = build_experiment_plan(config)

    assert config.seed_path == "seeds/harbor_lantern_v01"
    assert config.mode.value == "formal_real_provider"
    assert config.provider_mode.value == "real_provider"
    assert set(plan.scenario_distribution) == {"elin_inspect_cargo_manifest"}
    assert all(step.allow_real_llm is True for step in plan.steps)


@pytest.mark.parametrize("step_count", [19, 51])
def test_formal_experiment_rejects_invalid_step_count(step_count: int) -> None:
    with pytest.raises(ValidationError):
        FormalExperimentConfig(
            run_id="bad_step_count",
            seed_path="seeds/mistgate_v01",
            step_count=step_count,
            scenario_cycle=("ivo_inspect_workshop_safe_fixture",),
        )


def test_formal_deterministic_experiment_rejects_real_provider_scenario() -> None:
    with pytest.raises(ValidationError):
        FormalExperimentConfig(
            run_id="bad_real_provider",
            seed_path="seeds/mistgate_v01",
            mode="formal_deterministic",
            provider_mode="no_provider",
            step_count=20,
            scenario_cycle=("inspect_workshop_safe",),
        )


def test_baseline_and_ablation_modes_default_to_not_implemented() -> None:
    config = FormalExperimentConfig(
        run_id="placeholder_modes",
        seed_path="seeds/mistgate_v01",
        step_count=20,
        baseline_mode="shared_context_agents",
        ablation_mode="without_event_verification",
        scenario_cycle=("ivo_inspect_workshop_safe_fixture",),
    )

    assert config.baseline_implemented is False
    assert config.ablation_implemented is False
    assert config.safe_dict()["baseline_implemented"] is False
    assert config.safe_dict()["ablation_implemented"] is False


def test_baseline_and_ablation_implemented_flags_require_modes() -> None:
    with pytest.raises(ValidationError):
        FormalExperimentConfig(
            run_id="bad_baseline_flag",
            seed_path="seeds/mistgate_v01",
            step_count=20,
            baseline_implemented=True,
            scenario_cycle=("ivo_inspect_workshop_safe_fixture",),
        )
    with pytest.raises(ValidationError):
        FormalExperimentConfig(
            run_id="bad_ablation_flag",
            seed_path="seeds/mistgate_v01",
            step_count=20,
            ablation_implemented=True,
            scenario_cycle=("ivo_inspect_workshop_safe_fixture",),
        )
