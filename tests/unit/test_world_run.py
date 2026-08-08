from __future__ import annotations

import json
from pathlib import Path

import pytest

import aethelis.runtime.world_run as world_run_module
from aethelis.agents.context import build_agent_context
from aethelis.events.conversion import action_proposal_to_event_candidate
from aethelis.runtime.single_step import SingleStepResult, build_committed_event
from aethelis.runtime.state_apply import ControlledStateDiffApplier
from aethelis.runtime.world_run import (
    WorldRunConfigurationError,
    load_run_config,
    run_world,
)
from aethelis.schemas.events import ActionIntent, ActionProposal, VerificationDecision
from aethelis.schemas.run import (
    RunConfig,
    RunMode,
    RunStepPlanItem,
    WorldRunResult,
    WorldRunState,
)
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator
from aethelis.trace.formal import (
    build_world_run_trace_preview,
    validate_formal_trace_file,
    write_world_run_trace_preview,
)
from aethelis.verification.deterministic import DeterministicVerifier

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
STANDARD_RUN = ROOT / "configs" / "standard_run_deterministic_regression.yaml"
REAL_STANDARD_RUN = ROOT / "configs" / "standard_run.yaml"
HARBOR_SEED = ROOT / "seeds" / "harbor_lantern_v01"
HARBOR_RUN = ROOT / "configs" / "harbor_lantern_standard_run_deterministic_regression.yaml"
REAL_HARBOR_RUN = ROOT / "configs" / "harbor_lantern_standard_run.yaml"


def test_run_config_and_world_run_state_schema() -> None:
    config = RunConfig(
        run_id="test_run",
        mode=RunMode.DETERMINISTIC_PREVIEW,
        formal_experiment_result=False,
        allow_real_llm=False,
        dry_run=True,
        apply=False,
        step_plan=(
            RunStepPlanItem(
                step_id="step_mira",
                agent_id="mira",
                actor_type="agent",
                scenario_id="mira_search_archive_wrong_key",
            ),
        ),
    )
    state = WorldRunState(run_id=config.run_id, seed_id="mistgate_v01")

    assert config.mode == RunMode.DETERMINISTIC_PREVIEW
    assert config.formal_experiment_result is False
    assert config.allow_real_llm is False
    assert state.dry_run is True
    assert state.provider_called is False
    assert state.evolution_state.safe_summary()["causal_node_count"] == 0


def test_standard_run_config_is_deterministic_preview() -> None:
    config = load_run_config(STANDARD_RUN)

    assert config.mode == RunMode.DETERMINISTIC_PREVIEW
    assert config.formal_experiment_result is False
    assert config.allow_real_llm is False
    assert config.dry_run is True
    assert config.apply is False
    assert config.activation.mode.value == "static_trace"
    assert config.activation.scoring_version.value == "rule_based_v0"
    assert config.activation.allow_real_llm is False
    assert config.activation.allow_private_belief_scoring is False
    assert {step.scenario_id for step in config.step_plan} == {
        "ivo_inspect_workshop_safe_fixture",
        "mira_search_archive_wrong_key",
        "selka_consume_stabilizer_part_fixture",
        "selka_restock_market_credit_fixture",
        "malformed_or_incomplete_action",
        "unsafe_force_open_safe",
        "player_claim_key_in_hand",
        "player_request_open_workshop_safe",
    }
    assert all(not step.allow_real_llm for step in config.step_plan)


def test_standard_run_config_is_real_provider_preview() -> None:
    config = load_run_config(REAL_STANDARD_RUN)

    assert config.mode == RunMode.REAL_PROVIDER_PREVIEW
    assert config.allow_real_llm is True
    assert config.step_plan[0].scenario_id == "inspect_workshop_safe"
    assert config.step_plan[0].allow_real_llm is True


def test_harbor_standard_run_config_is_real_provider_preview() -> None:
    config = load_run_config(REAL_HARBOR_RUN)

    assert config.mode == RunMode.REAL_PROVIDER_PREVIEW
    assert config.allow_real_llm is True
    assert config.step_plan[0].agent_id == "elin"
    assert config.step_plan[0].scenario_id == "elin_inspect_cargo_manifest"
    assert config.step_plan[0].allow_real_llm is True


def test_run_world_executes_deterministic_plan_without_provider_call(monkeypatch) -> None:
    config = load_run_config(STANDARD_RUN)

    def fail_if_provider_called(*args, **kwargs):
        raise AssertionError("run-world deterministic preview must not call real providers")

    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fail_if_provider_called,
    )

    result = run_world(seed_path=VALID_SEED, config=config)

    assert isinstance(result, WorldRunResult)
    assert result.step_count == 8
    assert result.provider_called is False
    assert result.formal_experiment_result is False
    assert result.wrote_runs is False
    assert result.wrote_reports is False
    assert result.raw_text_saved is False
    assert result.safe_summary()["activation_trace_included"] is True
    assert result.safe_summary()["activation_mode"] == "static_trace"
    assert result.safe_summary()["activation_provider_called"] is False
    assert result.decisions == (
        VerificationDecision.COMMIT,
        VerificationDecision.REJECT,
        VerificationDecision.COMMIT,
        VerificationDecision.COMMIT,
        VerificationDecision.REVISE,
        VerificationDecision.PENDING_GATE,
        VerificationDecision.REJECT,
        VerificationDecision.PENDING_GATE,
    )
    assert result.committed_event_count == 3
    assert result.state_diff_applied is False
    assert result.final_state_summary is not None
    assert result.final_state_summary["discovered_resources"] == []
    assert result.final_evolution_state_summary is not None
    assert result.final_evolution_state_summary["causal_node_count"] == 0
    assert result.final_evolution_state_summary["causal_edge_count"] == 0
    assert result.final_evolution_state_summary["pressure_update_count"] == 0
    assert result.causal_runtime_summary == {
        "causal_node_count": 0,
        "causal_edge_count": 0,
        "latest_committed_event_ids": [],
        "causal_update_count": 0,
    }
    assert result.pressure_runtime_summary == {
        "pressure_update_count": 0,
        "pressure_keys": [],
        "latest_pressure_levels": [],
        "pressure_update_journal_count": 0,
    }
    assert result.cognitive_runtime_summary == {
        "belief_update_count": 0,
        "memory_signal_count": 0,
        "relationship_signal_count": 0,
        "agent_belief_update_counts": {},
        "agent_memory_signal_counts": {},
        "relationship_signal_counts": {},
        "belief_update_journal_count": 0,
        "memory_signal_journal_count": 0,
        "relationship_signal_journal_count": 0,
        "latest_cognitive_update_refs": {
            "belief": [],
            "memory": [],
            "relationship": [],
        },
    }
    assert result.safe_summary()["causal_runtime_summary"] == result.causal_runtime_summary
    assert result.safe_summary()["pressure_runtime_summary"] == result.pressure_runtime_summary
    assert result.safe_summary()["cognitive_runtime_summary"] == (result.cognitive_runtime_summary)
    assert result.final_evolution_state_summary["belief_update_count"] == 0
    assert result.final_evolution_state_summary["memory_update_count"] == 0
    assert result.final_evolution_state_summary["relationship_update_count"] == 0
    assert result.steps[0].committed_event_id is not None
    assert result.steps[0].state_diff_id is not None
    assert result.steps[0].state_transition is not None
    assert result.steps[0].state_transition.applied is False
    assert result.steps[0].causal_projection is not None
    assert result.steps[0].evolution_update is not None
    assert result.steps[0].evolution_update.applied_update_count == 4
    commit_indices = {0, 2, 3}
    assert all(
        (step.committed_event_id is not None) == (index in commit_indices)
        for index, step in enumerate(result.steps)
    )
    assert all(
        (step.state_diff_id is not None) == (index in commit_indices)
        for index, step in enumerate(result.steps)
    )
    assert all(
        (step.state_transition is not None) == (index in commit_indices)
        for index, step in enumerate(result.steps)
    )
    assert all(
        (step.causal_projection is not None) == (index in commit_indices)
        for index, step in enumerate(result.steps)
    )
    assert result.steps[1].evolution_update is not None
    assert result.steps[1].evolution_update.applied_update_count == 0
    assert all(step.activation_result is not None for step in result.steps)
    assert all(
        step.activation_result.candidate_count == 6
        for step in result.steps
        if step.actor_type == "agent" and step.activation_result is not None
    )
    assert all(
        step.activation_result.selected_candidate.agent_id == step.agent_id
        for step in result.steps
        if step.actor_type == "agent" and step.activation_result is not None
    )
    assert all(
        step.proposal_summary is not None for step in result.steps if step.actor_type == "agent"
    )
    assert all(
        step.retrieval_summary is not None for step in result.steps if step.actor_type == "agent"
    )
    assert all(
        step.proposal_source is not None for step in result.steps if step.actor_type == "agent"
    )
    assert all(
        step.candidate_summary is not None for step in result.steps if step.actor_type == "agent"
    )
    assert all(step.candidate_summary is not None for step in result.steps)
    assert result.steps[0].proposal_summary is not None
    assert result.steps[0].proposal_summary.generated_by == "deterministic_fixture"
    assert result.steps[0].proposal_source == "deterministic_fixture"
    assert result.steps[0].retrieval_summary is not None
    assert result.steps[0].retrieval_summary["pressure_context_available"] is True
    assert result.steps[0].retrieval_summary["evolution_context_available"] is True
    assert result.steps[0].proposal_summary.contains_state_diff is False
    assert result.steps[0].candidate_summary is not None
    assert result.steps[0].candidate_summary.can_modify_world_state is False
    assert result.steps[0].candidate_summary.predicted_state_diff_id is None
    assert result.steps[2].state_transition is not None
    assert result.steps[2].state_transition.patches[0].path == (
        "/resource/stabilizer_parts/quantity"
    )
    assert result.steps[2].state_transition.patches[0].before_summary == 3
    assert result.steps[2].state_transition.patches[0].after_summary == 2
    assert result.steps[3].state_transition is not None
    assert result.steps[3].state_transition.patches[0].path == "/resource/market_credit/quantity"
    assert result.steps[3].state_transition.patches[0].before_summary == 5
    assert result.steps[3].state_transition.patches[0].after_summary == 6
    assert result.steps[-1].player_input_summary is not None
    assert result.steps[-2].player_input_summary is not None
    assert result.steps[-2].player_input_summary["route"] == "rejected_claim"
    assert result.steps[-2].player_input_summary["belief_candidate_id"] is not None
    assert result.steps[-2].player_input_summary["canon_updated"] is False
    assert result.steps[-2].player_input_summary["world_state_modified"] is False
    assert result.steps[-1].player_input_summary["route"] == "event_candidate"
    assert result.steps[-1].player_input_summary["event_candidate_id"] is not None
    assert result.steps[-1].player_input_summary["canon_updated"] is False
    assert result.steps[-1].player_input_summary["world_state_modified"] is False
    assert result.steps[-1].decision == VerificationDecision.PENDING_GATE
    assert result.steps[-1].committed_event_id is None
    assert result.steps[-1].state_diff_id is None
    assert [step.agent_id for step in result.steps] == [step.agent_id for step in config.step_plan]
    assert [step.scenario_id for step in result.steps] == [
        step.scenario_id for step in config.step_plan
    ]


def test_harbor_run_world_executes_proposed_runtime_without_provider_call(monkeypatch) -> None:
    config = load_run_config(HARBOR_RUN)

    def fail_if_provider_called(*args, **kwargs):
        raise AssertionError("harbor deterministic preview must not call real providers")

    monkeypatch.setattr(
        "aethelis.llm.openai_compatible.OpenAICompatibleLLMProvider.generate",
        fail_if_provider_called,
    )

    result = run_world(seed_path=HARBOR_SEED, config=config)

    assert result.step_count == 6
    assert result.provider_called is False
    assert result.raw_text_saved is False
    assert result.wrote_runs is False
    assert result.decisions == (
        VerificationDecision.COMMIT,
        VerificationDecision.COMMIT,
        VerificationDecision.REJECT,
        VerificationDecision.PENDING_GATE,
        VerificationDecision.REJECT,
        VerificationDecision.PENDING_GATE,
    )
    assert result.committed_event_count == 2
    assert result.state_diff_applied is False
    assert result.final_state_summary is not None
    assert result.final_state_summary["discovered_resources"] == []
    assert result.steps[0].state_transition is not None
    assert result.steps[0].state_transition.patches[0].target_id == "harbor_pass"
    assert result.steps[1].state_transition is not None
    assert result.steps[1].state_transition.patches[0].target_id == "relief_crates"
    assert result.steps[4].player_input_summary is not None
    assert result.steps[4].player_input_summary["route"] == "rejected_claim"
    assert result.steps[5].player_input_summary is not None
    assert result.steps[5].player_input_summary["route"] == "event_candidate"
    assert "calibration_key" not in str(result.safe_summary())


def test_run_world_apply_applies_only_commit_step(monkeypatch) -> None:
    config = load_run_config(STANDARD_RUN)
    original_run_single_step = world_run_module.run_single_step
    observed_discovered_by_step: list[list[str]] = []
    observed_quantities_by_step: list[tuple[int, int]] = []

    def recording_run_single_step(**kwargs):
        world_state = kwargs["world_state_override"]
        resource = next(
            resource for resource in world_state.resources if resource.id == "calibration_key"
        )
        stabilizer_parts = next(
            resource for resource in world_state.resources if resource.id == "stabilizer_parts"
        )
        market_credit = next(
            resource for resource in world_state.resources if resource.id == "market_credit"
        )
        observed_discovered_by_step.append(list(resource.discovery_state.discovered_by_agent_ids))
        observed_quantities_by_step.append((stabilizer_parts.quantity, market_credit.quantity))
        return original_run_single_step(**kwargs)

    monkeypatch.setattr(world_run_module, "run_single_step", recording_run_single_step)

    result = run_world(seed_path=VALID_SEED, config=config, apply=True)

    assert result.apply_requested is True
    assert result.dry_run is False
    assert result.state_diff_applied is True
    assert result.state_diff_applied_count == 3
    assert result.steps[0].decision == VerificationDecision.COMMIT
    assert result.steps[0].state_diff_applied is True
    assert result.steps[0].state_transition is not None
    assert result.steps[0].state_transition.applied is True
    assert result.steps[0].state_transition.applied_patch_count == 1
    assert result.steps[0].causal_projection is not None
    assert result.steps[0].evolution_update is not None
    assert result.steps[0].evolution_update.world_state_updated is True
    commit_indices = {0, 2, 3}
    assert all(
        step.state_diff_applied == (index in commit_indices)
        for index, step in enumerate(result.steps)
    )
    assert all(
        (step.state_transition is not None) == (index in commit_indices)
        for index, step in enumerate(result.steps)
    )
    assert all(
        (step.causal_projection is not None) == (index in commit_indices)
        for index, step in enumerate(result.steps)
    )
    assert observed_discovered_by_step[0] == []
    assert observed_discovered_by_step[1:] == [
        ["ivo"],
        ["ivo"],
        ["ivo"],
        ["ivo"],
        ["ivo"],
        ["ivo"],
        ["ivo"],
    ]
    assert observed_quantities_by_step == [
        (3, 5),
        (3, 5),
        (3, 5),
        (2, 5),
        (2, 6),
        (2, 6),
        (2, 6),
        (2, 6),
    ]
    assert result.final_state_summary is not None
    assert result.final_state_summary["discovered_resources"] == [
        {"resource_id": "calibration_key", "discovered_by_agent_ids": ["ivo"]}
    ]
    assert result.final_evolution_state_summary is not None
    assert result.final_evolution_state_summary["causal_node_count"] == 12
    assert result.final_evolution_state_summary["causal_edge_count"] == 11
    assert result.final_evolution_state_summary["pressure_update_count"] == 3
    assert result.causal_runtime_summary is not None
    assert result.causal_runtime_summary["causal_node_count"] == 12
    assert result.causal_runtime_summary["causal_edge_count"] == 11
    assert len(result.causal_runtime_summary["latest_committed_event_ids"]) == 3
    second_step_reasons = {
        reason.reason_type: reason
        for reason in result.steps[1].activation_result.selected_candidate.reasons
    }
    assert second_step_reasons["recent_committed_event_relevance"].score == 1
    assert second_step_reasons["causal_open_thread_relevance"].score == 1
    assert result.pressure_runtime_summary is not None
    assert result.pressure_runtime_summary["pressure_keys"] == ["regulator_instability"]
    assert result.pressure_runtime_summary["latest_pressure_levels"] == [
        {
            "pressure_type": "regulator_instability",
            "after_level": 7,
        }
    ]
    assert result.final_evolution_state_summary["belief_update_count"] == 3
    assert result.final_evolution_state_summary["memory_update_count"] == 3
    assert result.final_evolution_state_summary["relationship_update_count"] == 3
    assert result.cognitive_runtime_summary is not None
    assert result.cognitive_runtime_summary["belief_update_count"] == 3
    assert result.cognitive_runtime_summary["memory_signal_count"] == 3
    assert result.cognitive_runtime_summary["relationship_signal_count"] == 3
    assert result.cognitive_runtime_summary["agent_belief_update_counts"] == {
        "ivo": 1,
        "selka": 2,
    }
    assert result.cognitive_runtime_summary["agent_memory_signal_counts"] == {
        "ivo": 1,
        "selka": 2,
    }
    assert result.cognitive_runtime_summary["relationship_signal_counts"] == {
        "rel_ivo_mira": 1,
        "rel_rowan_selka": 2,
    }
    assert "belief_ivo_key_in_safe" not in str(result.final_evolution_state_summary)
    assert "secret_" not in str(result.final_evolution_state_summary)


def test_run_world_apply_only_committed_event_state_diff(monkeypatch) -> None:
    config = RunConfig(
        run_id="fixture_commit_apply_run",
        step_plan=(
            RunStepPlanItem(
                step_id="step_fixture_commit",
                agent_id="mira",
                actor_type="agent",
                scenario_id="mira_search_archive_wrong_key",
                apply=True,
            ),
            RunStepPlanItem(
                step_id="step_fixture_revise",
                agent_id="ivo",
                actor_type="agent",
                scenario_id="malformed_or_incomplete_action",
                apply=True,
            ),
        ),
    )
    original_run_single_step = world_run_module.run_single_step

    def fixture_run_single_step(**kwargs):
        if kwargs["scenario_id"] != "mira_search_archive_wrong_key":
            return original_run_single_step(**kwargs)
        assert kwargs["apply"] is True
        bundle = _load_valid_bundle()
        observation, cognition = build_agent_context(
            bundle,
            agent_id="ivo",
            scenario_id="inspect_workshop_safe",
        )
        proposal = ActionProposal(
            id="proposal_fixture_commit_apply",
            proposer_agent_id="ivo",
            intent=ActionIntent.INVESTIGATE,
            rationale="Fixture commit path for run-world apply boundary.",
            target_location_id="workshop_lane",
            target_entity_ids=("workshop_safe",),
            expected_outcome="Inspect the workshop safe for the calibration key.",
        )
        candidate = action_proposal_to_event_candidate(
            proposal,
            scenario_id="inspect_workshop_safe",
        )
        verification = DeterministicVerifier().verify(
            bundle=bundle,
            observation=observation,
            cognition=cognition,
            proposal=proposal,
            candidate=candidate,
            scenario_id="inspect_workshop_safe",
        )
        committed_event = build_committed_event(
            candidate=candidate,
            verification=verification,
            scenario_id="inspect_workshop_safe",
        )
        assert committed_event is not None
        applied_world_state, apply_report = ControlledStateDiffApplier().apply(
            world_state=bundle.world,
            committed_event=committed_event,
            verification_result=verification,
        )
        return SingleStepResult(
            scenario_id=kwargs["scenario_id"],
            agent_id=kwargs["agent_id"],
            dry_run=False,
            state_diff_applied=apply_report.applied,
            action_proposal=proposal,
            event_candidate=candidate,
            verification_result=verification,
            committed_event=committed_event,
            apply_report=apply_report,
            applied_world_state=applied_world_state,
        )

    monkeypatch.setattr(world_run_module, "run_single_step", fixture_run_single_step)

    result = run_world(seed_path=VALID_SEED, config=config, apply=True)

    assert result.apply_requested is True
    assert result.steps[0].decision == VerificationDecision.COMMIT
    assert result.steps[0].committed_event_id is not None
    assert result.steps[0].state_diff_id is not None
    assert result.steps[0].state_diff_applied is True
    assert result.steps[0].activation_result is not None
    assert result.steps[0].activation_result.action_proposal_generated is False
    assert result.steps[0].activation_result.world_state_modified is False
    assert result.steps[0].apply_report is not None
    assert result.steps[0].apply_report["applied"] is True
    assert result.steps[0].apply_report["applied_patch_count"] == 1
    assert result.steps[1].decision == VerificationDecision.REVISE
    assert result.steps[1].proposal_summary is not None
    assert result.steps[1].proposal_summary.target_entity_ids == ()
    assert result.steps[1].committed_event_id is None
    assert result.steps[1].state_diff_id is None
    assert result.steps[1].state_diff_applied is False
    assert result.state_diff_applied_count == 1
    assert result.committed_event_count == 1
    assert result.provider_called is False
    assert result.final_evolution_state_summary is not None
    assert result.final_evolution_state_summary["causal_node_count"] == 4
    assert result.final_evolution_state_summary["pressure_update_count"] == 1


def test_run_world_rejects_real_llm_scenario_by_default(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_real_llm_run.yaml"
    config_path.write_text(
        """
run_id: bad_real_llm_run
mode: deterministic_preview
formal_experiment_result: false
allow_real_llm: false
dry_run: true
apply: false
step_plan:
  - step_id: step_ivo_real
    agent_id: ivo
    actor_type: agent
    scenario_id: inspect_workshop_safe
    allow_real_llm: false
    apply: false
""",
        encoding="utf-8",
    )
    config = load_run_config(config_path)

    with pytest.raises(WorldRunConfigurationError, match="scenario_requires_real_llm"):
        run_world(seed_path=VALID_SEED, config=config)


def test_run_world_rejects_step_level_real_llm_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_step_flag.yaml"
    config_path.write_text(
        """
run_id: bad_step_flag
mode: deterministic_preview
formal_experiment_result: false
allow_real_llm: false
dry_run: true
apply: false
step_plan:
  - step_id: step_mira
    agent_id: mira
    actor_type: agent
    scenario_id: mira_search_archive_wrong_key
    allow_real_llm: true
    apply: false
""",
        encoding="utf-8",
    )

    with pytest.raises(WorldRunConfigurationError, match="step_real_llm_not_allowed"):
        load_run_config(config_path)


def test_world_run_trace_preview_is_safe_and_valid(tmp_path: Path) -> None:
    config = load_run_config(STANDARD_RUN)
    result = run_world(seed_path=VALID_SEED, config=config)
    trace = build_world_run_trace_preview(result, seed_id="mistgate_v01")

    assert trace.formal_experiment_result is False
    assert trace.runtime_phase == "runtime_foundation_preview"
    assert trace.metadata["wrote_runs"] is False
    assert trace.metadata["wrote_reports"] is False
    assert trace.metadata["raw_text_saved"] is False
    assert trace.metadata["provider_called"] is False
    assert trace.metadata["activation_trace_included"] is True
    assert trace.metadata["activation_mode"] == "static_trace"
    assert trace.metadata["activation_provider_called"] is False
    assert trace.metadata["final_evolution_state_summary"] == result.final_evolution_state_summary
    assert len(trace.records) == 8
    assert all(record.activation_summary is not None for record in trace.records)
    assert trace.records[0].proposal_summary is not None
    assert trace.records[0].proposal_source == "deterministic_fixture"
    assert trace.records[0].retrieval_summary is not None
    assert trace.records[0].candidate_summary is not None
    assert trace.records[0].candidate_summary["can_modify_world_state"] is False
    assert trace.records[0].candidate_summary["predicted_state_diff_id"] is None
    assert trace.records[0].state_transition is not None
    assert trace.records[0].state_transition["applied"] is False
    assert trace.records[0].causal_projection is not None
    assert trace.records[0].evolution_update is not None
    assert trace.records[0].evolution_update["applied_update_count"] == 4
    commit_indices = {0, 2, 3}
    assert all(
        (record.state_transition is not None) == (index in commit_indices)
        for index, record in enumerate(trace.records)
    )
    assert all(
        (record.causal_projection is not None) == (index in commit_indices)
        for index, record in enumerate(trace.records)
    )
    assert trace.records[0].verification_checks
    assert trace.records[0].verification_reasons

    trace_path = write_world_run_trace_preview(
        result,
        tmp_path / "world_run_preview.json",
        seed_id="mistgate_v01",
    )
    report = validate_formal_trace_file(trace_path)
    raw = trace_path.read_text(encoding="utf-8")

    assert report.success is True
    assert report.record_count == 8
    assert report.formal_experiment_result is False
    assert "activation_summary" in raw
    assert "proposal_summary" in raw
    assert "candidate_summary" in raw
    assert "retrieval_summary" in raw
    assert "player_input_summary" in raw
    assert "state_transition" in raw
    assert "causal_projection" in raw
    assert "evolution_update" in raw
    assert "private_summary" not in raw
    assert "secret_" not in raw
    assert "calibration key is in the workshop safe" not in raw.lower()
    assert "canon_key_in_workshop_safe" not in raw
    assert '"raw_llm_text"' not in raw
    assert '"full_raw_text"' not in raw
    assert '"raw_text_content"' not in raw
    assert "authorization" not in raw.lower()
    assert "sk-" not in raw
    assert json.loads(raw)["metadata"]["provider_called"] is False
    assert json.loads(raw)["metadata"]["final_evolution_state_summary"] == (
        result.final_evolution_state_summary
    )
    assert (
        json.loads(raw)["metadata"]["final_evolution_state_summary"]["cognitive_runtime_summary"]
        == result.cognitive_runtime_summary
    )


def _load_valid_bundle():
    load_result = SeedLoader().load(VALID_SEED)
    report = SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )
    assert report.success
    assert load_result.bundle is not None
    return load_result.bundle
