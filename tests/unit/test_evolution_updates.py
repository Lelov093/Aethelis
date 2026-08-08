from __future__ import annotations

from pathlib import Path

from aethelis.evolution import (
    DeterministicEvolutionBuilder,
    append_applied_evolution_update,
    evolution_state_safe_context,
)
from aethelis.runtime.single_step import run_single_step
from aethelis.schemas.common import ConfidenceBand
from aethelis.schemas.agents import RelationshipKind, RelationshipRecord
from aethelis.schemas.events import VerificationCheck, VerificationDecision, VerificationResult
from aethelis.schemas.evolution import (
    BeliefUpdateSummary,
    CausalEventGraphSummary,
    CausalEventNode,
    CausalNodeType,
    EvolutionRuntimeState,
    EvolutionUpdateSummary,
    MemoryUpdateSummary,
    RelationshipUpdateSummary,
)
from aethelis.schemas.ledger import BeliefTruthStatus, MemoryKind
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator

VALID_SEED = Path("seeds/mistgate_v01")
HARBOR_SEED = Path("seeds/harbor_lantern_v01")


def test_commit_evolution_summary_contains_minimal_updates() -> None:
    bundle = _load_valid_bundle()
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        settings=object(),
        apply=True,
    )
    assert result.verification_result is not None
    assert result.committed_event is not None

    summary = DeterministicEvolutionBuilder().build_for_step(
        bundle=bundle,
        step_id="step_ivo_fixture_commit",
        scenario_id=result.scenario_id,
        agent_id=result.agent_id,
        decision=result.verification_result.decision,
        committed_event_id=result.committed_event.id,
        state_diff_id=result.committed_event.state_diff.id,
        verification_result_id=result.verification_result.id,
        event_candidate_id=result.event_candidate.id if result.event_candidate else None,
        state_diff_applied=result.state_diff_applied,
    )

    assert summary is not None
    assert summary.decision == VerificationDecision.COMMIT
    assert summary.applied_update_count == 4
    assert summary.world_state_updated is True
    assert summary.canon_updated is False
    assert summary.causal_graph is not None
    assert len(summary.causal_graph.nodes) == 4
    assert len(summary.causal_graph.edges) == 3
    assert summary.pressure_updates[0].pressure_type == "regulator_instability"
    assert summary.pressure_updates[0].applied is True
    assert summary.pressure_updates[0].source_state_diff_id == result.committed_event.state_diff.id
    assert summary.pressure_updates[0].governance_basis == "commit_applied_state_diff"
    assert summary.belief_updates[0].source_belief_id == "belief_ivo_key_in_safe"
    assert summary.belief_updates[0].source_state_diff_id == result.committed_event.state_diff.id
    assert summary.belief_updates[0].canon_updated is False
    assert summary.memory_updates[0].source_event_id == result.committed_event.id
    assert summary.memory_updates[0].source_state_diff_id == result.committed_event.state_diff.id
    assert summary.memory_updates[0].reason
    assert summary.relationship_updates[0].relationship_id == "rel_ivo_mira"
    assert (
        summary.relationship_updates[0].source_state_diff_id == result.committed_event.state_diff.id
    )
    assert summary.relationship_updates[0].trust_after <= 5

    safe = summary.safe_dict()
    assert safe["belief_updates"][0]["canon_updated"] is False
    assert safe["belief_updates"][0]["governance_basis"] == "commit_applied_state_diff"
    assert safe["memory_updates"][0]["reason"].startswith(
        "Committed event creates a bounded observation memory signal."
    )
    assert "retained=" in safe["memory_updates"][0]["reason"]
    assert "reinforcement=" in safe["memory_updates"][0]["reason"]
    assert "summary" not in safe["memory_updates"][0]
    assert summary.opportunity_route == "record"
    assert summary.opportunity_score is not None
    assert summary.causal_graph.edges[-1].confidence >= 0.55


def test_non_commit_evolution_summary_is_trace_only_pressure_signal() -> None:
    bundle = _load_valid_bundle()
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="player",
        scenario_id="player_claim_key_in_hand",
        settings=object(),
        apply=True,
    )
    assert result.verification_result is not None

    summary = DeterministicEvolutionBuilder().build_for_step(
        bundle=bundle,
        step_id="step_player_claim_reject",
        scenario_id=result.scenario_id,
        agent_id=result.agent_id,
        decision=result.verification_result.decision,
        committed_event_id=None,
        state_diff_id=None,
        verification_result_id=result.verification_result.id,
        event_candidate_id=result.event_candidate.id if result.event_candidate else None,
        state_diff_applied=False,
    )

    assert summary is not None
    assert summary.applied_update_count == 0
    assert summary.world_state_updated is False
    assert summary.causal_graph is None
    assert summary.belief_updates == ()
    assert summary.memory_updates == ()
    assert summary.relationship_updates == ()
    assert summary.pressure_updates[0].pressure_type == "rumor_spread"
    assert summary.pressure_updates[0].applied is False
    assert summary.pressure_updates[0].governance_basis == "non_commit_trace_only"
    assert summary.pressure_updates[0].source_state_diff_id is None


def test_runtime_state_accepts_only_applied_commit_updates() -> None:
    bundle = _load_valid_bundle()
    commit_result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        settings=object(),
        apply=True,
    )
    reject_result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="player",
        scenario_id="player_claim_key_in_hand",
        settings=object(),
        apply=True,
    )
    assert commit_result.verification_result is not None
    assert commit_result.committed_event is not None
    assert reject_result.verification_result is not None

    commit_update = DeterministicEvolutionBuilder().build_for_step(
        bundle=bundle,
        step_id="step_ivo_fixture_commit",
        scenario_id=commit_result.scenario_id,
        agent_id=commit_result.agent_id,
        decision=commit_result.verification_result.decision,
        committed_event_id=commit_result.committed_event.id,
        state_diff_id=commit_result.committed_event.state_diff.id,
        verification_result_id=commit_result.verification_result.id,
        event_candidate_id=(
            commit_result.event_candidate.id if commit_result.event_candidate else None
        ),
        state_diff_applied=commit_result.state_diff_applied,
    )
    reject_update = DeterministicEvolutionBuilder().build_for_step(
        bundle=bundle,
        step_id="step_player_claim_reject",
        scenario_id=reject_result.scenario_id,
        agent_id=reject_result.agent_id,
        decision=reject_result.verification_result.decision,
        committed_event_id=None,
        state_diff_id=None,
        verification_result_id=reject_result.verification_result.id,
        event_candidate_id=(
            reject_result.event_candidate.id if reject_result.event_candidate else None
        ),
        state_diff_applied=False,
    )

    state = EvolutionRuntimeState()
    state = append_applied_evolution_update(state, reject_update)

    assert state.safe_summary()["pressure_update_count"] == 0
    assert state.cognitive_runtime_summary()["belief_update_count"] == 0
    assert state.cognitive_runtime_summary()["memory_signal_count"] == 0
    assert state.cognitive_runtime_summary()["relationship_signal_count"] == 0

    state = append_applied_evolution_update(state, commit_update)
    safe_context = evolution_state_safe_context(state)

    assert safe_context["causal_node_count"] == 4
    assert safe_context["causal_edge_count"] == 3
    assert safe_context["causal_runtime_summary"]["latest_committed_event_ids"] == [
        commit_result.committed_event.id
    ]
    assert safe_context["causal_runtime_summary"]["causal_update_count"] == 1
    assert safe_context["pressure_update_count"] == 1
    assert safe_context["pressure_runtime_summary"]["pressure_keys"] == ["regulator_instability"]
    assert safe_context["pressure_runtime_summary"]["latest_pressure_levels"] == [
        {
            "pressure_type": "regulator_instability",
            "after_level": 7,
        }
    ]
    assert safe_context["belief_update_count"] == 1
    assert safe_context["memory_update_count"] == 1
    assert safe_context["relationship_update_count"] == 1
    assert safe_context["cognitive_runtime_summary"]["belief_update_count"] == 1
    assert safe_context["cognitive_runtime_summary"]["memory_signal_count"] == 1
    assert safe_context["cognitive_runtime_summary"]["relationship_signal_count"] == 1
    assert safe_context["cognitive_runtime_summary"]["agent_belief_update_counts"] == {"ivo": 1}
    assert safe_context["cognitive_runtime_summary"]["agent_memory_signal_counts"] == {"ivo": 1}
    assert safe_context["cognitive_runtime_summary"]["relationship_signal_counts"] == {
        "rel_ivo_mira": 1
    }
    assert safe_context["latest_pressure_levels"] == [
        {
            "pressure_type": "regulator_instability",
            "after_level": 7,
            "source_step_id": "step_ivo_fixture_commit",
        }
    ]
    assert "belief_ivo_key_in_safe" not in str(safe_context)
    assert "secret_" not in str(safe_context)


def test_dry_run_commit_update_does_not_enter_runtime_state() -> None:
    bundle = _load_valid_bundle()
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        settings=object(),
        apply=False,
    )
    assert result.verification_result is not None
    assert result.committed_event is not None

    update = DeterministicEvolutionBuilder().build_for_step(
        bundle=bundle,
        step_id="step_ivo_fixture_commit",
        scenario_id=result.scenario_id,
        agent_id=result.agent_id,
        decision=result.verification_result.decision,
        committed_event_id=result.committed_event.id,
        state_diff_id=result.committed_event.state_diff.id,
        verification_result_id=result.verification_result.id,
        event_candidate_id=result.event_candidate.id if result.event_candidate else None,
        state_diff_applied=result.state_diff_applied,
    )

    state = append_applied_evolution_update(EvolutionRuntimeState(), update)

    assert state.safe_summary()["causal_node_count"] == 0
    assert state.safe_summary()["pressure_update_count"] == 0
    assert state.cognitive_runtime_summary()["belief_update_count"] == 0


def test_harbor_commit_evolution_uses_harbor_targets() -> None:
    bundle = _load_valid_bundle(HARBOR_SEED)
    result = run_single_step(
        seed_path=HARBOR_SEED,
        agent_id="elin",
        scenario_id="elin_inspect_cargo_manifest_fixture",
        settings=object(),
        apply=True,
    )
    assert result.verification_result is not None
    assert result.committed_event is not None

    summary = DeterministicEvolutionBuilder().build_for_step(
        bundle=bundle,
        step_id="step_elin_manifest_commit",
        scenario_id=result.scenario_id,
        agent_id=result.agent_id,
        decision=result.verification_result.decision,
        committed_event_id=result.committed_event.id,
        state_diff_id=result.committed_event.state_diff.id,
        verification_result_id=result.verification_result.id,
        event_candidate_id=result.event_candidate.id if result.event_candidate else None,
        state_diff_applied=result.state_diff_applied,
    )

    assert summary is not None
    assert summary.causal_graph is not None
    assert summary.pressure_updates[0].pressure_type == "gate_access"
    assert summary.memory_updates[0].related_resource_ids == ("harbor_pass",)
    assert summary.memory_updates[0].related_entity_ids == ("cargo_manifest",)
    safe = str(summary.safe_dict())
    assert "harbor_pass" in safe
    assert "calibration_key" not in safe
    assert "workshop_safe" not in safe
    assert "Ivo" not in safe


def test_runtime_state_adds_sequential_committed_event_edge() -> None:
    first = _minimal_commit_update(
        step_id="step_first_commit",
        committed_event_id="committed_first",
    )
    second = _minimal_commit_update(
        step_id="step_second_commit",
        committed_event_id="committed_second",
    )

    state = append_applied_evolution_update(EvolutionRuntimeState(), first)
    state = append_applied_evolution_update(state, second)

    edge_ids = {edge.id for edge in state.causal_edges}
    assert state.causal_runtime_summary()["latest_committed_event_ids"] == [
        "committed_first",
        "committed_second",
    ]
    assert "causal:edge:committed_first:then:committed_second" in edge_ids
    assert state.causal_runtime_summary()["causal_update_count"] == 2


def test_runtime_state_filters_unsafe_cognitive_updates() -> None:
    update = _minimal_commit_update(
        step_id="step_cognitive_filter",
        committed_event_id="committed_cognitive_filter",
    ).model_copy(
        update={
            "applied_update_count": 4,
            "belief_updates": (_belief_update(canon_updated=True),),
            "memory_updates": (_memory_update(applied=False),),
            "relationship_updates": (_relationship_update(applied=False),),
        }
    )

    state = append_applied_evolution_update(EvolutionRuntimeState(), update)

    summary = state.cognitive_runtime_summary()
    assert summary["belief_update_count"] == 0
    assert summary["memory_signal_count"] == 0
    assert summary["relationship_signal_count"] == 0


def test_evolution_updates_vary_with_verification_quality_and_risk() -> None:
    bundle = _load_valid_bundle()
    high = _quality_result("verification_high", passed=True, risk_flags=())
    low = _quality_result("verification_low", passed=False, risk_flags=("boundary_risk",))

    high_summary = DeterministicEvolutionBuilder().build_for_step(
        bundle=bundle,
        step_id="step_high_quality",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        agent_id="ivo",
        decision=VerificationDecision.COMMIT,
        committed_event_id="committed_high_quality",
        state_diff_id="diff_committed_high_quality_calibration_key",
        verification_result_id=high.id,
        event_candidate_id="candidate_high_quality",
        state_diff_applied=True,
        verification_result=high,
    )
    low_summary = DeterministicEvolutionBuilder().build_for_step(
        bundle=bundle,
        step_id="step_low_quality",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        agent_id="ivo",
        decision=VerificationDecision.COMMIT,
        committed_event_id="committed_unsafe_low_quality",
        state_diff_id="diff_committed_low_quality",
        verification_result_id=low.id,
        event_candidate_id="candidate_low_quality",
        state_diff_applied=True,
        verification_result=low,
    )

    assert high_summary is not None
    assert low_summary is not None
    assert high_summary.belief_updates[0].confidence_after.value != low_summary.belief_updates[0].confidence_after.value
    assert "suppression=0.00" in high_summary.memory_updates[0].reason
    assert "suppression=" in low_summary.memory_updates[0].reason
    assert high_summary.relationship_updates[0].trust_delta > low_summary.relationship_updates[0].trust_delta
    assert high_summary.pressure_updates
    assert low_summary.pressure_updates == ()
    assert low_summary.opportunity_route == "blocked"
    assert high_summary.causal_graph is not None
    assert low_summary.causal_graph is not None
    assert high_summary.causal_graph.edges[-1].confidence > low_summary.causal_graph.edges[-1].confidence


def test_relationship_update_selects_ranked_affected_relationship_not_first_shortcut() -> None:
    bundle = _load_valid_bundle()
    extra = RelationshipRecord(
        id="rel_ivo_taren",
        source_agent_id="ivo",
        target_agent_id="taren",
        kind=RelationshipKind.TENSE,
        summary="Ivo has sharper tension with Taren.",
        trust=5,
    )
    relationships = (bundle.agents.relationships[0].model_copy(update={"trust": 1}), extra)
    bundle = bundle.model_copy(
        update={"agents": bundle.agents.model_copy(update={"relationships": relationships})}
    )

    summary = DeterministicEvolutionBuilder().build_for_step(
        bundle=bundle,
        step_id="step_relationship_rank",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        agent_id="ivo",
        decision=VerificationDecision.COMMIT,
        committed_event_id="committed_relationship_rank",
        state_diff_id="diff_committed_relationship_rank_calibration_key",
        verification_result_id="verification_relationship_rank",
        event_candidate_id="candidate_relationship_rank",
        state_diff_applied=True,
    )

    assert summary is not None
    assert summary.relationship_updates[0].relationship_id == "rel_ivo_taren"


def _load_valid_bundle(seed_path: Path = VALID_SEED):
    load_result = SeedLoader().load(seed_path)
    report = SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )
    assert report.success
    assert load_result.bundle is not None
    return load_result.bundle


def _minimal_commit_update(
    *,
    step_id: str,
    committed_event_id: str,
) -> EvolutionUpdateSummary:
    return EvolutionUpdateSummary(
        step_id=step_id,
        scenario_id="ivo_inspect_workshop_safe_fixture",
        decision=VerificationDecision.COMMIT,
        applied_update_count=1,
        causal_graph=CausalEventGraphSummary(
            nodes=(
                CausalEventNode(
                    id=f"causal:event:{committed_event_id}",
                    node_type=CausalNodeType.COMMITTED_EVENT,
                    source_id=committed_event_id,
                    label="Committed test event",
                ),
            )
        ),
        world_state_updated=True,
    )


def _belief_update(*, canon_updated: bool) -> BeliefUpdateSummary:
    return BeliefUpdateSummary(
        id="belief_update:test",
        owner_agent_id="ivo",
        source_belief_id="belief_ivo_key_in_safe",
        source_event_id="committed_cognitive_filter",
        source_state_diff_id="diff_committed_cognitive_filter",
        truth_status_before=BeliefTruthStatus.UNKNOWN,
        truth_status_after=BeliefTruthStatus.PARTIALLY_TRUE,
        confidence_before=ConfidenceBand.MEDIUM,
        confidence_after=ConfidenceBand.HIGH,
        canon_updated=canon_updated,
        reason="Test cognitive filter.",
    )


def _memory_update(*, applied: bool) -> MemoryUpdateSummary:
    return MemoryUpdateSummary(
        id="memory_update:test",
        owner_agent_id="ivo",
        memory_kind=MemoryKind.OBSERVATION,
        source_event_id="committed_cognitive_filter",
        source_state_diff_id="diff_committed_cognitive_filter",
        summary="Test private memory signal.",
        applied=applied,
        reason="Test memory signal.",
    )


def _relationship_update(*, applied: bool) -> RelationshipUpdateSummary:
    return RelationshipUpdateSummary(
        id="relationship_update:test",
        relationship_id="rel_ivo_mira",
        source_agent_id="ivo",
        target_agent_id="mira",
        source_event_id="committed_cognitive_filter",
        source_state_diff_id="diff_committed_cognitive_filter",
        trust_before=0,
        trust_delta=1,
        trust_after=1,
        applied=applied,
        reason="Test relationship signal.",
    )


def _quality_result(
    verification_id: str,
    *,
    passed: bool,
    risk_flags: tuple[str, ...],
) -> VerificationResult:
    return VerificationResult(
        id=verification_id,
        event_candidate_id=f"candidate_{verification_id}",
        decision=VerificationDecision.COMMIT,
        verifier="unit_quality",
        checks=(
            VerificationCheck(name="schema", passed=passed, message="schema"),
            VerificationCheck(name="state_safety", passed=passed, message="state safety"),
        ),
        reasons=("unit quality result",),
        risk_flags=risk_flags,
    )
