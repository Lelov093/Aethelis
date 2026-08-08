from __future__ import annotations

import json
from pathlib import Path

from aethelis.agents.action_proposal import ProposalSourceMode, ProviderProposalFailureCode
from aethelis.agents.context import ObservationBuilder, build_agent_context
from aethelis.events.conversion import action_proposal_to_event_candidate
from aethelis.llm.base import LLMResult
from aethelis.llm.structured import _structured_prompt, generate_structured
from aethelis.providers import ProviderAttempt
from aethelis.runtime.player_input import assess_player_claim
from aethelis.runtime.single_step import build_committed_event, build_step_context, run_single_step
from aethelis.schemas.events import (
    ActionIntent,
    ActionProposal,
    VerificationDecision,
)
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator
from aethelis.verification.deterministic import DeterministicVerifier

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
HARBOR_SEED = ROOT / "seeds" / "harbor_lantern_v01"


def load_valid_bundle():
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


def load_valid_harbor_bundle():
    load_result = SeedLoader().load(HARBOR_SEED)
    report = SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )
    assert report.success
    assert load_result.bundle is not None
    return load_result.bundle


def ivo_proposal() -> ActionProposal:
    return ActionProposal(
        id="proposal_inspect_workshop_safe_ivo",
        proposer_agent_id="ivo",
        intent=ActionIntent.INVESTIGATE,
        rationale="Ivo has a lawful private reason to inspect his own workshop safe.",
        target_location_id="workshop_lane",
        target_entity_ids=("workshop_safe",),
        expected_outcome="Inspect the workshop safe for the calibration key.",
    )


def test_ivo_context_contains_own_private_belief_and_not_hidden_canon() -> None:
    bundle = load_valid_bundle()

    context = build_step_context(
        bundle,
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
    )

    belief_ids = {belief.id for belief in context.cognition.owned_beliefs}
    assert "belief_ivo_key_in_safe" in belief_ids
    assert "workshop_safe" in {entity.id for entity in context.observation.visible_entities}
    assert "calibration_key" not in {
        resource.id for resource in context.observation.visible_resources
    }
    assert "canon_key_in_workshop_safe" not in context.prompt
    assert context.retrieval_summary is not None
    assert context.retrieval_summary["own_belief_count"] >= 1
    assert context.retrieval_summary["hidden_context_used"] is False


def test_ivo_prompt_is_compact_and_keeps_required_safety_boundaries() -> None:
    bundle = load_valid_bundle()

    context = build_step_context(
        bundle,
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
    )
    structured_prompt = _structured_prompt(context.prompt, ActionProposal)

    assert len(context.prompt) < 1800
    assert len(structured_prompt) < 3300
    assert "workshop_safe" in context.prompt
    assert "belief_ivo_key_in_safe" in context.prompt
    assert "canon_key_in_workshop_safe" not in context.prompt
    assert "belief_mira_key_in_archive" not in context.prompt
    assert "rowan" not in context.prompt.lower()
    assert "selka" not in context.prompt.lower()
    assert "StateDiff" in context.prompt
    assert "CanonFact" in context.prompt


def test_ivo_candidate_verifies_commit_and_generates_bound_state_diff() -> None:
    bundle = load_valid_bundle()
    observation, cognition = build_agent_context(
        bundle,
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
    )
    proposal = ivo_proposal()
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

    assert verification.decision == VerificationDecision.COMMIT
    assert all(check.passed for check in verification.checks)
    assert committed_event is not None
    assert committed_event.event_candidate_id == candidate.id
    assert committed_event.state_diff.committed_event_id == committed_event.id
    assert committed_event.state_diff.source_action_proposal_id is None
    assert committed_event.state_diff.patches[0].target_id == "calibration_key"


def test_mira_wrong_belief_does_not_generate_key_location_state_diff() -> None:
    bundle = load_valid_bundle()
    observation, cognition = build_agent_context(
        bundle,
        agent_id="mira",
        scenario_id="search_archive",
    )
    proposal = ActionProposal(
        id="proposal_search_archive_mira",
        proposer_agent_id="mira",
        intent=ActionIntent.INVESTIGATE,
        rationale="Mira searches based on her mistaken archive belief.",
        target_location_id="central_archive",
        target_entity_ids=("harmonic_tuner",),
        expected_outcome="Search archive records for the calibration key.",
    )
    candidate = action_proposal_to_event_candidate(proposal, scenario_id="search_archive")
    verification = DeterministicVerifier().verify(
        bundle=bundle,
        observation=observation,
        cognition=cognition,
        proposal=proposal,
        candidate=candidate,
        scenario_id="search_archive",
    )

    assert verification.decision == VerificationDecision.REJECT
    assert (
        build_committed_event(
            candidate=candidate,
            verification=verification,
            scenario_id="search_archive",
        )
        is None
    )
    key_fact = next(
        fact for fact in bundle.world.canon_facts if fact.id == "canon_key_in_workshop_safe"
    )
    assert key_fact.object_ids == ("workshop_safe",)


def test_rowan_context_does_not_include_ivo_private_belief_or_secret() -> None:
    bundle = load_valid_bundle()

    observation, cognition = build_agent_context(
        bundle,
        agent_id="rowan",
        scenario_id="guard_archive",
    )

    prompt_material = f"{observation.prompt_dict()} {cognition.prompt_dict()}"
    assert "belief_ivo_key_in_safe" not in prompt_material
    assert "secret_ivo_safe_suspicion" not in prompt_material
    assert "workshop safe contains the calibration key" not in prompt_material.lower()


def test_product_aligned_observation_boundaries_for_new_agents() -> None:
    bundle = load_valid_bundle()
    builder = ObservationBuilder()

    nara_observation, nara_cognition = build_agent_context(
        bundle,
        agent_id="nara",
        scenario_id="report_market_rumor",
    )
    assert nara_observation.location.id == "market_row"
    assert "rumor_key_passed_market" in {rumor.id for rumor in nara_observation.visible_rumors}
    assert "belief_nara_market_key_rumor" in {belief.id for belief in nara_cognition.owned_beliefs}
    assert "canon_key_in_workshop_safe" not in str(nara_observation.prompt_dict())

    taren_observation, taren_cognition = build_agent_context(
        bundle,
        agent_id="taren",
        scenario_id="observe_old_aqueduct",
    )
    assert taren_observation.location.id == "old_aqueduct"
    assert "dawn_regulator" in {entity.id for entity in taren_observation.visible_entities}
    assert "public_fact_aqueduct_surge" in {
        fact.id for fact in taren_observation.visible_public_facts
    }
    assert "belief_taren_aqueduct_needs_permit" in {
        belief.id for belief in taren_cognition.owned_beliefs
    }

    player_observation = builder.build_observation(
        bundle,
        actor_id="player",
        actor_type="player",
        scenario_id="player_claim_key_in_hand",
    )
    assert player_observation.agent_id == "player"
    assert player_observation.location.id == "council_square"
    assert "rowan" in player_observation.visible_agent_ids


def test_player_claim_is_rejected_and_does_not_enter_canon_or_state_diff() -> None:
    assessment = assess_player_claim(
        claim_id="claim_player_has_key",
        player_id="player",
        claim="The key is in my hand.",
    )

    assert assessment.verification_result.decision == VerificationDecision.REJECT
    assert assessment.routed_input is not None
    assert assessment.routed_input.safe_summary()["route"] == "rejected_claim"
    assert not assessment.canon_updated
    assert assessment.state_diff_id is None
    assert assessment.verification_result.rejected_claim_ids == ("claim_player_has_key",)


def test_run_single_step_reports_proposal_source_and_player_input_summary() -> None:
    deterministic = run_single_step(
        seed_path=VALID_SEED,
        agent_id="mira",
        scenario_id="mira_search_archive_wrong_key",
    )
    player = run_single_step(
        seed_path=VALID_SEED,
        agent_id="player",
        scenario_id="player_claim_key_in_hand",
    )

    assert deterministic.proposal_source == "deterministic_fixture"
    assert deterministic.retrieval_summary is not None
    assert deterministic.retrieval_summary["provider_called"] is False
    assert player.player_input_summary is not None
    assert player.player_input_summary["route"] == "rejected_claim"
    assert player.player_input_summary["canon_updated"] is False


def test_real_provider_scenario_defaults_to_provider_gate_without_settings() -> None:
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
    )

    assert result.provider_called is False
    assert result.provider_mode == "fallback"
    assert result.fallback_used is True
    assert result.fallback_reason == "provider_unavailable"
    assert result.evidence_class == "fallback"
    assert result.failure_code == ProviderProposalFailureCode.PROVIDER_UNAVAILABLE
    assert result.action_proposal is None
    assert result.event_candidate is None
    assert result.verification_result is None
    assert result.committed_event is None


def test_harbor_provider_scenario_prompt_uses_harbor_contract() -> None:
    context = build_step_context(
        load_valid_harbor_bundle(),
        agent_id="elin",
        scenario_id="elin_inspect_cargo_manifest",
    )
    prompt = json.loads(context.prompt)

    assert prompt["required"]["proposer_agent_id"] == "elin"
    assert prompt["required"]["target_location_id"] == "ledger_house"
    assert prompt["required"]["target_entity_ids"] == ["cargo_manifest"]
    assert prompt["required"]["intent"] == "investigate"


def test_harbor_provider_scenario_fallback_metadata_without_settings() -> None:
    result = run_single_step(
        seed_path=HARBOR_SEED,
        agent_id="elin",
        scenario_id="elin_inspect_cargo_manifest",
    )

    assert result.provider_called is False
    assert result.provider_mode == "fallback"
    assert result.fallback_used is True
    assert result.fallback_reason == "provider_unavailable"
    assert result.evidence_class == "fallback"
    assert result.action_proposal is None
    assert result.event_candidate is None
    assert result.committed_event is None


def test_malformed_provider_output_stops_before_event_candidate() -> None:
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
        provider=_FixtureProvider("{not-json"),
        proposal_source=ProposalSourceMode.PROVIDER_STRUCTURED,
        provider_proposals_enabled=True,
        allow_real_provider=True,
    )

    assert result.provider_called is True
    assert result.provider_mode == "real_provider"
    assert result.fallback_used is False
    assert result.fallback_reason == "malformed_output"
    assert result.evidence_class == "internal_failure_path"
    assert result.failure_code == ProviderProposalFailureCode.MALFORMED_OUTPUT
    assert result.action_proposal is None
    assert result.event_candidate is None
    assert result.verification_result is None
    assert result.committed_event is None


def test_valid_provider_output_still_uses_governance_chain() -> None:
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
        provider=_FixtureProvider(valid_provider_action_json()),
        proposal_source=ProposalSourceMode.PROVIDER_STRUCTURED,
        provider_proposals_enabled=True,
        allow_real_provider=True,
        apply=True,
    )

    assert result.provider_called is True
    assert result.provider_mode == "real_provider"
    assert result.fallback_used is False
    assert result.evidence_class == "real_provider"
    assert result.failure_code is None
    assert result.action_proposal is not None
    assert result.event_candidate is not None
    assert result.verification_result is not None
    assert result.verification_result.decision == VerificationDecision.COMMIT
    assert result.committed_event is not None
    assert result.committed_event.state_diff.source_action_proposal_id is None
    assert result.state_diff_applied is True


def test_state_diff_contract_uses_current_world_for_continuous_apply() -> None:
    first = run_single_step(
        seed_path=VALID_SEED,
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
        provider=_FixtureProvider(valid_provider_action_json()),
        proposal_source=ProposalSourceMode.PROVIDER_STRUCTURED,
        provider_proposals_enabled=True,
        allow_real_provider=True,
        apply=True,
    )
    assert first.applied_world_state is not None

    second = run_single_step(
        seed_path=VALID_SEED,
        agent_id="ivo",
        scenario_id="inspect_workshop_safe",
        provider=_FixtureProvider(valid_provider_action_json()),
        proposal_source=ProposalSourceMode.PROVIDER_STRUCTURED,
        provider_proposals_enabled=True,
        allow_real_provider=True,
        apply=True,
        world_state_override=first.applied_world_state,
    )

    assert second.committed_event is not None
    patch = second.committed_event.state_diff.patches[0]
    assert patch.before == ["ivo"]
    assert patch.after == ["ivo"]
    assert second.state_diff_applied is True


def test_player_request_run_step_routes_to_event_candidate_gate_without_mutation() -> None:
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="player",
        scenario_id="player_request_open_workshop_safe",
        apply=True,
    )

    assert result.event_candidate is not None
    assert result.event_candidate.actor_agent_id == "player"
    assert result.event_candidate.involved_location_ids == ("workshop_lane",)
    assert result.event_candidate.involved_entity_ids == ("workshop_safe",)
    assert result.verification_result is not None
    assert result.verification_result.decision == VerificationDecision.PENDING_GATE
    assert result.committed_event is None
    assert result.apply_report is None
    assert result.state_diff_applied is False
    assert result.player_input_summary is not None
    assert result.player_input_summary["route"] == "event_candidate"
    assert result.player_input_summary["canon_updated"] is False
    assert result.player_input_summary["world_state_modified"] is False


class _FixtureProvider:
    provider_name = "fixture_test_provider"

    def __init__(self, content: str) -> None:
        self.content = content

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0):
        return LLMResult(
            content=self.content,
            model="fixture-test-model",
            latency_ms=1,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            attempts=(ProviderAttempt(model="fixture-test-model", success=True, latency_ms=1),),
        )

    def generate_structured(self, prompt, schema_type, *, max_tokens=512, temperature=0.0):
        return generate_structured(
            self,
            prompt,
            schema_type,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def valid_provider_action_json() -> str:
    return (
        '{"id":"proposal_inspect_workshop_safe_ivo",'
        '"proposer_agent_id":"ivo",'
        '"intent":"investigate",'
        '"rationale":"Inspect through fixture provider.",'
        '"target_location_id":"workshop_lane",'
        '"target_entity_ids":["workshop_safe"],'
        '"expected_outcome":"Inspect the workshop safe for the calibration key."}'
    )
