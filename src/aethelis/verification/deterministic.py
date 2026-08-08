from __future__ import annotations

from dataclasses import dataclass

from aethelis.agents.context import CognitionContext, ObservationContext
from aethelis.runtime.scenario_matrix import (
    VerifierRulePack,
    get_verifier_rule_pack,
    get_verifier_rule_pack_by_id,
)
from aethelis.schemas.events import (
    ActionProposal,
    EventCandidate,
    VerificationDecision,
    VerificationResult,
)
from aethelis.schemas.seed import SeedBundle
from aethelis.verification.contracts import (
    VerifierContext,
    VerifierRegistry,
    rule_result,
)


@dataclass(frozen=True)
class DeterministicVerifier:
    verifier_name: str = "deterministic_governance_v0"

    def verify(
        self,
        *,
        bundle: SeedBundle,
        observation: ObservationContext,
        cognition: CognitionContext,
        proposal: ActionProposal,
        candidate: EventCandidate,
        scenario_id: str,
    ) -> VerificationResult:
        context = VerifierContext(
            bundle=bundle,
            scenario_id=scenario_id,
            proposal=proposal,
            candidate=candidate,
            observation=observation,
            cognition=cognition,
        )
        return self.verify_context(context)

    def verify_context(self, context: VerifierContext) -> VerificationResult:
        rules = (
            _candidate_from_action_proposal,
            _actor_matches_agent,
            _agent_location_matches_observation,
            _target_location_exists,
            _proposal_location_matches_observation,
            _target_entity_exists,
            _target_visible_to_agent,
            _private_belief_allowed_for_self,
            _resource_update_targets_known_resource,
            _scenario_governance_rule,
        )
        return VerifierRegistry(verifier_name=self.verifier_name).verify(context, rules)


def _candidate_from_action_proposal(context: VerifierContext):
    passed = context.candidate.source_action_proposal_id == context.proposal.id
    return rule_result(
        "candidate_from_action_proposal",
        passed,
        "EventCandidate must reference the ActionProposal that produced it.",
    )


def _actor_matches_agent(context: VerifierContext):
    if context.cognition is None:
        return rule_result("actor_matches_agent", False, "Cognition context is required.")
    passed = (
        context.candidate.actor_agent_id
        == context.cognition.agent.id
        == context.proposal.proposer_agent_id
    )
    return rule_result(
        "actor_matches_agent",
        passed,
        "Candidate actor and proposal proposer must match the selected agent.",
    )


def _agent_location_matches_observation(context: VerifierContext):
    if context.observation is None or context.cognition is None:
        return rule_result(
            "agent_location_matches_observation",
            False,
            "Observation and cognition context are required.",
        )
    passed = context.cognition.agent.current_location_id == context.observation.location.id
    return rule_result(
        "agent_location_matches_observation",
        passed,
        "Agent current_location_id must match observation location.",
    )


def _target_location_exists(context: VerifierContext):
    if context.proposal.target_location_id is None:
        return rule_result(
            "target_location_exists",
            True,
            "Proposal does not require a target location.",
        )
    passed = any(
        location.id == context.proposal.target_location_id
        for location in context.bundle.world.locations
    )
    return rule_result(
        "target_location_exists",
        passed,
        "Proposal target_location_id must exist in WorldState.",
    )


def _proposal_location_matches_observation(context: VerifierContext):
    if context.observation is None:
        return rule_result(
            "proposal_location_matches_observation",
            False,
            "Observation context is required.",
        )
    if context.proposal.target_location_id is None:
        return rule_result(
            "proposal_location_matches_observation",
            True,
            "Proposal does not require a target location.",
        )
    passed = context.proposal.target_location_id == context.observation.location.id
    return rule_result(
        "proposal_location_matches_observation",
        passed,
        "Proposal target_location_id must match the observed local location.",
    )


def _target_entity_exists(context: VerifierContext):
    passed = all(
        any(entity.id == entity_id for entity in context.bundle.world.entities)
        for entity_id in context.candidate.involved_entity_ids
    )
    return rule_result(
        "target_entity_exists",
        passed,
        "All target entities must exist in the seed.",
    )


def _target_visible_to_agent(context: VerifierContext):
    if context.observation is None:
        return rule_result("target_visible_to_agent", False, "Observation context is required.")
    visible_entity_ids = {entity.id for entity in context.observation.visible_entities}
    passed = all(
        entity_id in visible_entity_ids for entity_id in context.candidate.involved_entity_ids
    )
    return rule_result(
        "target_visible_to_agent",
        passed,
        "Agent observation must include every target entity.",
    )


def _private_belief_allowed_for_self(context: VerifierContext):
    if context.cognition is None:
        return rule_result(
            "private_belief_allowed_for_self",
            False,
            "Cognition context is required.",
        )
    passed = all(
        belief.owner_agent_id == context.cognition.agent.id
        for belief in context.cognition.owned_beliefs
    )
    return rule_result(
        "private_belief_allowed_for_self",
        passed,
        "Only the selected agent's own beliefs can enter cognition context.",
    )


def _resource_update_targets_known_resource(context: VerifierContext):
    pack = _rule_pack_for_scenario(context.scenario_id)
    if pack is None or pack.rule_kind != "resource_quantity_commit":
        return rule_result(
            "resource_update_targets_known_resource",
            True,
            "Scenario does not require a resource quantity update.",
        )
    passed = any(resource.id == pack.resource_id for resource in context.bundle.world.resources)
    return rule_result(
        "resource_update_targets_known_resource",
        passed,
        "Resource quantity update scenarios must target a known resource.",
    )


def _scenario_governance_rule(context: VerifierContext):
    pack = _rule_pack_for_scenario(context.scenario_id)
    if pack is None:
        return rule_result(
            "unsupported_scenario_governance",
            False,
            f"Unsupported deterministic verification scenario: {context.scenario_id}",
            suggested_decision=VerificationDecision.REJECT,
            risk_flags=("unsupported_scenario",),
        )
    if pack.rule_kind == "target_match":
        passed = (
            context.proposal.intent == pack.intent
            and context.proposal.target_location_id == pack.target_location_id
            and all(
                entity_id in context.proposal.target_entity_ids
                for entity_id in pack.target_entity_ids
            )
        )
        return rule_result(
            pack.rule_id,
            passed,
            pack.message,
        )
    if pack.rule_kind == "false_belief_reject":
        passed = _canon_fact_targets_expected_object(context.bundle, pack)
        return rule_result(
            pack.rule_id,
            passed,
            pack.message,
            suggested_decision=pack.suggested_decision,
            risk_flags=pack.risk_flags,
            reason=pack.reason,
        )
    if pack.rule_kind == "resource_quantity_commit":
        passed = context.proposal.intent == pack.intent and (
            context.proposal.target_location_id == pack.expected_location_id
        )
        return rule_result(
            pack.rule_id,
            passed,
            pack.message,
        )
    if pack.rule_kind in {"gated_access", "malformed_action"}:
        return rule_result(
            pack.rule_id,
            False,
            pack.message,
            suggested_decision=pack.suggested_decision,
            risk_flags=pack.risk_flags,
            reason=pack.reason,
        )
    return rule_result(
        "unsupported_scenario_governance",
        False,
        f"Unsupported deterministic verification scenario: {context.scenario_id}",
        suggested_decision=VerificationDecision.REJECT,
        risk_flags=("unsupported_scenario",),
    )


def _rule_pack_for_scenario(scenario_id: str) -> VerifierRulePack | None:
    if scenario_id == "search_archive":
        return get_verifier_rule_pack_by_id("rule_pack_mistgate_false_belief_reject")
    try:
        return get_verifier_rule_pack(scenario_id)
    except ValueError:
        return None


def _canon_fact_targets_expected_object(bundle: SeedBundle, pack: VerifierRulePack) -> bool:
    return any(
        fact.id == pack.canon_fact_id and fact.object_ids == pack.canon_object_ids
        for fact in bundle.world.canon_facts
    )
