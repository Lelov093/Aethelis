from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from aethelis.agents.action_proposal import (
    ActionProposalEngine,
    ActionProposalGenerationResult,
    ActionProposalSource,
    ProposalBehaviorDecision,
    repair_action_proposal,
)
from aethelis.agents.dynamics import (
    AgentProposalFrame,
    JointIntentCandidate,
    MultiAgentDynamicsSummary,
    MultiAgentProposalBundle,
    ProposalArbitrationRecommendation,
    analyze_multi_agent_dynamics,
    recommend_arbitration,
)
from aethelis.agents.retrieval import (
    ActiveAgentContextFrame,
    CognitionRetriever,
    MultiAgentStepContext,
    RetrievedCognitionContext,
    build_multi_agent_step_context,
)
from aethelis.events.commit import build_committed_event_from_verification
from aethelis.events.conversion import (
    CandidateBehaviorRoute,
    action_proposal_to_event_candidate,
    candidate_behavior_route,
    candidate_gate_verification_result,
)
from aethelis.runtime.state_apply import ControlledStateDiffApplier, StateApplyReport
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.events import (
    ActionProposal,
    CommittedEvent,
    EventCandidate,
    StateDiff,
    VerificationResult,
)
from aethelis.schemas.seed import SeedBundle
from aethelis.schemas.world import WorldState
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator
from aethelis.verification.deterministic import DeterministicVerifier


class MultiAgentProposalRoute(AethelisModel):
    proposal_id: Identifier
    agent_id: Identifier
    route: Identifier
    reason_labels: tuple[Identifier, ...] = ()
    event_candidate_id: Identifier | None = None
    verification_result_id: Identifier | None = None
    committed_event_id: Identifier | None = None
    state_diff_id: Identifier | None = None


class VerifierRetrievalBoundary(AethelisModel):
    agent_id: Identifier
    context_source_ids: tuple[Identifier, ...]
    verifier_selected_belief_ids: tuple[Identifier, ...]
    verifier_selected_memory_ids: tuple[Identifier, ...]
    verifier_filtered_belief_ids: tuple[Identifier, ...]
    verifier_suppressed_memory_ids: tuple[Identifier, ...]
    matches_proposal_context: Literal[True] = True


class MultiAgentWorldStepResult(AethelisModel):
    step_id: Identifier
    scenario_id: Identifier
    active_agent_ids: tuple[Identifier, ...]
    context: MultiAgentStepContext
    context_frames: tuple[ActiveAgentContextFrame, ...]
    verifier_retrieval_boundaries: tuple[VerifierRetrievalBoundary, ...] = ()
    action_proposals: tuple[ActionProposal, ...]
    proposal_bundle: MultiAgentProposalBundle
    dynamics_summary: MultiAgentDynamicsSummary
    arbitration_recommendation: ProposalArbitrationRecommendation
    routes: tuple[MultiAgentProposalRoute, ...]
    routed_proposal_ids: tuple[Identifier, ...] = ()
    blocked_proposal_ids: tuple[Identifier, ...] = ()
    revision_required_proposal_ids: tuple[Identifier, ...] = ()
    independent_proposal_ids: tuple[Identifier, ...] = ()
    joint_intent_candidates: tuple[JointIntentCandidate, ...] = ()
    event_candidates: tuple[EventCandidate, ...] = ()
    verification_results: tuple[VerificationResult, ...] = ()
    committed_events: tuple[CommittedEvent, ...] = ()
    state_diffs: tuple[StateDiff, ...] = ()
    state_diff_applied: bool = False
    apply_reports: tuple[StateApplyReport, ...] = ()
    applied_world_state: WorldState | None = None
    provider_called: bool = False
    db_written: Literal[False] = False
    arbitration_direct_commit: Literal[False] = False
    dynamics_direct_commit: Literal[False] = False
    joint_intent_direct_commit: Literal[False] = False
    can_mutate_canon: Literal[False] = False
    direct_world_state_mutation: Literal[False] = False

    def safe_summary(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "scenario_id": self.scenario_id,
            "active_agent_ids": list(self.active_agent_ids),
            "context": self.context.safe_summary(),
            "verifier_retrieval_boundaries": [
                boundary.model_dump(mode="json")
                for boundary in self.verifier_retrieval_boundaries
            ],
            "proposal_ids": [proposal.id for proposal in self.action_proposals],
            "proposal_bundle": self.proposal_bundle.summary().model_dump(mode="json"),
            "dynamics": self.dynamics_summary.model_dump(mode="json"),
            "arbitration": self.arbitration_recommendation.model_dump(mode="json"),
            "routes": [route.model_dump(mode="json") for route in self.routes],
            "routed_proposal_ids": list(self.routed_proposal_ids),
            "blocked_proposal_ids": list(self.blocked_proposal_ids),
            "revision_required_proposal_ids": list(self.revision_required_proposal_ids),
            "independent_proposal_ids": list(self.independent_proposal_ids),
            "joint_intent_candidate_ids": [
                candidate.id for candidate in self.joint_intent_candidates
            ],
            "event_candidate_ids": [candidate.id for candidate in self.event_candidates],
            "verification_result_ids": [
                verification.id for verification in self.verification_results
            ],
            "committed_event_ids": [event.id for event in self.committed_events],
            "state_diff_ids": [diff.id for diff in self.state_diffs],
            "state_diff_applied": self.state_diff_applied,
            "apply_reports": [report.safe_dict() for report in self.apply_reports],
            "provider_called": self.provider_called,
            "db_written": self.db_written,
            "arbitration_direct_commit": self.arbitration_direct_commit,
            "dynamics_direct_commit": self.dynamics_direct_commit,
            "joint_intent_direct_commit": self.joint_intent_direct_commit,
            "can_mutate_canon": self.can_mutate_canon,
            "direct_world_state_mutation": self.direct_world_state_mutation,
        }


def run_multi_agent_world_step(
    *,
    seed_path: Path,
    step_id: str,
    scenario_id: str,
    active_agent_ids: tuple[str, ...],
    apply: bool = False,
    context_budget_per_agent: int = 12,
    proposal_results: Mapping[str, ActionProposalGenerationResult] | None = None,
    pressure_context: dict[str, object] | None = None,
    evolution_context: dict[str, object] | None = None,
) -> MultiAgentWorldStepResult:
    bundle = _load_valid_seed(seed_path)
    return run_multi_agent_world_step_for_bundle(
        bundle=bundle,
        step_id=step_id,
        scenario_id=scenario_id,
        active_agent_ids=active_agent_ids,
        apply=apply,
        context_budget_per_agent=context_budget_per_agent,
        proposal_results=proposal_results,
        pressure_context=pressure_context,
        evolution_context=evolution_context,
    )


def run_multi_agent_world_step_for_bundle(
    *,
    bundle: SeedBundle,
    step_id: str,
    scenario_id: str,
    active_agent_ids: tuple[str, ...],
    apply: bool = False,
    context_budget_per_agent: int = 12,
    proposal_results: Mapping[str, ActionProposalGenerationResult] | None = None,
    pressure_context: dict[str, object] | None = None,
    evolution_context: dict[str, object] | None = None,
) -> MultiAgentWorldStepResult:
    context = build_multi_agent_step_context(
        bundle,
        step_id=step_id,
        scenario_id=scenario_id,
        active_agent_ids=active_agent_ids,
        context_budget_per_agent=context_budget_per_agent,
        pressure_context=pressure_context,
        evolution_context=evolution_context,
    )
    retrieved_by_agent = _retrieved_contexts(
        bundle=bundle,
        context=context,
        scenario_id=scenario_id,
        active_agent_ids=active_agent_ids,
        pressure_context=pressure_context,
        evolution_context=evolution_context,
    )
    generated = _generate_proposals(
        active_agent_ids=active_agent_ids,
        scenario_id=scenario_id,
        proposal_results=proposal_results,
    )
    proposals = tuple(
        _normalize_proposal_for_bundle(
            result.proposal,
            agent_id=agent_id,
            fallback_index=index,
        )
        for index, (agent_id, result) in enumerate(generated.items())
        if result.proposal is not None
    )
    proposal_frames = tuple(
        AgentProposalFrame.from_proposal(
            proposal,
            context_frame=context.frame_for(proposal.proposer_agent_id),
            target_resource_ids=_target_resource_ids(bundle, proposal, scenario_id),
            risk_flags=_generation_risk_flags(generated[proposal.proposer_agent_id]),
            confidence=_generation_confidence(generated[proposal.proposer_agent_id]),
            priority_score=_generation_priority(generated[proposal.proposer_agent_id]),
            utility_score=_generation_utility(generated[proposal.proposer_agent_id]),
        )
        for proposal in proposals
    )
    proposal_bundle = MultiAgentProposalBundle(
        id=f"bundle:{step_id}",
        step_id=step_id,
        scenario_id=scenario_id,
        active_agent_ids=active_agent_ids,
        frames=proposal_frames,
    )
    dynamics = analyze_multi_agent_dynamics(bundle=bundle, proposal_bundle=proposal_bundle)
    arbitration = recommend_arbitration(bundle=bundle, proposal_bundle=proposal_bundle)
    route_records, governance = _route_governance(
        bundle=bundle,
        scenario_id=scenario_id,
        proposal_bundle=proposal_bundle,
        arbitration=arbitration,
        retrieved_by_agent=retrieved_by_agent,
        apply=apply,
    )
    return MultiAgentWorldStepResult(
        step_id=step_id,
        scenario_id=scenario_id,
        active_agent_ids=active_agent_ids,
        context=context,
        context_frames=context.frames,
        verifier_retrieval_boundaries=_verifier_boundaries(context, retrieved_by_agent),
        action_proposals=proposals,
        proposal_bundle=proposal_bundle,
        dynamics_summary=dynamics,
        arbitration_recommendation=arbitration,
        routes=route_records,
        routed_proposal_ids=tuple(
            route.proposal_id for route in route_records if route.event_candidate_id is not None
        ),
        blocked_proposal_ids=arbitration.proposals_blocked_by_hard_conflict,
        revision_required_proposal_ids=arbitration.proposals_requiring_revision,
        independent_proposal_ids=arbitration.proposals_that_can_proceed_independently,
        joint_intent_candidates=arbitration.joint_intent_candidates,
        event_candidates=governance["event_candidates"],
        verification_results=governance["verification_results"],
        committed_events=governance["committed_events"],
        state_diffs=tuple(event.state_diff for event in governance["committed_events"]),
        state_diff_applied=any(report.applied for report in governance["apply_reports"]),
        apply_reports=governance["apply_reports"],
        applied_world_state=governance["applied_world_state"],
        provider_called=any(result.provider_called for result in generated.values()),
    )


def _load_valid_seed(seed_path: Path) -> SeedBundle:
    load_result = SeedLoader().load(seed_path)
    report = SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )
    if not report.success or load_result.bundle is None:
        raise ValueError(f"Seed validation failed: {report.safe_dict()}")
    return load_result.bundle


def _retrieved_contexts(
    *,
    bundle: SeedBundle,
    context: MultiAgentStepContext,
    scenario_id: str,
    active_agent_ids: tuple[str, ...],
    pressure_context: dict[str, object] | None,
    evolution_context: dict[str, object] | None,
) -> dict[str, RetrievedCognitionContext]:
    retriever = CognitionRetriever()
    retrieved: dict[str, RetrievedCognitionContext] = {}
    for agent_id in active_agent_ids:
        result = retriever.retrieve(
            bundle,
            agent_id=agent_id,
            scenario_id=scenario_id,
            pressure_context=pressure_context,
            evolution_context=evolution_context,
        )
        _assert_retrieval_matches_context(context.frame_for(agent_id), result)
        retrieved[agent_id] = result
    return retrieved


def _assert_retrieval_matches_context(
    frame: ActiveAgentContextFrame,
    retrieved: RetrievedCognitionContext,
) -> None:
    summary = retrieved.summary
    if (
        frame.retrieval.selected_belief_ids != summary.selected_belief_ids
        or frame.retrieval.selected_memory_ids != summary.selected_memory_ids
        or frame.retrieval.filtered_belief_ids != summary.filtered_belief_ids
        or frame.retrieval.suppressed_memory_ids != summary.suppressed_memory_ids
    ):
        raise ValueError(
            "Verifier retrieval boundary mismatch for agent "
            f"{frame.agent_id}: selected/filtered ids differ from B2 context."
        )


def _verifier_boundaries(
    context: MultiAgentStepContext,
    retrieved_by_agent: dict[str, RetrievedCognitionContext],
) -> tuple[VerifierRetrievalBoundary, ...]:
    return tuple(
        VerifierRetrievalBoundary(
            agent_id=frame.agent_id,
            context_source_ids=frame.packed_source_ids,
            verifier_selected_belief_ids=retrieved_by_agent[
                frame.agent_id
            ].summary.selected_belief_ids,
            verifier_selected_memory_ids=retrieved_by_agent[
                frame.agent_id
            ].summary.selected_memory_ids,
            verifier_filtered_belief_ids=retrieved_by_agent[
                frame.agent_id
            ].summary.filtered_belief_ids,
            verifier_suppressed_memory_ids=retrieved_by_agent[
                frame.agent_id
            ].summary.suppressed_memory_ids,
        )
        for frame in context.frames
    )


def _generate_proposals(
    *,
    active_agent_ids: tuple[str, ...],
    scenario_id: str,
    proposal_results: Mapping[str, ActionProposalGenerationResult] | None,
) -> dict[str, ActionProposalGenerationResult]:
    if proposal_results is not None:
        missing = [agent_id for agent_id in active_agent_ids if agent_id not in proposal_results]
        if missing:
            raise ValueError(f"Missing proposal generation results for agents: {missing}")
        return {agent_id: proposal_results[agent_id] for agent_id in active_agent_ids}
    engine = ActionProposalEngine()
    return {
        agent_id: engine.generate_deterministic(agent_id=agent_id, scenario_id=scenario_id)
        for agent_id in active_agent_ids
    }


def _normalize_proposal_for_bundle(
    proposal: ActionProposal,
    *,
    agent_id: str,
    fallback_index: int,
) -> ActionProposal:
    if proposal.proposer_agent_id != agent_id:
        raise ValueError(
            "Proposal owner mismatch: proposal "
            f"{proposal.id} has proposer_agent_id={proposal.proposer_agent_id}, "
            f"but was supplied for active agent {agent_id}."
        )
    proposal_id = proposal.id or f"proposal:{agent_id}:{fallback_index}"
    return proposal.model_copy(
        update={
            "id": f"{proposal_id}:{agent_id}",
        }
    )


def _target_resource_ids(
    bundle: SeedBundle,
    proposal: ActionProposal,
    scenario_id: str,
) -> tuple[str, ...]:
    del scenario_id
    resource_ids = {resource.id for resource in bundle.world.resources}
    return tuple(item for item in proposal.target_entity_ids if item in resource_ids)


def _generation_risk_flags(result: ActionProposalGenerationResult) -> tuple[str, ...]:
    return result.behavior_risk_flags


def _generation_confidence(result: ActionProposalGenerationResult) -> float:
    if result.behavior_score is None:
        return 0.5
    return max(0.0, min(1.0, result.behavior_score))


def _generation_priority(result: ActionProposalGenerationResult) -> float:
    decision = result.behavior_decision
    if decision == ProposalBehaviorDecision.ACCEPT:
        return 0.8
    if decision == ProposalBehaviorDecision.REPAIR:
        return 0.6
    if decision == ProposalBehaviorDecision.HOLD:
        return 0.3
    if decision == ProposalBehaviorDecision.REJECT:
        return 0.1
    return 0.5


def _generation_utility(result: ActionProposalGenerationResult) -> float:
    return 0.75 if result.source == ActionProposalSource.DETERMINISTIC_FIXTURE else 0.65


def _route_governance(
    *,
    bundle: SeedBundle,
    scenario_id: str,
    proposal_bundle: MultiAgentProposalBundle,
    arbitration: ProposalArbitrationRecommendation,
    retrieved_by_agent: dict[str, RetrievedCognitionContext],
    apply: bool,
) -> tuple[tuple[MultiAgentProposalRoute, ...], dict[str, object]]:
    recommendation_by_id = {
        frame.proposal_id: frame for frame in arbitration.decision_frames
    }
    proposal_frame_by_id = {
        frame.proposal_id: frame for frame in proposal_bundle.frames
    }
    ordered_ids = _routed_proposal_ids(arbitration)
    current_world = bundle.world
    event_candidates: list[EventCandidate] = []
    verification_results: list[VerificationResult] = []
    committed_events: list[CommittedEvent] = []
    apply_reports: list[StateApplyReport] = []
    route_by_id: dict[str, MultiAgentProposalRoute] = {}
    for proposal_id in ordered_ids:
        proposal_frame = proposal_frame_by_id[proposal_id]
        decision = recommendation_by_id[proposal_id]
        candidate, verification, committed, apply_report, current_world = _governance_chain(
            bundle=bundle.model_copy(update={"world": current_world}),
            scenario_id=scenario_id,
            proposal=proposal_frame.proposal,
            retrieved=retrieved_by_agent[proposal_frame.agent_id],
            apply=apply,
            world_state=current_world,
        )
        event_candidates.append(candidate)
        verification_results.append(verification)
        if committed is not None:
            committed_events.append(committed)
        if apply_report is not None:
            apply_reports.append(apply_report)
        route_by_id[proposal_id] = MultiAgentProposalRoute(
            proposal_id=proposal_frame.proposal_id,
            agent_id=proposal_frame.agent_id,
            route=decision.recommendation,
            reason_labels=decision.reason_labels,
            event_candidate_id=candidate.id,
            verification_result_id=verification.id,
            committed_event_id=committed.id if committed is not None else None,
            state_diff_id=committed.state_diff.id if committed is not None else None,
        )
    for proposal_frame in proposal_bundle.frames:
        if proposal_frame.proposal_id in route_by_id:
            continue
        decision = recommendation_by_id[proposal_frame.proposal_id]
        route_by_id[proposal_frame.proposal_id] = MultiAgentProposalRoute(
            proposal_id=proposal_frame.proposal_id,
            agent_id=proposal_frame.agent_id,
            route=decision.recommendation,
            reason_labels=decision.reason_labels,
        )
    return (
        tuple(route_by_id[frame.proposal_id] for frame in proposal_bundle.frames),
        {
            "event_candidates": tuple(event_candidates),
            "verification_results": tuple(verification_results),
            "committed_events": tuple(committed_events),
            "apply_reports": tuple(apply_reports),
            "applied_world_state": current_world if apply_reports else None,
        },
    )


def _routed_proposal_ids(arbitration: ProposalArbitrationRecommendation) -> tuple[str, ...]:
    allowed = {
        "primary_recommendation",
        "can_proceed_independently",
    }
    ids = [
        frame.proposal_id
        for frame in arbitration.decision_frames
        if frame.recommendation in allowed
    ]
    primary = arbitration.primary_proposal_id
    return tuple(
        sorted(
            ids,
            key=lambda proposal_id: (
                0 if proposal_id == primary else 1,
                proposal_id,
            ),
        )
    )


def _governance_chain(
    *,
    bundle: SeedBundle,
    scenario_id: str,
    proposal: ActionProposal,
    retrieved: RetrievedCognitionContext,
    apply: bool,
    world_state: WorldState,
) -> tuple[
    EventCandidate,
    VerificationResult,
    CommittedEvent | None,
    StateApplyReport | None,
    WorldState,
]:
    proposal = repair_action_proposal(proposal)
    candidate = action_proposal_to_event_candidate(proposal, scenario_id=scenario_id)
    candidate_route, candidate_quality = candidate_behavior_route(proposal)
    if candidate_route != CandidateBehaviorRoute.UNDER_REVIEW:
        verification = candidate_gate_verification_result(
            candidate,
            route=candidate_route,
            quality=candidate_quality,
        )
    else:
        verification = DeterministicVerifier().verify(
            bundle=bundle,
            observation=retrieved.observation,
            cognition=retrieved.cognition,
            proposal=proposal,
            candidate=candidate,
            scenario_id=scenario_id,
        )
    committed = build_committed_event_from_verification(
        candidate=candidate,
        verification=verification,
        scenario_id=scenario_id,
        world_state=world_state,
    )
    if not apply or committed is None:
        return candidate, verification, committed, None, world_state
    applied_world, report = ControlledStateDiffApplier().apply(
        world_state=world_state,
        committed_event=committed,
        verification_result=verification,
    )
    return candidate, verification, committed, report, applied_world
