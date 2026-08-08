from __future__ import annotations

from itertools import combinations
from typing import Literal

from pydantic import Field, model_validator

from aethelis.agents.retrieval import ActiveAgentContextFrame
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.events import ActionIntent, ActionProposal
from aethelis.schemas.seed import SeedBundle


class AgentProposalFrame(AethelisModel):
    agent_id: Identifier
    proposal_id: Identifier
    proposal: ActionProposal
    intent: ActionIntent
    context_source_ids: tuple[Identifier, ...] = ()
    selected_belief_ids: tuple[Identifier, ...] = ()
    selected_memory_ids: tuple[Identifier, ...] = ()
    target_location_id: Identifier | None = None
    target_entity_ids: tuple[Identifier, ...] = ()
    target_resource_ids: tuple[Identifier, ...] = ()
    target_ids: tuple[Identifier, ...] = ()
    risk_flags: tuple[Identifier, ...] = ()
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    priority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    utility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    can_modify_world_state: Literal[False] = False
    can_mutate_canon: Literal[False] = False

    @classmethod
    def from_proposal(
        cls,
        proposal: ActionProposal,
        *,
        context_frame: ActiveAgentContextFrame | None = None,
        target_resource_ids: tuple[str, ...] = (),
        risk_flags: tuple[str, ...] = (),
        confidence: float = 0.5,
        priority_score: float = 0.5,
        utility_score: float = 0.5,
    ) -> AgentProposalFrame:
        context_source_ids: tuple[str, ...] = ()
        selected_belief_ids: tuple[str, ...] = ()
        selected_memory_ids: tuple[str, ...] = ()
        if context_frame is not None:
            context_source_ids = context_frame.packed_source_ids
            selected_belief_ids = context_frame.retrieval.selected_belief_ids
            selected_memory_ids = context_frame.retrieval.selected_memory_ids
        target_ids = (*proposal.target_entity_ids, *target_resource_ids)
        return cls(
            agent_id=proposal.proposer_agent_id,
            proposal_id=proposal.id,
            proposal=proposal,
            intent=proposal.intent,
            context_source_ids=context_source_ids,
            selected_belief_ids=selected_belief_ids,
            selected_memory_ids=selected_memory_ids,
            target_location_id=proposal.target_location_id,
            target_entity_ids=proposal.target_entity_ids,
            target_resource_ids=target_resource_ids,
            target_ids=target_ids,
            risk_flags=risk_flags,
            confidence=confidence,
            priority_score=priority_score,
            utility_score=utility_score,
        )

    @model_validator(mode="after")
    def validate_proposal_owner(self) -> AgentProposalFrame:
        if self.proposal.proposer_agent_id != self.agent_id:
            raise ValueError("proposal proposer must match frame agent")
        if self.proposal.id != self.proposal_id:
            raise ValueError("proposal id must match frame proposal_id")
        return self


class ProposalBundleSummary(AethelisModel):
    bundle_id: Identifier
    step_id: Identifier
    scenario_id: Identifier
    active_agent_ids: tuple[Identifier, ...]
    proposal_ids: tuple[Identifier, ...]
    can_modify_world_state: Literal[False] = False
    can_mutate_canon: Literal[False] = False


class MultiAgentProposalBundle(AethelisModel):
    id: Identifier
    step_id: Identifier
    scenario_id: Identifier
    active_agent_ids: tuple[Identifier, ...]
    frames: tuple[AgentProposalFrame, ...]
    can_modify_world_state: Literal[False] = False
    can_mutate_canon: Literal[False] = False
    emits_event_candidate: Literal[False] = False
    emits_verification_result: Literal[False] = False
    emits_committed_event: Literal[False] = False
    emits_state_diff: Literal[False] = False

    @model_validator(mode="after")
    def validate_active_set(self) -> MultiAgentProposalBundle:
        frame_agent_ids = tuple(frame.agent_id for frame in self.frames)
        if len(set(frame_agent_ids)) != len(frame_agent_ids):
            raise ValueError("proposal bundle cannot contain duplicate agent frames")
        if set(frame_agent_ids) != set(self.active_agent_ids):
            raise ValueError("proposal bundle frames must match active_agent_ids")
        proposal_ids = tuple(frame.proposal_id for frame in self.frames)
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("proposal bundle cannot contain duplicate proposal ids")
        return self

    def summary(self) -> ProposalBundleSummary:
        return ProposalBundleSummary(
            bundle_id=self.id,
            step_id=self.step_id,
            scenario_id=self.scenario_id,
            active_agent_ids=self.active_agent_ids,
            proposal_ids=tuple(frame.proposal_id for frame in self.frames),
        )

    def frame_for_proposal(self, proposal_id: str) -> AgentProposalFrame:
        for frame in self.frames:
            if frame.proposal_id == proposal_id:
                return frame
        raise ValueError(f"Unknown proposal id: {proposal_id}")


class ProposalContention(AethelisModel):
    id: Identifier
    contention_type: Identifier
    proposal_ids: tuple[Identifier, ...]
    target_ids: tuple[Identifier, ...]
    reason_labels: tuple[Identifier, ...]


class ProposalConflict(AethelisModel):
    id: Identifier
    conflict_type: Identifier
    proposal_ids: tuple[Identifier, ...]
    target_ids: tuple[Identifier, ...]
    reason_labels: tuple[Identifier, ...]


class ProposalCooperation(AethelisModel):
    id: Identifier
    cooperation_type: Identifier
    proposal_ids: tuple[Identifier, ...]
    target_ids: tuple[Identifier, ...]
    reason_labels: tuple[Identifier, ...]


class ProposalDependency(AethelisModel):
    id: Identifier
    dependency_type: Identifier
    prerequisite_proposal_id: Identifier
    dependent_proposal_id: Identifier
    target_ids: tuple[Identifier, ...]
    reason_labels: tuple[Identifier, ...]


class ProposalRelationshipSignal(AethelisModel):
    id: Identifier
    signal_type: Identifier
    proposal_ids: tuple[Identifier, ...]
    agent_ids: tuple[Identifier, ...]
    relationship_id: Identifier
    trust: int = Field(ge=-5, le=5)
    reason_labels: tuple[Identifier, ...]


class ProposalFactionSignal(AethelisModel):
    id: Identifier
    proposal_ids: tuple[Identifier, ...]
    agent_ids: tuple[Identifier, ...]
    faction_ids: tuple[Identifier, ...]
    reason_labels: tuple[Identifier, ...]


class ProposalBeliefDivergence(AethelisModel):
    id: Identifier
    proposal_ids: tuple[Identifier, ...]
    agent_ids: tuple[Identifier, ...]
    selected_belief_counts: tuple[int, ...]
    reason_labels: tuple[Identifier, ...]


class MultiAgentDynamicsSummary(AethelisModel):
    bundle_id: Identifier
    contentions: tuple[ProposalContention, ...] = ()
    conflicts: tuple[ProposalConflict, ...] = ()
    cooperations: tuple[ProposalCooperation, ...] = ()
    dependencies: tuple[ProposalDependency, ...] = ()
    relationship_signals: tuple[ProposalRelationshipSignal, ...] = ()
    faction_tension_signals: tuple[ProposalFactionSignal, ...] = ()
    belief_divergences: tuple[ProposalBeliefDivergence, ...] = ()
    pressure_aligned_proposal_ids: tuple[Identifier, ...] = ()
    can_modify_world_state: Literal[False] = False
    can_mutate_canon: Literal[False] = False
    emits_event_candidate: Literal[False] = False
    emits_verification_result: Literal[False] = False
    emits_committed_event: Literal[False] = False
    emits_state_diff: Literal[False] = False


class JointIntentCandidate(AethelisModel):
    id: Identifier
    proposal_ids: tuple[Identifier, ...]
    agent_ids: tuple[Identifier, ...]
    target_ids: tuple[Identifier, ...]
    intent_labels: tuple[Identifier, ...]
    reason_labels: tuple[Identifier, ...]
    can_modify_world_state: Literal[False] = False
    can_mutate_canon: Literal[False] = False


class ArbitrationDecisionFrame(AethelisModel):
    proposal_id: Identifier
    agent_id: Identifier
    recommendation: Identifier
    score: float = Field(ge=0.0, le=1.0)
    reason_labels: tuple[Identifier, ...]
    conflict_source_proposal_ids: tuple[Identifier, ...] = ()
    cooperation_source_proposal_ids: tuple[Identifier, ...] = ()


class ProposalArbitrationRecommendation(AethelisModel):
    bundle_id: Identifier
    primary_proposal_id: Identifier | None = None
    proposals_requiring_revision: tuple[Identifier, ...] = ()
    proposals_blocked_by_hard_conflict: tuple[Identifier, ...] = ()
    proposals_that_can_proceed_independently: tuple[Identifier, ...] = ()
    joint_intent_candidates: tuple[JointIntentCandidate, ...] = ()
    decision_frames: tuple[ArbitrationDecisionFrame, ...] = ()
    rationale: tuple[str, ...] = ()
    score_components: dict[Identifier, dict[str, float]] = Field(default_factory=dict)
    can_modify_world_state: Literal[False] = False
    can_mutate_canon: Literal[False] = False
    creates_event_candidate: Literal[False] = False
    creates_verification_result: Literal[False] = False
    creates_committed_event: Literal[False] = False
    creates_state_diff: Literal[False] = False


def analyze_multi_agent_dynamics(
    *,
    bundle: SeedBundle,
    proposal_bundle: MultiAgentProposalBundle,
) -> MultiAgentDynamicsSummary:
    contentions: list[ProposalContention] = []
    conflicts: list[ProposalConflict] = []
    cooperations: list[ProposalCooperation] = []
    dependencies: list[ProposalDependency] = []
    relationship_signals: list[ProposalRelationshipSignal] = []
    faction_signals: list[ProposalFactionSignal] = []
    belief_divergences: list[ProposalBeliefDivergence] = []
    pressure_ids = _pressure_aligned_ids(bundle, proposal_bundle)

    for left, right in combinations(proposal_bundle.frames, 2):
        pair_id = f"{left.proposal_id}:{right.proposal_id}"
        shared_targets = _shared_targets(left, right)
        same_location = _same_location(left, right)
        if shared_targets:
            contentions.append(
                ProposalContention(
                    id=f"contention:{pair_id}:same_target",
                    contention_type="same_target",
                    proposal_ids=(left.proposal_id, right.proposal_id),
                    target_ids=shared_targets,
                    reason_labels=("shared_target_contention",),
                )
            )
        shared_resources = _shared_resources(left, right)
        if shared_resources:
            contentions.append(
                ProposalContention(
                    id=f"contention:{pair_id}:resource",
                    contention_type="resource_contention",
                    proposal_ids=(left.proposal_id, right.proposal_id),
                    target_ids=shared_resources,
                    reason_labels=("resource_contention",),
                )
            )
        if same_location:
            contentions.append(
                ProposalContention(
                    id=f"contention:{pair_id}:location",
                    contention_type="location_contention",
                    proposal_ids=(left.proposal_id, right.proposal_id),
                    target_ids=(left.target_location_id,),
                    reason_labels=("location_contention",),
                )
            )
        if (shared_targets or same_location) and _conflicting_intents(left.intent, right.intent):
            conflicts.append(
                ProposalConflict(
                    id=f"conflict:{pair_id}:intent",
                    conflict_type="opposing_intents",
                    proposal_ids=(left.proposal_id, right.proposal_id),
                    target_ids=shared_targets or (left.target_location_id,),
                    reason_labels=("opposing_intents_same_target",),
                )
            )
        if (shared_targets or same_location) and _supportive_intents(left.intent, right.intent):
            cooperations.append(
                ProposalCooperation(
                    id=f"cooperation:{pair_id}:intent",
                    cooperation_type="mutually_supportive_intents",
                    proposal_ids=(left.proposal_id, right.proposal_id),
                    target_ids=shared_targets or (left.target_location_id,),
                    reason_labels=("supportive_intents_same_target",),
                )
            )
        dependency = _dependency(left, right, shared_targets, same_location)
        if dependency is not None:
            dependencies.append(dependency)
        relationship = _relationship_between(bundle, left.agent_id, right.agent_id)
        if relationship is not None and abs(relationship.trust) >= 2:
            signal_type = (
                "relationship_support" if relationship.trust > 0 else "relationship_distrust"
            )
            relationship_signals.append(
                ProposalRelationshipSignal(
                    id=f"relationship_signal:{pair_id}",
                    signal_type=signal_type,
                    proposal_ids=(left.proposal_id, right.proposal_id),
                    agent_ids=(left.agent_id, right.agent_id),
                    relationship_id=relationship.id,
                    trust=relationship.trust,
                    reason_labels=(signal_type,),
                )
            )
        factions = _factions(bundle, left.agent_id, right.agent_id)
        if (shared_targets or same_location) and len(set(factions)) > 1:
            faction_signals.append(
                ProposalFactionSignal(
                    id=f"faction_signal:{pair_id}",
                    proposal_ids=(left.proposal_id, right.proposal_id),
                    agent_ids=(left.agent_id, right.agent_id),
                    faction_ids=factions,
                    reason_labels=("different_factions_shared_target",),
                )
            )
        if (shared_targets or same_location) and set(left.selected_belief_ids) != set(
            right.selected_belief_ids
        ):
            belief_divergences.append(
                ProposalBeliefDivergence(
                    id=f"belief_divergence:{pair_id}",
                    proposal_ids=(left.proposal_id, right.proposal_id),
                    agent_ids=(left.agent_id, right.agent_id),
                    selected_belief_counts=(
                        len(left.selected_belief_ids),
                        len(right.selected_belief_ids),
                    ),
                    reason_labels=("selected_belief_set_divergence",),
                )
            )

    return MultiAgentDynamicsSummary(
        bundle_id=proposal_bundle.id,
        contentions=tuple(contentions),
        conflicts=tuple(conflicts),
        cooperations=tuple(cooperations),
        dependencies=tuple(dependencies),
        relationship_signals=tuple(relationship_signals),
        faction_tension_signals=tuple(faction_signals),
        belief_divergences=tuple(belief_divergences),
        pressure_aligned_proposal_ids=pressure_ids,
    )


def recommend_arbitration(
    *,
    bundle: SeedBundle,
    proposal_bundle: MultiAgentProposalBundle,
) -> ProposalArbitrationRecommendation:
    dynamics = analyze_multi_agent_dynamics(bundle=bundle, proposal_bundle=proposal_bundle)
    components = {
        frame.proposal_id: _score_components(frame, proposal_bundle, dynamics)
        for frame in proposal_bundle.frames
    }
    scores = {
        proposal_id: max(0.0, min(1.0, sum(component.values())))
        for proposal_id, component in components.items()
    }
    primary = max(scores, key=lambda proposal_id: (scores[proposal_id], proposal_id), default=None)
    blocked = _blocked_by_conflict_ids(dynamics, scores)
    revise = _revise_ids(dynamics, scores, blocked)
    joint_candidates = _joint_candidates(proposal_bundle, dynamics)
    joint_ids = {
        proposal_id
        for candidate in joint_candidates
        for proposal_id in candidate.proposal_ids
    }
    independent = tuple(
        frame.proposal_id
        for frame in proposal_bundle.frames
        if frame.proposal_id not in blocked
        and frame.proposal_id not in revise
        and frame.proposal_id not in joint_ids
        and frame.proposal_id != primary
    )
    decision_frames = tuple(
        ArbitrationDecisionFrame(
            proposal_id=frame.proposal_id,
            agent_id=frame.agent_id,
            recommendation=_recommendation_label(
                frame.proposal_id,
                primary,
                blocked,
                revise,
                joint_ids,
            ),
            score=scores[frame.proposal_id],
            reason_labels=_decision_reasons(frame.proposal_id, dynamics),
            conflict_source_proposal_ids=_related_conflict_ids(frame.proposal_id, dynamics),
            cooperation_source_proposal_ids=_related_cooperation_ids(frame.proposal_id, dynamics),
        )
        for frame in proposal_bundle.frames
    )
    return ProposalArbitrationRecommendation(
        bundle_id=proposal_bundle.id,
        primary_proposal_id=primary,
        proposals_requiring_revision=revise,
        proposals_blocked_by_hard_conflict=blocked,
        proposals_that_can_proceed_independently=independent,
        joint_intent_candidates=joint_candidates,
        decision_frames=decision_frames,
        rationale=(
            "Recommendation is pre-governance evidence only; it does not verify, commit, or diff.",
        ),
        score_components=components,
    )


def _score_components(
    frame: AgentProposalFrame,
    proposal_bundle: MultiAgentProposalBundle,
    dynamics: MultiAgentDynamicsSummary,
) -> dict[str, float]:
    return {
        "confidence": frame.confidence * 0.30,
        "priority": frame.priority_score * 0.30,
        "utility": frame.utility_score * 0.20,
        "pressure_alignment": (
            0.10 if frame.proposal_id in dynamics.pressure_aligned_proposal_ids else 0.0
        ),
        "relationship_support": _relationship_component(frame.proposal_id, dynamics, 0.08),
        "relationship_distrust": _relationship_component(frame.proposal_id, dynamics, -0.10),
        "faction_tension": _faction_component(frame.proposal_id, dynamics),
        "risk_penalty": -0.15 if frame.risk_flags else 0.0,
        "bundle_size_prior": 0.02 if len(proposal_bundle.frames) > 1 else 0.0,
    }


def _relationship_component(
    proposal_id: str,
    dynamics: MultiAgentDynamicsSummary,
    value: float,
) -> float:
    for signal in dynamics.relationship_signals:
        if proposal_id in signal.proposal_ids:
            if value > 0 and signal.signal_type == "relationship_support":
                return value
            if value < 0 and signal.signal_type == "relationship_distrust":
                return value
    return 0.0


def _faction_component(proposal_id: str, dynamics: MultiAgentDynamicsSummary) -> float:
    return (
        -0.04
        if any(proposal_id in signal.proposal_ids for signal in dynamics.faction_tension_signals)
        else 0.0
    )


def _shared_targets(left: AgentProposalFrame, right: AgentProposalFrame) -> tuple[str, ...]:
    return tuple(sorted(set(left.target_ids) & set(right.target_ids)))


def _shared_resources(left: AgentProposalFrame, right: AgentProposalFrame) -> tuple[str, ...]:
    return tuple(sorted(set(left.target_resource_ids) & set(right.target_resource_ids)))


def _same_location(left: AgentProposalFrame, right: AgentProposalFrame) -> bool:
    return (
        left.target_location_id is not None
        and left.target_location_id == right.target_location_id
    )


def _conflicting_intents(left: ActionIntent, right: ActionIntent) -> bool:
    return frozenset((left, right)) in {
        frozenset((ActionIntent.GUARD, ActionIntent.MOVE)),
        frozenset((ActionIntent.GUARD, ActionIntent.TRADE)),
        frozenset((ActionIntent.GUARD, ActionIntent.REPAIR)),
        frozenset((ActionIntent.OBSERVE, ActionIntent.MOVE)),
    }


def _supportive_intents(left: ActionIntent, right: ActionIntent) -> bool:
    return frozenset((left, right)) in {
        frozenset((ActionIntent.INVESTIGATE, ActionIntent.REPAIR)),
        frozenset((ActionIntent.OBSERVE, ActionIntent.GUARD)),
        frozenset((ActionIntent.NEGOTIATE, ActionIntent.TRADE)),
        frozenset((ActionIntent.OBSERVE, ActionIntent.REPAIR)),
    }


def _dependency(
    left: AgentProposalFrame,
    right: AgentProposalFrame,
    shared_targets: tuple[str, ...],
    same_location: bool,
) -> ProposalDependency | None:
    if not shared_targets and not same_location:
        return None
    prerequisite_intents = {ActionIntent.INVESTIGATE, ActionIntent.OBSERVE, ActionIntent.NEGOTIATE}
    dependent_intents = {ActionIntent.REPAIR, ActionIntent.TRADE, ActionIntent.MOVE}
    if left.intent in prerequisite_intents and right.intent in dependent_intents:
        prerequisite, dependent = left, right
    elif right.intent in prerequisite_intents and left.intent in dependent_intents:
        prerequisite, dependent = right, left
    else:
        return None
    return ProposalDependency(
        id=f"dependency:{prerequisite.proposal_id}:{dependent.proposal_id}",
        dependency_type="sequential_dependency",
        prerequisite_proposal_id=prerequisite.proposal_id,
        dependent_proposal_id=dependent.proposal_id,
        target_ids=shared_targets or (prerequisite.target_location_id,),
        reason_labels=("observe_or_negotiate_before_effectful_action",),
    )


def _relationship_between(bundle: SeedBundle, left_agent_id: str, right_agent_id: str):
    for relationship in bundle.agents.relationships:
        if {relationship.source_agent_id, relationship.target_agent_id} == {
            left_agent_id,
            right_agent_id,
        }:
            return relationship
    return None


def _factions(bundle: SeedBundle, left_agent_id: str, right_agent_id: str) -> tuple[str, ...]:
    by_agent = {agent.id: agent.faction_id for agent in bundle.agents.agents}
    return tuple(
        faction
        for faction in (by_agent.get(left_agent_id), by_agent.get(right_agent_id))
        if faction is not None
    )


def _pressure_aligned_ids(
    bundle: SeedBundle,
    proposal_bundle: MultiAgentProposalBundle,
) -> tuple[str, ...]:
    if bundle.metadata is None:
        return ()
    aligned: list[str] = []
    for frame in proposal_bundle.frames:
        for pressure in bundle.metadata.pressure_seeds:
            resource_match = (
                pressure.resource_id is not None
                and pressure.resource_id in frame.target_resource_ids
            )
            location_match = (
                pressure.location_id is not None
                and pressure.location_id == frame.target_location_id
            )
            if pressure.level >= 5 and (resource_match or location_match):
                aligned.append(frame.proposal_id)
                break
    return tuple(aligned)


def _blocked_by_conflict_ids(
    dynamics: MultiAgentDynamicsSummary,
    scores: dict[str, float],
) -> tuple[str, ...]:
    blocked: list[str] = []
    for conflict in dynamics.conflicts:
        ordered = sorted(
            conflict.proposal_ids,
            key=lambda proposal_id: (-scores[proposal_id], proposal_id),
        )
        blocked.extend(ordered[1:])
    return tuple(dict.fromkeys(blocked))


def _revise_ids(
    dynamics: MultiAgentDynamicsSummary,
    scores: dict[str, float],
    blocked: tuple[str, ...],
) -> tuple[str, ...]:
    revise: list[str] = []
    for contention in dynamics.contentions:
        candidates = [
            proposal_id
            for proposal_id in contention.proposal_ids
            if proposal_id not in blocked
        ]
        if len(candidates) < 2:
            continue
        ordered = sorted(candidates, key=lambda proposal_id: (-scores[proposal_id], proposal_id))
        revise.extend(ordered[1:])
    return tuple(dict.fromkeys(revise))


def _joint_candidates(
    proposal_bundle: MultiAgentProposalBundle,
    dynamics: MultiAgentDynamicsSummary,
) -> tuple[JointIntentCandidate, ...]:
    candidates: list[JointIntentCandidate] = []
    for cooperation in dynamics.cooperations:
        frames = tuple(
            proposal_bundle.frame_for_proposal(proposal_id)
            for proposal_id in cooperation.proposal_ids
        )
        candidates.append(
            JointIntentCandidate(
                id=f"joint_intent:{':'.join(cooperation.proposal_ids)}",
                proposal_ids=cooperation.proposal_ids,
                agent_ids=tuple(frame.agent_id for frame in frames),
                target_ids=cooperation.target_ids,
                intent_labels=tuple(frame.intent.value for frame in frames),
                reason_labels=cooperation.reason_labels,
            )
        )
    return tuple(candidates)


def _recommendation_label(
    proposal_id: str,
    primary: str | None,
    blocked: tuple[str, ...],
    revise: tuple[str, ...],
    joint_ids: set[str],
) -> str:
    if proposal_id in blocked:
        return "blocked_by_hard_conflict"
    if proposal_id in revise:
        return "requires_revision"
    if proposal_id in joint_ids:
        return "joint_intent_candidate"
    if proposal_id == primary:
        return "primary_recommendation"
    return "can_proceed_independently"


def _decision_reasons(
    proposal_id: str,
    dynamics: MultiAgentDynamicsSummary,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if proposal_id in dynamics.pressure_aligned_proposal_ids:
        reasons.append("pressure_aligned_priority")
    if any(proposal_id in item.proposal_ids for item in dynamics.conflicts):
        reasons.append("conflict_relation_detected")
    if any(proposal_id in item.proposal_ids for item in dynamics.contentions):
        reasons.append("contention_relation_detected")
    if any(proposal_id in item.proposal_ids for item in dynamics.cooperations):
        reasons.append("cooperation_relation_detected")
    if any(proposal_id in item.proposal_ids for item in dynamics.relationship_signals):
        reasons.append("relationship_signal")
    if any(proposal_id in item.proposal_ids for item in dynamics.faction_tension_signals):
        reasons.append("faction_tension_signal")
    return tuple(reasons or ("no_multi_agent_relation_detected",))


def _related_conflict_ids(
    proposal_id: str,
    dynamics: MultiAgentDynamicsSummary,
) -> tuple[str, ...]:
    related: list[str] = []
    for conflict in dynamics.conflicts:
        if proposal_id in conflict.proposal_ids:
            related.extend(item for item in conflict.proposal_ids if item != proposal_id)
    return tuple(dict.fromkeys(related))


def _related_cooperation_ids(
    proposal_id: str,
    dynamics: MultiAgentDynamicsSummary,
) -> tuple[str, ...]:
    related: list[str] = []
    for cooperation in dynamics.cooperations:
        if proposal_id in cooperation.proposal_ids:
            related.extend(item for item in cooperation.proposal_ids if item != proposal_id)
    return tuple(dict.fromkeys(related))
