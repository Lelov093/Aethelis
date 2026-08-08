from __future__ import annotations

from dataclasses import dataclass

from aethelis.product.command_contracts import ParsedPlayerIntent, PlayerCommand
from aethelis.product.content_contracts import ProductContentPackage
from aethelis.product.governance_contracts import GovernedWorldOutcome
from aethelis.runtime.state_apply import ControlledStateDiffApplier
from aethelis.schemas.common import ConfidenceBand
from aethelis.schemas.events import (
    ActionIntent,
    ActionProposal,
    CommittedEvent,
    EventCandidate,
    EventCandidateStatus,
    PatchOperation,
    PatchTargetType,
    StateDiff,
    StatePatch,
    VerificationCheck,
    VerificationDecision,
    VerificationResult,
)
from aethelis.schemas.ledger import (
    BeliefCandidate,
    BeliefKind,
    BeliefRecord,
    BeliefTruthStatus,
    MemoryKind,
    MemoryRecord,
)
from aethelis.schemas.world import WorldActivityRecord, WorldClockState, WorldState


@dataclass(frozen=True)
class _AdvancementPlan:
    actor_ids: tuple[str, ...]
    kind: str
    summary: str
    location_id: str | None
    source_claim_id: str | None = None
    propagated_candidate: BeliefCandidate | None = None
    propagated_belief: BeliefRecord | None = None
    propagated_memory: MemoryRecord | None = None
    moving_agent_id: str | None = None
    destination_id: str | None = None


def govern_living_world(
    *,
    command: PlayerCommand,
    intent: ParsedPlayerIntent,
    world_state: WorldState,
    content_package: ProductContentPackage | None,
) -> GovernedWorldOutcome:
    next_turn = world_state.clock.turn + 1
    event_id = f"committed_{command.id}"
    plan = _select_plan(world_state, command.id, event_id, next_turn)
    profiles_by_id = {profile.id: profile for profile in world_state.agent_profiles}
    checks = (
        VerificationCheck(
            name="supported_world_advancement",
            passed=intent.normalized_action == "advance_world",
            message="Bounded world advancement must use advance_world.",
        ),
        VerificationCheck(
            name="authorized_player_trigger",
            passed=intent.actor_id == command.actor_id,
            message="The player command may trigger but not impersonate an Agent.",
        ),
        VerificationCheck(
            name="living_world_content_available",
            passed=content_package is not None and bool(world_state.agent_profiles),
            message="Advancement requires product-persisted Agent profiles.",
        ),
        VerificationCheck(
            name="bounded_actor_selection",
            passed=bool(plan.actor_ids)
            and len(plan.actor_ids) <= 2
            and all(actor_id in profiles_by_id for actor_id in plan.actor_ids),
            message="One bounded independent or two-Agent plan must use known Agents.",
        ),
        VerificationCheck(
            name="clock_advances_once",
            passed=next_turn == world_state.clock.turn + 1,
            message="One command advances exactly one living-world turn.",
        ),
        VerificationCheck(
            name="claim_provenance_preserved",
            passed=plan.source_claim_id is None
            or any(claim.id == plan.source_claim_id for claim in world_state.agent_claims),
            message="Knowledge propagation must cite a persisted player claim.",
        ),
    )
    decision = (
        VerificationDecision.COMMIT
        if all(check.passed for check in checks)
        else VerificationDecision.REJECT
    )
    primary_actor = plan.actor_ids[0] if plan.actor_ids else "world_director"
    proposal = ActionProposal(
        id=f"proposal_{command.id}",
        proposer_agent_id=primary_actor,
        intent=ActionIntent.OBSERVE if plan.kind == "independent_action" else ActionIntent.DIALOGUE,
        rationale="A bounded deterministic scheduler selected an Agent plan from persisted state.",
        target_location_id=plan.location_id,
        target_entity_ids=plan.actor_ids[1:],
        expected_outcome=plan.summary,
    )
    candidate = EventCandidate(
        id=f"candidate_{command.id}",
        source_action_proposal_id=proposal.id,
        actor_agent_id=primary_actor,
        summary=plan.summary,
        status=EventCandidateStatus.UNDER_REVIEW,
        involved_location_ids=(plan.location_id,) if plan.location_id else (),
        involved_entity_ids=plan.actor_ids,
    )
    verification = VerificationResult(
        id=f"verification_{command.id}",
        event_candidate_id=candidate.id,
        decision=decision,
        verifier="product_living_world_governance_v1",
        checks=checks,
        reasons=(
            "The selected Agent plan is bounded, source-attributed, and contains no "
            "direct Canon mutation."
            if decision == VerificationDecision.COMMIT
            else "The living-world plan failed its deterministic product boundary.",
        ),
        risk_flags=() if decision == VerificationDecision.COMMIT else ("world_step_rejected",),
    )
    if decision != VerificationDecision.COMMIT:
        return GovernedWorldOutcome(
            proposal=proposal,
            candidate=candidate,
            verification=verification,
            committed_event=None,
            resulting_world_state=None,
            apply_report=None,
            player_message="The world could not advance safely from this state.",
            consequences=(),
        )

    clock = WorldClockState(
        turn=next_turn,
        elapsed_minutes=world_state.clock.elapsed_minutes + 15,
    )
    activity = WorldActivityRecord(
        id=f"activity_{command.id}",
        turn=next_turn,
        actor_agent_ids=plan.actor_ids,
        activity_kind=plan.kind,  # type: ignore[arg-type]
        summary=plan.summary,
        location_id=plan.location_id,
        source_claim_id=plan.source_claim_id,
        committed_event_id=event_id,
    )
    patches = [
        StatePatch(
            operation=PatchOperation.UPDATE,
            target_type=PatchTargetType.WORLD,
            target_id=world_state.world_id,
            path="/clock",
            before=world_state.clock.model_dump(mode="json"),
            after=clock.model_dump(mode="json"),
            reason="A deliberate wait advances one bounded world turn.",
        ),
        _collection_patch(
            path="/world_activities",
            before=world_state.world_activities,
            after=(*world_state.world_activities, activity),
            target_id=world_state.world_id,
            operation=PatchOperation.APPEND,
            reason="The verified Agent action becomes player-visible product history.",
        ),
    ]
    if (
        plan.propagated_candidate is not None
        and plan.propagated_belief is not None
        and plan.propagated_memory is not None
    ):
        patches.extend(
            (
                _collection_patch(
                    path="/agent_belief_candidates",
                    before=world_state.agent_belief_candidates,
                    after=(
                        *world_state.agent_belief_candidates,
                        plan.propagated_candidate,
                    ),
                    target_id=plan.actor_ids[-1],
                    operation=PatchOperation.APPEND,
                    reason="A listener receives a source-attributed propagated candidate.",
                ),
                _collection_patch(
                    path="/agent_memories",
                    before=world_state.agent_memories,
                    after=(*world_state.agent_memories, plan.propagated_memory),
                    target_id=plan.actor_ids[-1],
                    operation=PatchOperation.APPEND,
                    reason="The receiving Agent remembers who relayed the claim.",
                ),
                _collection_patch(
                    path="/agent_beliefs",
                    before=world_state.agent_beliefs,
                    after=(*world_state.agent_beliefs, plan.propagated_belief),
                    target_id=plan.actor_ids[-1],
                    operation=PatchOperation.APPEND,
                    reason="The propagated claim remains a low-confidence Agent belief.",
                ),
            )
        )
    if plan.moving_agent_id and plan.destination_id:
        moving_entity = next(
            entity for entity in world_state.entities if entity.id == plan.moving_agent_id
        )
        patches.append(
            StatePatch(
                operation=PatchOperation.UPDATE,
                target_type=PatchTargetType.ENTITY,
                target_id=moving_entity.id,
                path=f"/entity/{moving_entity.id}/location_id",
                before=moving_entity.location_id,
                after=plan.destination_id,
                reason="The verified routine Agent action changes visible character location.",
            )
        )
        profiles = tuple(
            profile.model_copy(update={"current_location_id": plan.destination_id})
            if profile.id == plan.moving_agent_id
            else profile
            for profile in world_state.agent_profiles
        )
        patches.append(
            _collection_patch(
                path="/agent_profiles",
                before=world_state.agent_profiles,
                after=profiles,
                target_id=plan.moving_agent_id,
                operation=PatchOperation.UPDATE,
                reason="Agent cognition and visible entity location stay synchronized.",
            )
        )
    committed = CommittedEvent(
        id=event_id,
        event_candidate_id=candidate.id,
        verification_result_id=verification.id,
        summary=plan.summary,
        state_diff=StateDiff(
            id=f"diff_{command.id}",
            source_event_candidate_id=candidate.id,
            committed_event_id=event_id,
            patches=tuple(patches),
        ),
    )
    resulting_world, report = ControlledStateDiffApplier().apply(
        world_state=world_state,
        committed_event=committed,
        verification_result=verification,
    )
    if not report.applied:
        raise RuntimeError("verified living-world StateDiff failed atomic application")
    return GovernedWorldOutcome(
        proposal=proposal,
        candidate=candidate.model_copy(update={"status": EventCandidateStatus.VERIFIED}),
        verification=verification,
        committed_event=committed,
        resulting_world_state=resulting_world,
        apply_report=report,
        player_message=plan.summary,
        consequences=(
            f"World time advanced to turn {next_turn} ({clock.elapsed_minutes} minutes).",
            f"Agent action committed: {', '.join(plan.actor_ids)}.",
        ),
    )


def _select_plan(
    world_state: WorldState,
    command_id: str,
    event_id: str,
    next_turn: int,
) -> _AdvancementPlan:
    propagated_claim_ids = {
        activity.source_claim_id
        for activity in world_state.world_activities
        if activity.source_claim_id is not None
    }
    claim = next(
        (
            claim
            for claim in reversed(world_state.agent_claims)
            if claim.id not in propagated_claim_ids
        ),
        None,
    )
    profiles = {profile.id: profile for profile in world_state.agent_profiles}
    if claim is not None and claim.listener_agent_id in profiles:
        recipient_id = _propagation_recipient(world_state, claim.listener_agent_id)
        if recipient_id is not None:
            speaker = profiles[claim.listener_agent_id]
            recipient = profiles[recipient_id]
            memory = MemoryRecord(
                id=f"memory_propagated_{command_id}",
                owner_agent_id=recipient.id,
                kind=MemoryKind.CONVERSATION,
                summary=(
                    f"{speaker.name} relayed a player claim to {recipient.name}: {claim.statement}"
                ),
                related_location_id=recipient.current_location_id,
                related_agent_ids=(speaker.id,),
                source_event_id=event_id,
                salience=2,
            )
            candidate = BeliefCandidate(
                id=f"belief_candidate_propagated_{command_id}",
                source_type="agent_propagation",
                source_id=speaker.id,
                claim=claim.statement,
                confidence=ConfidenceBand.LOW,
                owner_agent_id=recipient.id,
                trace_reference_id=event_id,
            )
            belief = BeliefRecord(
                id=f"belief_propagated_{command_id}",
                owner_agent_id=recipient.id,
                kind=BeliefKind.RUMOR,
                claim=claim.statement,
                truth_status=BeliefTruthStatus.UNKNOWN,
                confidence=ConfidenceBand.LOW,
                source_memory_ids=(memory.id,),
            )
            return _AdvancementPlan(
                actor_ids=(speaker.id, recipient.id),
                kind="knowledge_propagation",
                summary=(
                    f"{speaker.name}把玩家提供的说法转告给{recipient.name}；"
                    "消息仍被标记为未经证实，而不是世界事实。"
                ),
                location_id=recipient.current_location_id,
                source_claim_id=claim.id,
                propagated_candidate=candidate,
                propagated_belief=belief,
                propagated_memory=memory,
            )

    mira = profiles.get("mira")
    rowan = profiles.get("rowan")
    if next_turn % 3 == 1 and mira is not None and rowan is not None:
        return _AdvancementPlan(
            actor_ids=(mira.id, rowan.id),
            kind="cooperation",
            summary="米拉离开中央档案馆，前往议会广场与罗文核对维修记录和通行安排。",
            location_id="council_square",
            moving_agent_id=mira.id,
            destination_id="council_square",
        )
    taren = profiles.get("taren")
    if taren is not None:
        return _AdvancementPlan(
            actor_ids=(taren.id,),
            kind="independent_action",
            summary="塔伦没有等待玩家指令，独自复查了旧水道的调节器波动并更新现场标记。",
            location_id=taren.current_location_id,
        )
    fallback = next(iter(profiles.values()))
    return _AdvancementPlan(
        actor_ids=(fallback.id,),
        kind="independent_action",
        summary=f"{fallback.name}继续推进自己的当前目标。",
        location_id=fallback.current_location_id,
    )


def _propagation_recipient(world_state: WorldState, source_id: str) -> str | None:
    for relationship in sorted(world_state.agent_relationships, key=lambda item: -item.trust):
        if relationship.source_agent_id == source_id:
            return relationship.target_agent_id
        if relationship.target_agent_id == source_id:
            return relationship.source_agent_id
    return None


def _collection_patch(
    *,
    path: str,
    before,
    after,
    target_id: str,
    operation: PatchOperation,
    reason: str,
) -> StatePatch:
    return StatePatch(
        operation=operation,
        target_type=PatchTargetType.AGENT_STATE,
        target_id=target_id,
        path=path,
        before=[item.model_dump(mode="json") for item in before],
        after=[item.model_dump(mode="json") for item in after],
        reason=reason,
    )
