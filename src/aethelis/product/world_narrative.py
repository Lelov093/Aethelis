from __future__ import annotations

from aethelis.product.command_contracts import ParsedPlayerIntent, PlayerCommand
from aethelis.product.content_contracts import ProductContentPackage
from aethelis.product.governance_contracts import GovernedWorldOutcome
from aethelis.runtime.state_apply import ControlledStateDiffApplier
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
from aethelis.schemas.world import (
    CanonVisibility,
    DialogueActKind,
    DialogueTargetKind,
    PlayerDialogueTurn,
    RequestedEffectStatus,
    WorldState,
)


def govern_world_narrative(
    *,
    command: PlayerCommand,
    intent: ParsedPlayerIntent,
    world_state: WorldState,
    content_package: ProductContentPackage | None,
) -> GovernedWorldOutcome:
    current_location = world_state.player.current_location_id if world_state.player else None
    location = next(
        (item for item in world_state.locations if item.id == current_location),
        None,
    )
    interaction_turns = tuple(
        turn
        for turn in (world_state.player.dialogue_history if world_state.player else ())
        if command.dialogue_interaction_id
        and turn.interaction_id == command.dialogue_interaction_id
    )
    act = intent.dialogue_act or DialogueActKind.WORLD_OBSERVATION
    checks = (
        VerificationCheck(
            name="supported_world_narrative",
            passed=intent.normalized_action == "ask_world",
            message="World narrative must use ask_world.",
        ),
        VerificationCheck(
            name="intent_actor_matches_command",
            passed=intent.actor_id == command.actor_id,
            message="Narrative intent cannot replace the player actor.",
        ),
        VerificationCheck(
            name="player_location_is_valid",
            passed=location is not None and command.location_id == current_location,
            message="Narrative context must use the player's current location.",
        ),
        VerificationCheck(
            name="visible_world_content_available",
            passed=content_package is not None,
            message="World narrative requires visible product-world content.",
        ),
        VerificationCheck(
            name="narrative_interaction_target_is_consistent",
            passed=all(
                turn.target_kind == DialogueTargetKind.WORLD_NARRATIVE
                for turn in interaction_turns
            ),
            message="One interaction cannot silently switch into world narrative.",
        ),
    )
    decision = (
        VerificationDecision.COMMIT
        if world_state.player is not None and all(check.passed for check in checks)
        else VerificationDecision.REJECT
    )
    proposal = ActionProposal(
        id=f"proposal_{command.id}",
        proposer_agent_id=command.actor_id,
        intent=ActionIntent.OBSERVE,
        rationale="The player consulted a visibility-limited narrative projection.",
        target_location_id=current_location,
        expected_outcome=(
            "Return visible state or clarify a requested world action without mutation."
        ),
    )
    candidate = EventCandidate(
        id=f"candidate_{command.id}",
        source_action_proposal_id=proposal.id,
        actor_agent_id=command.actor_id,
        summary=f"Player world-narrative interaction: {act.value}.",
        status=EventCandidateStatus.UNDER_REVIEW,
        involved_location_ids=(current_location,) if current_location else (),
    )
    verification = VerificationResult(
        id=f"verification_{command.id}",
        event_candidate_id=candidate.id,
        decision=decision,
        verifier="product_world_narrative_governance_v1",
        checks=checks,
        reasons=(
            "The response is derived only from player-visible committed state."
            if decision == VerificationDecision.COMMIT
            else "A visibility-safe narrative context was not available.",
        ),
        risk_flags=() if decision == VerificationDecision.COMMIT else ("narrative_rejected",),
    )
    if decision != VerificationDecision.COMMIT or world_state.player is None or location is None:
        return GovernedWorldOutcome(
            proposal=proposal,
            candidate=candidate,
            verification=verification,
            committed_event=None,
            resulting_world_state=None,
            apply_report=None,
            player_message="The world cannot provide a grounded narrative response here.",
            consequences=(),
        )

    event_id = f"committed_{command.id}"
    response = _visible_response(
        world_state,
        location.id,
        location.name,
        location.summary,
        act,
        recent_turn=interaction_turns[-1] if interaction_turns else None,
    )
    effect_status = (
        RequestedEffectStatus.NEEDS_CLARIFICATION
        if act == DialogueActKind.WORLD_ACTION
        else RequestedEffectStatus.NONE
    )
    turn = PlayerDialogueTurn(
        id=f"dialogue_{command.id}",
        interaction_id=command.dialogue_interaction_id or f"interaction_{command.id}",
        character_id=None,
        dialogue_option_id=None,
        target_kind=DialogueTargetKind.WORLD_NARRATIVE,
        dialogue_act=act,
        player_utterance=command.text,
        utterance=response,
        requested_effect_status=effect_status,
        committed_event_id=event_id,
    )
    before = [item.model_dump(mode="json") for item in world_state.player.dialogue_history]
    committed = CommittedEvent(
        id=event_id,
        event_candidate_id=candidate.id,
        verification_result_id=verification.id,
        summary="The player consulted the visibility-safe world narrative.",
        state_diff=StateDiff(
            id=f"diff_{command.id}",
            source_event_candidate_id=candidate.id,
            committed_event_id=event_id,
            patches=(
                StatePatch(
                    operation=PatchOperation.APPEND,
                    target_type=PatchTargetType.WORLD,
                    target_id=world_state.world_id,
                    path="/player/dialogue_history",
                    before=before,
                    after=[*before, turn.model_dump(mode="json")],
                    reason=(
                        "The grounded narrative interaction is retained without inventing truth."
                    ),
                ),
            ),
        ),
    )
    resulting_world, report = ControlledStateDiffApplier().apply(
        world_state=world_state,
        committed_event=committed,
        verification_result=verification,
    )
    if not report.applied:
        raise RuntimeError("verified world narrative StateDiff failed atomic application")
    consequences = (
        (
            "The attempted action requires a concrete target or contextual action before it "
            "can change the world.",
        )
        if effect_status == RequestedEffectStatus.NEEDS_CLARIFICATION
        else ("The response used only player-visible committed state.",)
    )
    return GovernedWorldOutcome(
        proposal=proposal,
        candidate=candidate.model_copy(update={"status": EventCandidateStatus.VERIFIED}),
        verification=verification,
        committed_event=committed,
        resulting_world_state=resulting_world,
        apply_report=report,
        player_message=response,
        consequences=consequences,
    )


def _visible_response(
    world_state: WorldState,
    location_id: str,
    location_name: str,
    location_summary: str,
    act: DialogueActKind,
    recent_turn: PlayerDialogueTurn | None = None,
) -> str:
    characters = [
        entity.name
        for entity in world_state.entities
        if entity.location_id == location_id and "character" in entity.tags
    ]
    player_id = world_state.player.id if world_state.player else None
    resources = [
        resource.name
        for resource in world_state.resources
        if resource.location_id == location_id
        and player_id in resource.discovery_state.discovered_by_agent_ids
    ]
    public_signals = [
        fact.statement
        for fact in world_state.canon_facts
        if fact.visibility == CanonVisibility.PUBLIC and fact.location_id in {None, location_id}
    ]
    visible = [f"你位于{location_name}。{location_summary}"]
    if characters:
        visible.append(f"在场人物：{'、'.join(characters)}。")
    if resources:
        visible.append(f"你已经确认的现场资源：{'、'.join(resources)}。")
    if public_signals:
        visible.append(f"公开信息表明：{public_signals[0]}")
    if recent_turn is not None and act == DialogueActKind.WORLD_OBSERVATION:
        visible.append(
            f"你上一轮提到“{(recent_turn.player_utterance or '先前的预设话题')[:120]}”；"
            f"当时可见的回应是：{recent_turn.utterance[:180]}"
        )
    if act == DialogueActKind.WORLD_ACTION:
        visible.append("你的话包含一次行动尝试；旁白不会替你执行，请明确目标或使用现场行动。")
    return "".join(visible)
