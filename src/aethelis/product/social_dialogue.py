from __future__ import annotations

from aethelis.product.command_contracts import ParsedPlayerIntent, PlayerCommand
from aethelis.product.content_contracts import ProductContentPackage
from aethelis.product.dialogue_expression import (
    DialogueExpressionService,
    resolve_dialogue_expression,
)
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
from aethelis.schemas.world import (
    AgentClaimRecord,
    DialogueActKind,
    DialogueTargetKind,
    PlayerDialogueTurn,
    PlayerRelationshipState,
    RequestedEffectStatus,
    WorldState,
)


def govern_social_dialogue(
    *,
    command: PlayerCommand,
    intent: ParsedPlayerIntent,
    world_state: WorldState,
    content_package: ProductContentPackage | None,
    dialogue_expression: DialogueExpressionService | None,
) -> GovernedWorldOutcome:
    current_location = world_state.player.current_location_id if world_state.player else None
    character_id = intent.target_ids[0] if len(intent.target_ids) == 1 else None
    character = next(
        (
            entity
            for entity in world_state.entities
            if entity.id == character_id
            and entity.location_id == current_location
            and "character" in entity.tags
        ),
        None,
    )
    profile = next(
        (profile for profile in world_state.agent_profiles if profile.id == character_id),
        None,
    )
    interaction_turns = tuple(
        turn
        for turn in (world_state.player.dialogue_history if world_state.player else ())
        if command.dialogue_interaction_id
        and turn.interaction_id == command.dialogue_interaction_id
    )
    act = intent.dialogue_act or DialogueActKind.QUESTION
    checks = (
        VerificationCheck(
            name="supported_social_dialogue",
            passed=intent.normalized_action == "ask_character",
            message="Social dialogue must use ask_character.",
        ),
        VerificationCheck(
            name="intent_actor_matches_command",
            passed=intent.actor_id == command.actor_id,
            message="Dialogue cannot replace the authorized player actor.",
        ),
        VerificationCheck(
            name="character_is_present",
            passed=character is not None,
            message="The listener must be present at the player's location.",
        ),
        VerificationCheck(
            name="product_agent_cognition_exists",
            passed=profile is not None,
            message="Free social dialogue requires product-persisted Agent cognition.",
        ),
        VerificationCheck(
            name="player_location_matches_world",
            passed=command.location_id == current_location and current_location is not None,
            message="Command location must match the current world location.",
        ),
        VerificationCheck(
            name="claim_has_source_text",
            passed=act != DialogueActKind.CLAIM or bool(intent.claim_text),
            message="A claim must preserve the player's asserted text.",
        ),
        VerificationCheck(
            name="interaction_target_is_consistent",
            passed=all(
                turn.target_kind == DialogueTargetKind.CHARACTER
                and turn.character_id == character_id
                for turn in interaction_turns
            ),
            message="One dialogue interaction cannot silently switch its target.",
        ),
    )
    decision = (
        VerificationDecision.COMMIT
        if world_state.player is not None
        and content_package is not None
        and character is not None
        and profile is not None
        and all(check.passed for check in checks)
        else VerificationDecision.REJECT
    )
    proposal = ActionProposal(
        id=f"proposal_{command.id}",
        proposer_agent_id=command.actor_id,
        intent=ActionIntent.DIALOGUE,
        rationale="The player addressed one present character through bounded social dialogue.",
        target_location_id=current_location,
        target_entity_ids=(character_id,) if character_id else (),
        expected_outcome=(
            "Persist a grounded reply and only the separately governed social effects."
        ),
    )
    candidate = EventCandidate(
        id=f"candidate_{command.id}",
        source_action_proposal_id=proposal.id,
        actor_agent_id=command.actor_id,
        summary=f"Player social dialogue act: {act.value}.",
        status=EventCandidateStatus.UNDER_REVIEW,
        involved_location_ids=(current_location,) if current_location else (),
        involved_entity_ids=(character_id,) if character_id else (),
    )
    verification = VerificationResult(
        id=f"verification_{command.id}",
        event_candidate_id=candidate.id,
        decision=decision,
        verifier="product_social_dialogue_governance_v1",
        checks=checks,
        reasons=(
            "The turn is grounded in one present character's product cognition."
            if decision == VerificationDecision.COMMIT
            else "The social turn lacks a valid listener or product cognition boundary."
        ,),
        risk_flags=() if decision == VerificationDecision.COMMIT else ("dialogue_rejected",),
    )
    if (
        decision != VerificationDecision.COMMIT
        or world_state.player is None
        or character is None
        or profile is None
        or content_package is None
    ):
        return GovernedWorldOutcome(
            proposal=proposal,
            candidate=candidate,
            verification=verification,
            committed_event=None,
            resulting_world_state=None,
            apply_report=None,
            player_message="This character cannot enter free dialogue in this content version.",
            consequences=(),
        )

    event_id = f"committed_{command.id}"
    safe_beliefs = tuple(
        belief
        for belief in world_state.agent_beliefs
        if belief.owner_agent_id == character.id
        and belief.kind not in {BeliefKind.PRIVATE_BELIEF, BeliefKind.REJECTED_CLAIM}
    )
    referenced_belief = _relevant_belief(command.text or "", safe_beliefs)
    authored_response = _authored_social_response(
        act=act,
        character_name=character.name,
        public_summary=character.summary,
        relevant_belief=referenced_belief,
        recent_turns=interaction_turns[-6:],
    )
    allowed_knowledge = (
        {referenced_belief.id: referenced_belief.claim} if referenced_belief is not None else {}
    )
    expression = resolve_dialogue_expression(
        dialogue_expression,
        policy=content_package.blueprint.dialogue_expression_policy,
        locale=command.locale,
        character_id=character.id,
        character_name=character.name,
        character_summary=character.summary,
        dialogue_option_id=f"social_{act.value}",
        authored_utterance=authored_response,
        allowed_knowledge=allowed_knowledge,
        conversation_context=tuple(
            {
                "player": turn.player_utterance or "[preset choice]",
                "character": turn.utterance,
            }
            for turn in interaction_turns[-6:]
        ),
    )

    relationships = _updated_player_relationships(
        world_state.player.relationships,
        character_id=character.id,
        event_id=event_id,
    )
    belief_candidate = None
    belief = None
    memory = None
    claim = None
    if act == DialogueActKind.CLAIM and intent.claim_text:
        memory = MemoryRecord(
            id=f"memory_{command.id}",
            owner_agent_id=character.id,
            kind=MemoryKind.CONVERSATION,
            summary=f"The player told {character.name}: {intent.claim_text}",
            related_location_id=current_location,
            related_agent_ids=(command.actor_id,),
            source_event_id=event_id,
            salience=3,
        )
        belief_candidate = BeliefCandidate(
            id=f"belief_candidate_{command.id}",
            source_type="player_claim",
            source_id=command.actor_id,
            claim=intent.claim_text,
            confidence=ConfidenceBand.LOW,
            owner_agent_id=character.id,
            trace_reference_id=event_id,
        )
        belief = BeliefRecord(
            id=f"belief_{command.id}",
            owner_agent_id=character.id,
            kind=BeliefKind.RUMOR,
            claim=intent.claim_text,
            truth_status=BeliefTruthStatus.UNKNOWN,
            confidence=ConfidenceBand.LOW,
            source_memory_ids=(memory.id,),
        )
        claim = AgentClaimRecord(
            id=f"claim_{command.id}",
            speaker_id=command.actor_id,
            listener_agent_id=character.id,
            statement=intent.claim_text,
            confidence=ConfidenceBand.LOW,
            belief_candidate_id=belief_candidate.id,
            committed_event_id=event_id,
        )

    turn = PlayerDialogueTurn(
        id=f"dialogue_{command.id}",
        interaction_id=command.dialogue_interaction_id or f"interaction_{command.id}",
        character_id=character.id,
        dialogue_option_id=None,
        target_kind=DialogueTargetKind.CHARACTER,
        dialogue_act=act,
        player_utterance=command.text,
        utterance=expression.utterance,
        belief_candidate_ids=(belief_candidate.id,) if belief_candidate else (),
        requested_effect_status=RequestedEffectStatus.NONE,
        committed_event_id=event_id,
        expression_evidence=expression.evidence,
    )
    patches = [
        _append_patch(
            world_state,
            path="/player/relationships",
            before=world_state.player.relationships,
            after=relationships,
            target_type=PatchTargetType.RELATIONSHIP,
            target_id=character.id,
            reason="The committed social turn advances interaction continuity.",
        ),
        _append_patch(
            world_state,
            path="/player/dialogue_history",
            before=world_state.player.dialogue_history,
            after=(*world_state.player.dialogue_history, turn),
            target_type=PatchTargetType.WORLD,
            target_id=world_state.world_id,
            reason="The grounded social turn is retained for save and fork continuity.",
        ),
    ]
    if belief_candidate and belief and memory and claim:
        patches.extend(
            (
                _append_patch(
                    world_state,
                    path="/agent_belief_candidates",
                    before=world_state.agent_belief_candidates,
                    after=(*world_state.agent_belief_candidates, belief_candidate),
                    target_type=PatchTargetType.AGENT_STATE,
                    target_id=character.id,
                    reason="Player speech becomes a listener-local candidate, not Canon.",
                ),
                _append_patch(
                    world_state,
                    path="/agent_claims",
                    before=world_state.agent_claims,
                    after=(*world_state.agent_claims, claim),
                    target_type=PatchTargetType.AGENT_STATE,
                    target_id=character.id,
                    reason="The committed claim preserves speaker and listener provenance.",
                ),
                _append_patch(
                    world_state,
                    path="/agent_memories",
                    before=world_state.agent_memories,
                    after=(*world_state.agent_memories, memory),
                    target_type=PatchTargetType.AGENT_STATE,
                    target_id=character.id,
                    reason="The listener retains the conversation as owned memory.",
                ),
                _append_patch(
                    world_state,
                    path="/agent_beliefs",
                    before=world_state.agent_beliefs,
                    after=(*world_state.agent_beliefs, belief),
                    target_type=PatchTargetType.AGENT_STATE,
                    target_id=character.id,
                    reason="The player claim becomes a low-confidence listener belief.",
                ),
            )
        )
    committed = CommittedEvent(
        id=event_id,
        event_candidate_id=candidate.id,
        verification_result_id=verification.id,
        summary=f"{command.actor_id} completed a {act.value} turn with {character.name}.",
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
        raise RuntimeError("verified social dialogue StateDiff failed atomic application")
    consequences = ["Conversation retained without granting an unsupported world effect."]
    if claim is not None:
        consequences.append(
            f"{character.name} remembers this as a low-confidence claim from the player."
        )
    return GovernedWorldOutcome(
        proposal=proposal,
        candidate=candidate.model_copy(update={"status": EventCandidateStatus.VERIFIED}),
        verification=verification,
        committed_event=committed,
        resulting_world_state=resulting_world,
        apply_report=report,
        player_message=expression.utterance,
        consequences=tuple(consequences),
    )


def _relevant_belief(text: str, beliefs: tuple[BeliefRecord, ...]) -> BeliefRecord | None:
    lowered = text.casefold()
    for belief in beliefs:
        tokens = (*belief.subject_ids, *belief.object_ids)
        if any(token.casefold() in lowered for token in tokens):
            return belief
    return None


def _authored_social_response(
    *,
    act: DialogueActKind,
    character_name: str,
    public_summary: str,
    relevant_belief: BeliefRecord | None,
    recent_turns: tuple[PlayerDialogueTurn, ...] = (),
) -> str:
    if act == DialogueActKind.GREETING:
        return f"{character_name}向你点头致意。‘你好。这里的局势还没有安定下来。’"
    if act == DialogueActKind.CLAIM:
        return "‘我听见了。我会记住这是你提供的说法，但在看到证据前不会把它当成事实。’"
    if relevant_belief is not None:
        return f"‘就我目前所知，{relevant_belief.claim}’"
    if recent_turns:
        previous = recent_turns[-1]
        previous_player = previous.player_utterance or "刚才选择的话题"
        return (
            f"{character_name}顺着先前的话继续回应：‘你刚才提到“{previous_player}”。"
            f"我上一句的意思是：{previous.utterance}’"
        )
    return f"{character_name}思考片刻。‘我只能根据自己知道和亲眼见到的事情回答。{public_summary}’"


def _updated_player_relationships(
    relationships: tuple[PlayerRelationshipState, ...],
    *,
    character_id: str,
    event_id: str,
) -> tuple[PlayerRelationshipState, ...]:
    updated = list(relationships)
    index = next(
        (index for index, item in enumerate(updated) if item.character_id == character_id),
        None,
    )
    if index is None:
        updated.append(
            PlayerRelationshipState(
                character_id=character_id,
                interaction_count=1,
                last_committed_event_id=event_id,
            )
        )
    else:
        current = updated[index]
        updated[index] = current.model_copy(
            update={
                "interaction_count": current.interaction_count + 1,
                "last_committed_event_id": event_id,
            }
        )
    return tuple(updated)


def _append_patch(
    world_state: WorldState,
    *,
    path: str,
    before,
    after,
    target_type: PatchTargetType,
    target_id: str,
    reason: str,
) -> StatePatch:
    del world_state
    return StatePatch(
        operation=PatchOperation.APPEND,
        target_type=target_type,
        target_id=target_id,
        path=path,
        before=[item.model_dump(mode="json") for item in before],
        after=[item.model_dump(mode="json") for item in after],
        reason=reason,
    )
