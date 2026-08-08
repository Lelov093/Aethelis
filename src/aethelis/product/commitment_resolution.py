from __future__ import annotations

from aethelis.product.command_contracts import ParsedPlayerIntent, PlayerCommand
from aethelis.product.content_contracts import (
    ProductCommitmentBreachOption,
    ProductContentPackage,
    ProductRepairOption,
)
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
    DialogueExpressionEvidence,
    PlayerCommitmentStatus,
    PlayerDialogueTurn,
    PlayerKnowledgeKind,
    PlayerKnowledgeRecord,
    WorldState,
)


def govern_repair(
    *,
    command: PlayerCommand,
    intent: ParsedPlayerIntent,
    world_state: WorldState,
    content_package: ProductContentPackage | None,
) -> GovernedWorldOutcome:
    current_location = world_state.player.current_location_id if world_state.player else None
    target_id = intent.target_ids[0] if len(intent.target_ids) == 1 else None
    option = _find_repair_option(content_package, current_location, target_id)
    target = next(
        (
            entity
            for entity in world_state.entities
            if entity.id == target_id and entity.location_id == current_location
        ),
        None,
    )
    inventory_item = (
        next(
            (
                item
                for item in world_state.player.inventory
                if option is not None and item.resource_id == option.resource_id
            ),
            None,
        )
        if world_state.player
        else None
    )
    commitment = (
        next(
            (
                item
                for item in world_state.player.commitments
                if option is not None and item.id == option.commitment_id
            ),
            None,
        )
        if world_state.player
        else None
    )
    target_tags = set(target.tags) if target is not None else set()
    knowledge_ids = (
        {item.id for item in world_state.player.knowledge} if world_state.player else set()
    )
    checks = (
        VerificationCheck(
            name="supported_product_action",
            passed=intent.normalized_action == "repair_regulator",
            message="Action must be repair_regulator.",
        ),
        VerificationCheck(
            name="intent_actor_matches_command",
            passed=intent.actor_id == command.actor_id,
            message="Normalized intent cannot replace the authorized player actor.",
        ),
        VerificationCheck(
            name="content_repair_option_exists",
            passed=option is not None,
            message="Repair must use a versioned option for this target and location.",
        ),
        VerificationCheck(
            name="repair_target_is_present",
            passed=target is not None,
            message="The repair target must be present at the player's location.",
        ),
        VerificationCheck(
            name="player_location_matches_world",
            passed=command.location_id == current_location and current_location is not None,
            message="Command location must match the player's current world location.",
        ),
        VerificationCheck(
            name="required_resource_is_held",
            passed=(
                option is not None
                and inventory_item is not None
                and inventory_item.quantity >= option.quantity
            ),
            message="The player must hold the required repair material.",
        ),
        VerificationCheck(
            name="related_commitment_exists",
            passed=(
                commitment is not None
                and commitment.status
                in {PlayerCommitmentStatus.ACTIVE, PlayerCommitmentStatus.BROKEN}
            ),
            message="The repair must resolve or follow the recorded resource obligation.",
        ),
        VerificationCheck(
            name="repair_target_state_is_eligible",
            passed=(option is not None and set(option.required_target_tags).issubset(target_tags)),
            message="The target is not in a repairable state.",
        ),
        VerificationCheck(
            name="repair_outcome_is_new",
            passed=(
                option is not None
                and not set(option.result_target_tags).issubset(target_tags)
                and option.knowledge_id not in knowledge_ids
            ),
            message="This repair outcome has already been committed.",
        ),
    )
    decision = (
        VerificationDecision.COMMIT
        if option is not None
        and target is not None
        and inventory_item is not None
        and commitment is not None
        and all(check.passed for check in checks)
        else VerificationDecision.REJECT
    )
    proposal = ActionProposal(
        id=f"proposal_{command.id}",
        proposer_agent_id=command.actor_id,
        intent=ActionIntent.REPAIR,
        rationale=(
            "Player attempts a content-defined repair with held material and an explicit "
            "obligation."
        ),
        target_location_id=current_location,
        target_entity_ids=(target_id,) if target_id else (),
        expected_outcome=(
            "Consume the repair material, record verified progress, and resolve the "
            "obligation when eligible."
        ),
    )
    candidate = EventCandidate(
        id=f"candidate_{command.id}",
        source_action_proposal_id=proposal.id,
        actor_agent_id=command.actor_id,
        summary="Player attempts a bounded Mistgate regulator repair.",
        status=EventCandidateStatus.UNDER_REVIEW,
        involved_location_ids=(current_location,) if current_location else (),
        involved_entity_ids=(target_id,) if target_id else (),
    )
    verification = VerificationResult(
        id=f"verification_{command.id}",
        event_candidate_id=candidate.id,
        decision=decision,
        verifier="product_repair_governance_v1",
        checks=checks,
        reasons=(
            "Location, material custody, target state, obligation, and outcome tags permit repair."
            if decision == VerificationDecision.COMMIT
            else "The repair cannot commit under current world state.",
        ),
        risk_flags=() if decision == VerificationDecision.COMMIT else ("repair_rejected",),
    )
    if (
        decision != VerificationDecision.COMMIT
        or option is None
        or target is None
        or inventory_item is None
        or commitment is None
        or world_state.player is None
        or content_package is None
    ):
        return GovernedWorldOutcome(
            proposal=proposal,
            candidate=candidate,
            verification=verification,
            committed_event=None,
            resulting_world_state=None,
            apply_report=None,
            player_message=_repair_recovery_message(
                content_package, command.locale, checks
            ),
            consequences=(),
        )

    event_id = f"committed_{command.id}"
    inventory_after = []
    for item in world_state.player.inventory:
        if item.id != inventory_item.id:
            inventory_after.append(item)
        elif item.quantity > option.quantity:
            inventory_after.append(
                item.model_copy(update={"quantity": item.quantity - option.quantity})
            )
    commitments_after = list(world_state.player.commitments)
    commitment_index = next(
        index for index, item in enumerate(commitments_after) if item.id == commitment.id
    )
    if commitment.status == PlayerCommitmentStatus.ACTIVE:
        commitments_after[commitment_index] = commitment.model_copy(
            update={
                "status": PlayerCommitmentStatus.FULFILLED,
                "resolved_event_id": event_id,
            }
        )
    target_tags_after = tuple(
        dict.fromkeys(
            (
                *(tag for tag in target.tags if tag != "unstable"),
                *option.result_target_tags,
            )
        )
    )
    knowledge = PlayerKnowledgeRecord(
        id=option.knowledge_id,
        kind=PlayerKnowledgeKind.CONFIRMED_FACT,
        statement=content_package.text(option.knowledge_statement_key, command.locale),
        source_entity_id=target.id,
        subject_ids=(target.id,),
        confidence="high",
        committed_event_id=event_id,
    )
    before_inventory = _dump_records(world_state.player.inventory)
    before_commitments = _dump_records(world_state.player.commitments)
    before_knowledge = _dump_records(world_state.player.knowledge)
    committed = CommittedEvent(
        id=event_id,
        event_candidate_id=candidate.id,
        verification_result_id=verification.id,
        summary=f"{command.actor_id} advanced the repair of {target.name}.",
        tags=option.committed_event_tags,
        state_diff=StateDiff(
            id=f"diff_{command.id}",
            source_event_candidate_id=candidate.id,
            committed_event_id=event_id,
            patches=(
                StatePatch(
                    operation=PatchOperation.UPDATE,
                    target_type=PatchTargetType.WORLD,
                    target_id=world_state.world_id,
                    path="/player/inventory",
                    before=before_inventory,
                    after=_dump_records(inventory_after),
                    reason="Verified repair consumed the held stabilizer material.",
                ),
                StatePatch(
                    operation=PatchOperation.UPDATE,
                    target_type=PatchTargetType.WORLD,
                    target_id=world_state.world_id,
                    path="/player/commitments",
                    before=before_commitments,
                    after=_dump_records(commitments_after),
                    reason=(
                        "Verified repair fulfilled the active evidence obligation when applicable."
                    ),
                ),
                StatePatch(
                    operation=PatchOperation.UPDATE,
                    target_type=PatchTargetType.ENTITY,
                    target_id=target.id,
                    path=f"/entity/{target.id}/tags",
                    before=list(target.tags),
                    after=list(target_tags_after),
                    reason="Verified repair persisted pressure containment and outcome evidence.",
                ),
                StatePatch(
                    operation=PatchOperation.APPEND,
                    target_type=PatchTargetType.WORLD,
                    target_id=world_state.world_id,
                    path="/player/knowledge",
                    before=before_knowledge,
                    after=[*before_knowledge, knowledge.model_dump(mode="json")],
                    reason="The player observed verified repair progress.",
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
        raise RuntimeError("verified repair StateDiff failed atomic application")
    consequences = [
        f"Resource consumed: {inventory_item.resource_id} x{option.quantity}",
        f"Repair progress recorded: {target.name}",
        f"Outcome reached: {option.outcome_id}",
    ]
    if commitment.status == PlayerCommitmentStatus.ACTIVE:
        consequences.insert(1, f"Commitment fulfilled: {commitment.description}")
    else:
        consequences.insert(1, f"Commitment remains broken: {commitment.description}")
    return GovernedWorldOutcome(
        proposal=proposal,
        candidate=candidate.model_copy(update={"status": EventCandidateStatus.VERIFIED}),
        verification=verification,
        committed_event=committed,
        resulting_world_state=resulting_world,
        apply_report=report,
        player_message=content_package.text(option.response_key, command.locale),
        consequences=tuple(consequences),
    )


def govern_commitment_breach(
    *,
    command: PlayerCommand,
    intent: ParsedPlayerIntent,
    world_state: WorldState,
    content_package: ProductContentPackage | None,
) -> GovernedWorldOutcome:
    current_location = world_state.player.current_location_id if world_state.player else None
    character_id = intent.target_ids[0] if len(intent.target_ids) == 1 else None
    option = _find_breach_option(content_package, current_location, character_id)
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
    commitment = (
        next(
            (
                item
                for item in world_state.player.commitments
                if option is not None and item.id == option.commitment_id
            ),
            None,
        )
        if world_state.player
        else None
    )
    relationship = (
        next(
            (
                item
                for item in world_state.player.relationships
                if item.character_id == character_id
            ),
            None,
        )
        if world_state.player
        else None
    )
    checks = (
        VerificationCheck(
            name="supported_product_action",
            passed=intent.normalized_action == "break_commitment",
            message="Action must be break_commitment.",
        ),
        VerificationCheck(
            name="intent_actor_matches_command",
            passed=intent.actor_id == command.actor_id,
            message="Normalized intent cannot replace the authorized player actor.",
        ),
        VerificationCheck(
            name="content_breach_option_exists",
            passed=option is not None,
            message="Commitment breach must use a versioned option.",
        ),
        VerificationCheck(
            name="counterparty_is_present",
            passed=character is not None,
            message="The commitment counterparty must be present.",
        ),
        VerificationCheck(
            name="player_location_matches_world",
            passed=command.location_id == current_location and current_location is not None,
            message="Command location must match the player's current world location.",
        ),
        VerificationCheck(
            name="commitment_is_active",
            passed=(commitment is not None and commitment.status == PlayerCommitmentStatus.ACTIVE),
            message="Only an active commitment can be explicitly broken.",
        ),
        VerificationCheck(
            name="relationship_exists",
            passed=relationship is not None,
            message="The social consequence requires the existing counterparty relationship.",
        ),
    )
    decision = (
        VerificationDecision.COMMIT
        if option is not None
        and character is not None
        and commitment is not None
        and relationship is not None
        and all(check.passed for check in checks)
        else VerificationDecision.REJECT
    )
    proposal = ActionProposal(
        id=f"proposal_{command.id}",
        proposer_agent_id=command.actor_id,
        intent=ActionIntent.NEGOTIATE,
        rationale=(
            "Player explicitly renounces an active resource obligation before its counterparty."
        ),
        target_location_id=current_location,
        target_entity_ids=(character_id,) if character_id else (),
        expected_outcome="Mark the commitment broken and persist its relationship consequence.",
    )
    candidate = EventCandidate(
        id=f"candidate_{command.id}",
        source_action_proposal_id=proposal.id,
        actor_agent_id=command.actor_id,
        summary="Player explicitly breaks a Mistgate repair commitment.",
        status=EventCandidateStatus.UNDER_REVIEW,
        involved_location_ids=(current_location,) if current_location else (),
        involved_entity_ids=(character_id,) if character_id else (),
    )
    verification = VerificationResult(
        id=f"verification_{command.id}",
        event_candidate_id=candidate.id,
        decision=decision,
        verifier="product_commitment_breach_governance_v1",
        checks=checks,
        reasons=(
            "The active obligation and present counterparty permit an explicit breach."
            if decision == VerificationDecision.COMMIT
            else "The commitment cannot be broken in the current world state.",
        ),
        risk_flags=() if decision == VerificationDecision.COMMIT else ("breach_rejected",),
    )
    if (
        decision != VerificationDecision.COMMIT
        or option is None
        or character is None
        or commitment is None
        or relationship is None
        or world_state.player is None
        or content_package is None
    ):
        return GovernedWorldOutcome(
            proposal=proposal,
            candidate=candidate,
            verification=verification,
            committed_event=None,
            resulting_world_state=None,
            apply_report=None,
            player_message="This commitment cannot be renounced in the current world state.",
            consequences=(),
        )

    event_id = f"committed_{command.id}"
    commitments_after = list(world_state.player.commitments)
    commitment_index = next(
        index for index, item in enumerate(commitments_after) if item.id == commitment.id
    )
    commitments_after[commitment_index] = commitment.model_copy(
        update={
            "status": PlayerCommitmentStatus.BROKEN,
            "resolved_event_id": event_id,
        }
    )
    relationships_after = list(world_state.player.relationships)
    relationship_index = next(
        index
        for index, item in enumerate(relationships_after)
        if item.character_id == relationship.character_id
    )
    relationships_after[relationship_index] = relationship.model_copy(
        update={
            "trust": max(-5, relationship.trust + option.relationship_delta),
            "interaction_count": relationship.interaction_count + 1,
            "last_committed_event_id": event_id,
        }
    )
    response = content_package.text(option.response_key, command.locale)
    dialogue_turn = PlayerDialogueTurn(
        id=f"dialogue_{command.id}",
        interaction_id=command.dialogue_interaction_id or f"interaction_{command.id}",
        character_id=character.id,
        dialogue_option_id=option.id,
        utterance=response,
        committed_event_id=event_id,
        expression_evidence=DialogueExpressionEvidence(source="authored"),
    )
    before_commitments = _dump_records(world_state.player.commitments)
    before_relationships = _dump_records(world_state.player.relationships)
    before_dialogue = _dump_records(world_state.player.dialogue_history)
    committed = CommittedEvent(
        id=event_id,
        event_candidate_id=candidate.id,
        verification_result_id=verification.id,
        summary=f"{command.actor_id} broke a commitment to {character.name}.",
        tags=(f"commitment_broken:{commitment.id}",),
        state_diff=StateDiff(
            id=f"diff_{command.id}",
            source_event_candidate_id=candidate.id,
            committed_event_id=event_id,
            patches=(
                StatePatch(
                    operation=PatchOperation.UPDATE,
                    target_type=PatchTargetType.WORLD,
                    target_id=world_state.world_id,
                    path="/player/commitments",
                    before=before_commitments,
                    after=_dump_records(commitments_after),
                    reason="The player explicitly broke the active obligation.",
                ),
                StatePatch(
                    operation=PatchOperation.APPEND,
                    target_type=PatchTargetType.RELATIONSHIP,
                    target_id=character.id,
                    path="/player/relationships",
                    before=before_relationships,
                    after=_dump_records(relationships_after),
                    reason="The explicit breach persisted its social consequence.",
                ),
                StatePatch(
                    operation=PatchOperation.APPEND,
                    target_type=PatchTargetType.WORLD,
                    target_id=world_state.world_id,
                    path="/player/dialogue_history",
                    before=before_dialogue,
                    after=[*before_dialogue, dialogue_turn.model_dump(mode="json")],
                    reason="The breach exchange remains visible in dialogue continuity.",
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
        raise RuntimeError("verified commitment breach StateDiff failed atomic application")
    return GovernedWorldOutcome(
        proposal=proposal,
        candidate=candidate.model_copy(update={"status": EventCandidateStatus.VERIFIED}),
        verification=verification,
        committed_event=committed,
        resulting_world_state=resulting_world,
        apply_report=report,
        player_message=response,
        consequences=(
            f"Commitment broken: {commitment.description}",
            f"Relationship changed: {character.name} {option.relationship_delta:+d}",
            "Held repair materials remain in player custody.",
        ),
    )


def _repair_recovery_message(
    package: ProductContentPackage | None,
    locale: str,
    checks: tuple[VerificationCheck, ...],
) -> str:
    failed = {check.name for check in checks if not check.passed}
    if failed & {"required_resource_is_held", "related_commitment_exists"}:
        return _recovery_text(
            package,
            "recovery_missing_parts",
            locale,
            "Find the stabilizer parts and establish their repair obligation at Market Row.",
        )
    return "The regulator cannot be repaired with the current evidence and resources."


def _recovery_text(
    package: ProductContentPackage | None,
    recovery_id: str,
    locale: str,
    fallback: str,
) -> str:
    if package is None:
        return fallback
    recovery = next(
        (item for item in package.blueprint.recovery_paths if item.id == recovery_id),
        None,
    )
    return package.text(recovery.guidance_key, locale) if recovery else fallback


def _find_repair_option(
    package: ProductContentPackage | None,
    location_id: str | None,
    target_id: str | None,
) -> ProductRepairOption | None:
    if package is None or location_id is None or target_id is None:
        return None
    matches = tuple(
        option
        for option in package.blueprint.repair_options
        if option.action_id == "repair_regulator"
        and option.location_id == location_id
        and option.target_entity_id == target_id
    )
    return matches[0] if len(matches) == 1 else None


def _find_breach_option(
    package: ProductContentPackage | None,
    location_id: str | None,
    character_id: str | None,
) -> ProductCommitmentBreachOption | None:
    if package is None or location_id is None or character_id is None:
        return None
    matches = tuple(
        option
        for option in package.blueprint.commitment_breach_options
        if option.action_id == "break_commitment"
        and option.location_id == location_id
        and option.character_id == character_id
    )
    return matches[0] if len(matches) == 1 else None


def _dump_records(records) -> list[dict[str, object]]:
    return [record.model_dump(mode="json") for record in records]
