from __future__ import annotations

from aethelis.product.command_contracts import ParsedPlayerIntent, PlayerCommand
from aethelis.product.content_contracts import (
    ProductContentPackage,
    ProductResourceExchangeOption,
)
from aethelis.product.dialogue_expression import (
    DialogueExpressionService,
    resolve_dialogue_expression,
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
    PlayerCommitment,
    PlayerDialogueTurn,
    PlayerInventoryItem,
    WorldState,
)


def govern_resource_exchange(
    *,
    command: PlayerCommand,
    intent: ParsedPlayerIntent,
    world_state: WorldState,
    content_package: ProductContentPackage | None,
    dialogue_expression: DialogueExpressionService | None = None,
) -> GovernedWorldOutcome:
    current_location = world_state.player.current_location_id if world_state.player else None
    character_id = intent.target_ids[0] if len(intent.target_ids) == 1 else None
    option = _find_option(content_package, current_location, character_id)
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
    resource = next(
        (
            item
            for item in world_state.resources
            if option is not None and item.id == option.resource_id
        ),
        None,
    )
    known_ids = (
        {record.id for record in world_state.player.knowledge}
        if world_state.player
        else set()
    )
    relationship = next(
        (
            record
            for record in world_state.player.relationships
            if record.character_id == character_id
        ),
        None,
    ) if world_state.player else None
    commitment_ids = {
        record.id for record in world_state.player.commitments
    } if world_state.player else set()
    inventory_resource_ids = {
        record.resource_id for record in world_state.player.inventory
    } if world_state.player else set()
    checks = (
        VerificationCheck(
            name="supported_product_action",
            passed=intent.normalized_action == "negotiate_resource",
            message="Action must be negotiate_resource.",
        ),
        VerificationCheck(
            name="intent_actor_matches_command",
            passed=intent.actor_id == command.actor_id,
            message="Normalized intent cannot replace the authorized player actor.",
        ),
        VerificationCheck(
            name="content_exchange_option_exists",
            passed=option is not None,
            message="Exchange must use a versioned option for this character and location.",
        ),
        VerificationCheck(
            name="character_is_present",
            passed=character is not None,
            message="The counterparty must be present at the player's location.",
        ),
        VerificationCheck(
            name="player_location_matches_world",
            passed=command.location_id == current_location and current_location is not None,
            message="Command location must match the player's current world location.",
        ),
        VerificationCheck(
            name="prerequisite_knowledge_is_known",
            passed=(
                option is not None
                and set(option.prerequisite_knowledge_ids).issubset(known_ids)
            ),
            message="The player must know the authored facts required for this negotiation.",
        ),
        VerificationCheck(
            name="minimum_relationship_is_met",
            passed=(
                option is not None
                and relationship is not None
                and relationship.trust >= option.minimum_trust
            ),
            message="The player relationship does not yet support this exchange.",
        ),
        VerificationCheck(
            name="resource_stock_is_available",
            passed=(
                option is not None
                and resource is not None
                and resource.location_id == current_location
                and resource.quantity >= option.quantity
            ),
            message="The requested local resource stock is unavailable.",
        ),
        VerificationCheck(
            name="exchange_has_not_been_claimed",
            passed=(
                option is not None
                and option.commitment_id not in commitment_ids
                and option.resource_id not in inventory_resource_ids
            ),
            message="This one-time governed exchange has already been claimed.",
        ),
    )
    decision = (
        VerificationDecision.COMMIT
        if option is not None
        and character is not None
        and resource is not None
        and all(check.passed for check in checks)
        else VerificationDecision.REJECT
    )
    proposal = ActionProposal(
        id=f"proposal_{command.id}",
        proposer_agent_id=command.actor_id,
        intent=ActionIntent.NEGOTIATE,
        rationale="Player accepted a versioned resource exchange with an explicit obligation.",
        target_location_id=current_location,
        target_entity_ids=tuple(
            target_id
            for target_id in (character_id, resource.id if resource else None)
            if target_id
        ),
        expected_outcome="Transfer bounded stock to player custody and record its commitment cost.",
    )
    candidate = EventCandidate(
        id=f"candidate_{command.id}",
        source_action_proposal_id=proposal.id,
        actor_agent_id=command.actor_id,
        summary="Player negotiates a bounded Mistgate resource allotment.",
        status=EventCandidateStatus.UNDER_REVIEW,
        involved_location_ids=(current_location,) if current_location else (),
        involved_entity_ids=(character_id,) if character_id else (),
    )
    verification = VerificationResult(
        id=f"verification_{command.id}",
        event_candidate_id=candidate.id,
        decision=decision,
        verifier="product_resource_exchange_governance_v1",
        checks=checks,
        reasons=(
            "Knowledge, trust, stock, custody, and commitment terms permit this exchange."
            if decision == VerificationDecision.COMMIT
            else "The resource exchange cannot commit under current world state.",
        ),
        risk_flags=() if decision == VerificationDecision.COMMIT else ("exchange_rejected",),
    )
    if (
        decision != VerificationDecision.COMMIT
        or option is None
        or character is None
        or resource is None
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
            player_message="The exchange terms are not available in the current world state.",
            consequences=(),
        )

    event_id = f"committed_{command.id}"
    inventory_item = PlayerInventoryItem(
        id=f"inventory_{option.resource_id}",
        resource_id=option.resource_id,
        quantity=option.quantity,
        acquired_from_entity_id=character.id,
        acquired_event_id=event_id,
    )
    commitment = PlayerCommitment(
        id=option.commitment_id,
        counterparty_entity_id=character.id,
        description=content_package.text(option.commitment_description_key, command.locale),
        related_resource_ids=(option.resource_id,),
        committed_event_id=event_id,
    )
    relationships = list(world_state.player.relationships)
    relationship_index = next(
        index
        for index, record in enumerate(relationships)
        if record.character_id == character.id
    )
    relationships[relationship_index] = relationship.model_copy(
        update={
            "interaction_count": relationship.interaction_count + 1,
            "last_committed_event_id": event_id,
        }
    )
    authored_response = content_package.text(option.response_key, command.locale)
    commitment_description = content_package.text(
        option.commitment_description_key,
        command.locale,
    )
    known_by_id = {record.id: record.statement for record in world_state.player.knowledge}
    expression = resolve_dialogue_expression(
        dialogue_expression,
        policy=content_package.blueprint.dialogue_expression_policy,
        locale=command.locale,
        character_id=character.id,
        character_name=character.name,
        character_summary=character.summary,
        dialogue_option_id=option.id,
        authored_utterance=authored_response,
        allowed_knowledge={
            knowledge_id: known_by_id[knowledge_id]
            for knowledge_id in option.prerequisite_knowledge_ids
        },
        required_terms=(
            f"{resource.name} x{option.quantity}",
            commitment_description,
        ),
    )
    response = expression.utterance
    dialogue_turn = PlayerDialogueTurn(
        id=f"dialogue_{command.id}",
        interaction_id=command.dialogue_interaction_id or f"interaction_{command.id}",
        character_id=character.id,
        dialogue_option_id=option.id,
        utterance=response,
        knowledge_record_ids=option.prerequisite_knowledge_ids,
        committed_event_id=event_id,
        expression_evidence=expression.evidence,
    )
    before_inventory = _dump_records(world_state.player.inventory)
    before_commitments = _dump_records(world_state.player.commitments)
    before_relationships = _dump_records(world_state.player.relationships)
    before_dialogue = _dump_records(world_state.player.dialogue_history)
    committed = CommittedEvent(
        id=event_id,
        event_candidate_id=candidate.id,
        verification_result_id=verification.id,
        summary=(
            f"{command.actor_id} received {option.quantity} {resource.name} "
            f"from {character.name}."
        ),
        state_diff=StateDiff(
            id=f"diff_{command.id}",
            source_event_candidate_id=candidate.id,
            committed_event_id=event_id,
            patches=(
                StatePatch(
                    operation=PatchOperation.DECREMENT,
                    target_type=PatchTargetType.RESOURCE,
                    target_id=resource.id,
                    path=f"/resource/{resource.id}/quantity",
                    before=resource.quantity,
                    after=resource.quantity - option.quantity,
                    reason="Verified exchange removed the allotted quantity from local stock.",
                ),
                StatePatch(
                    operation=PatchOperation.APPEND,
                    target_type=PatchTargetType.WORLD,
                    target_id=world_state.world_id,
                    path="/player/inventory",
                    before=before_inventory,
                    after=[*before_inventory, inventory_item.model_dump(mode="json")],
                    reason="Verified exchange placed the allotment in player custody.",
                ),
                StatePatch(
                    operation=PatchOperation.APPEND,
                    target_type=PatchTargetType.WORLD,
                    target_id=world_state.world_id,
                    path="/player/commitments",
                    before=before_commitments,
                    after=[*before_commitments, commitment.model_dump(mode="json")],
                    reason="Verified exchange recorded its explicit deferred cost.",
                ),
                StatePatch(
                    operation=PatchOperation.APPEND,
                    target_type=PatchTargetType.RELATIONSHIP,
                    target_id=character.id,
                    path="/player/relationships",
                    before=before_relationships,
                    after=_dump_records(relationships),
                    reason="Verified negotiation advanced relationship interaction history.",
                ),
                StatePatch(
                    operation=PatchOperation.APPEND,
                    target_type=PatchTargetType.WORLD,
                    target_id=world_state.world_id,
                    path="/player/dialogue_history",
                    before=before_dialogue,
                    after=[*before_dialogue, dialogue_turn.model_dump(mode="json")],
                    reason="Verified negotiation dialogue was retained for continuity.",
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
        raise RuntimeError("verified resource exchange StateDiff failed atomic application")
    return GovernedWorldOutcome(
        proposal=proposal,
        candidate=candidate.model_copy(update={"status": EventCandidateStatus.VERIFIED}),
        verification=verification,
        committed_event=committed,
        resulting_world_state=resulting_world,
        apply_report=report,
        player_message=response,
        consequences=(
            f"Resource acquired: {resource.name} x{option.quantity}",
            f"Commitment recorded: {commitment.description}",
            f"Local stock remaining: {resource.quantity - option.quantity}",
        ),
    )


def _find_option(
    package: ProductContentPackage | None,
    location_id: str | None,
    character_id: str | None,
) -> ProductResourceExchangeOption | None:
    if package is None or location_id is None or character_id is None:
        return None
    matches = tuple(
        option
        for option in package.blueprint.resource_exchange_options
        if option.action_id == "negotiate_resource"
        and option.location_id == location_id
        and option.character_id == character_id
    )
    return matches[0] if len(matches) == 1 else None


def _dump_records(records) -> list[dict[str, object]]:
    return [record.model_dump(mode="json") for record in records]
