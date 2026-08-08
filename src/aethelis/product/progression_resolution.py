from __future__ import annotations

from collections.abc import Iterable

from aethelis.product.command_contracts import ParsedPlayerIntent, PlayerCommand
from aethelis.product.content_contracts import (
    ProductContentPackage,
    ProductFinalRepairOption,
    ProductResourceReleaseOption,
    ProductResourceValidationOption,
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
    PlayerInventoryItem,
    PlayerKnowledgeKind,
    PlayerKnowledgeRecord,
    WorldState,
)


def govern_resource_release(
    *,
    command: PlayerCommand,
    intent: ParsedPlayerIntent,
    world_state: WorldState,
    content_package: ProductContentPackage | None,
) -> GovernedWorldOutcome:
    location_id = world_state.player.current_location_id if world_state.player else None
    character_id = intent.target_ids[0] if len(intent.target_ids) == 1 else None
    option = _release_option(content_package, location_id, character_id)
    character = _local_entity(world_state, character_id, location_id, required_tag="character")
    container = _local_entity(
        world_state,
        option.container_entity_id if option else None,
        location_id,
    )
    resource = _resource(world_state, option.resource_id if option else None)
    player = world_state.player
    discovered = bool(
        resource
        and (
            not option
            or not option.required_discovery
            or command.actor_id in resource.discovery_state.discovered_by_agent_ids
        )
    )
    checks = (
        _check("supported_product_action", intent.normalized_action == "request_calibration_key"),
        _check("intent_actor_matches_command", intent.actor_id == command.actor_id),
        _check("content_release_option_exists", option is not None),
        _check("authorized_character_is_present", character is not None),
        _check("owned_container_is_present", container is not None),
        _check("player_location_matches_world", command.location_id == location_id),
        _check("resource_was_discovered", discovered),
        _check(
            "resource_is_still_available",
            bool(resource and option and resource.quantity >= option.quantity),
        ),
        _check(
            "resource_not_already_held",
            bool(
                player
                and option
                and all(item.resource_id != option.resource_id for item in player.inventory)
            ),
        ),
        _check(
            "release_is_new",
            bool(
                player
                and option
                and all(item.id != option.knowledge_id for item in player.knowledge)
            ),
        ),
    )
    proposal, candidate, verification = _governance_records(
        command=command,
        intent=ActionIntent.NEGOTIATE,
        location_id=location_id,
        target_ids=(character_id,) if character_id else (),
        summary="Player requests the governed release of the Mistgate calibration key.",
        expected="Transfer the discovered key from its lawful container into player custody.",
        verifier="product_resource_release_governance_v1",
        checks=checks,
    )
    if not all(check.passed for check in checks) or not all(
        (option, character, container, resource, player, content_package)
    ):
        return _rejected(
            proposal,
            candidate,
            verification,
            _recovery_text(
                content_package,
                "recovery_missing_key",
                command.locale,
                "Investigate the workshop safe before asking Ivo to release the key.",
            ),
        )

    event_id = f"committed_{command.id}"
    inventory_after = [*player.inventory]
    inventory_after.append(
        PlayerInventoryItem(
            id=f"inventory_{option.resource_id}",
            resource_id=option.resource_id,
            quantity=option.quantity,
            acquired_from_entity_id=container.id,
            acquired_event_id=event_id,
        )
    )
    container_tags_after = tuple(
        dict.fromkeys(
            (*(tag for tag in container.tags if tag != "locked"), "access_granted", "key_released")
        )
    )
    knowledge = _knowledge(
        option.knowledge_id,
        content_package.text(option.knowledge_statement_key, command.locale),
        character.id,
        (resource.id, container.id),
        event_id,
    )
    committed = _committed_event(
        command=command,
        candidate=candidate,
        verification=verification,
        summary=f"{character.name} released {resource.name} from {container.name}.",
        tags=("calibration_key_released",),
        patches=(
            StatePatch(
                operation=PatchOperation.DECREMENT,
                target_type=PatchTargetType.RESOURCE,
                target_id=resource.id,
                path=f"/resource/{resource.id}/quantity",
                before=resource.quantity,
                after=resource.quantity - option.quantity,
                reason="The authorized custodian released the discovered key.",
            ),
            StatePatch(
                operation=PatchOperation.APPEND,
                target_type=PatchTargetType.WORLD,
                target_id=world_state.world_id,
                path="/player/inventory",
                before=_dump(player.inventory),
                after=_dump(inventory_after),
                reason="The released key entered player custody.",
            ),
            StatePatch(
                operation=PatchOperation.UPDATE,
                target_type=PatchTargetType.ENTITY,
                target_id=container.id,
                path=f"/entity/{container.id}/tags",
                before=list(container.tags),
                after=list(container_tags_after),
                reason="The lawful release persisted the safe access state.",
            ),
            _knowledge_patch(
                world_state, player.knowledge, knowledge, "The lawful key transfer was observed."
            ),
        ),
    )
    return _applied(
        world_state,
        committed,
        proposal,
        candidate,
        verification,
        content_package.text(option.response_key, command.locale),
        (f"Key item acquired: {resource.name}", f"Container access recorded: {container.name}"),
    )


def govern_resource_validation(
    *,
    command: PlayerCommand,
    intent: ParsedPlayerIntent,
    world_state: WorldState,
    content_package: ProductContentPackage | None,
) -> GovernedWorldOutcome:
    location_id = world_state.player.current_location_id if world_state.player else None
    resource_id = intent.target_ids[0] if len(intent.target_ids) == 1 else None
    option = _validation_option(content_package, location_id, resource_id)
    resource = _resource(world_state, resource_id)
    player = world_state.player
    discovered = bool(
        resource
        and (
            not option
            or not option.required_discovery
            or command.actor_id in resource.discovery_state.discovered_by_agent_ids
        )
    )
    checks = (
        _check("supported_product_action", intent.normalized_action == "validate_gate_lens"),
        _check("intent_actor_matches_command", intent.actor_id == command.actor_id),
        _check("content_validation_option_exists", option is not None),
        _check("player_location_matches_world", command.location_id == location_id),
        _check(
            "resource_is_local",
            bool(resource and resource.location_id == location_id and resource.quantity > 0),
        ),
        _check("resource_was_discovered", discovered),
        _check(
            "validation_is_new",
            bool(
                player
                and option
                and all(item.id != option.knowledge_id for item in player.knowledge)
            ),
        ),
    )
    proposal, candidate, verification = _governance_records(
        command=command,
        intent=ActionIntent.INVESTIGATE,
        location_id=location_id,
        target_ids=(resource_id,) if resource_id else (),
        summary="Player validates the discovered Mistgate gate lens.",
        expected="Persist the lens reading as verified player knowledge.",
        verifier="product_resource_validation_governance_v1",
        checks=checks,
    )
    if not all(check.passed for check in checks) or not all(
        (option, resource, player, content_package)
    ):
        return _rejected(
            proposal,
            candidate,
            verification,
            _recovery_text(
                content_package,
                "recovery_missing_lens",
                command.locale,
                "Investigate the Old Aqueduct before validating its gate lens.",
            ),
        )
    event_id = f"committed_{command.id}"
    knowledge = _knowledge(
        option.knowledge_id,
        content_package.text(option.knowledge_statement_key, command.locale),
        resource.id,
        (resource.id,),
        event_id,
    )
    committed = _committed_event(
        command=command,
        candidate=candidate,
        verification=verification,
        summary=f"{command.actor_id} validated {resource.name}.",
        tags=("gate_lens_validated",),
        patches=(
            _knowledge_patch(
                world_state, player.knowledge, knowledge, "The lens reading was verified."
            ),
        ),
    )
    return _applied(
        world_state,
        committed,
        proposal,
        candidate,
        verification,
        content_package.text(option.response_key, command.locale),
        (f"Repair evidence validated: {resource.name}",),
    )


def govern_final_repair(
    *,
    command: PlayerCommand,
    intent: ParsedPlayerIntent,
    world_state: WorldState,
    content_package: ProductContentPackage | None,
) -> GovernedWorldOutcome:
    location_id = world_state.player.current_location_id if world_state.player else None
    target_id = intent.target_ids[0] if len(intent.target_ids) == 1 else None
    option = _final_repair_option(content_package, location_id, target_id)
    target = _local_entity(world_state, target_id, location_id)
    player = world_state.player
    inventory_item = (
        next(
            (
                item
                for item in player.inventory
                if option and item.resource_id == option.consumed_resource_id
            ),
            None,
        )
        if player
        else None
    )
    knowledge_ids = {item.id for item in player.knowledge} if player else set()
    target_tags = set(target.tags) if target else set()
    checks = (
        _check("supported_product_action", intent.normalized_action == "stabilize_regulator"),
        _check("intent_actor_matches_command", intent.actor_id == command.actor_id),
        _check("content_final_repair_option_exists", option is not None),
        _check("player_location_matches_world", command.location_id == location_id),
        _check("repair_target_is_present", target is not None),
        _check(
            "calibration_key_is_held",
            bool(inventory_item and option and inventory_item.quantity >= option.quantity),
        ),
        _check(
            "repair_evidence_is_complete",
            bool(option and set(option.prerequisite_knowledge_ids).issubset(knowledge_ids)),
        ),
        _check(
            "intermediate_repair_is_complete",
            bool(option and set(option.required_target_tags).issubset(target_tags)),
        ),
        _check(
            "ending_is_new",
            bool(option and not set(option.result_target_tags).issubset(target_tags)),
        ),
    )
    proposal, candidate, verification = _governance_records(
        command=command,
        intent=ActionIntent.REPAIR,
        location_id=location_id,
        target_ids=(target_id,) if target_id else (),
        summary="Player attempts the definitive Mistgate regulator calibration.",
        expected="Consume the key and replace temporary containment with a verified ending state.",
        verifier="product_final_repair_governance_v1",
        checks=checks,
    )
    if not all(check.passed for check in checks) or not all(
        (option, target, inventory_item, player, content_package)
    ):
        return _rejected(
            proposal,
            candidate,
            verification,
            _final_repair_recovery_message(
                content_package,
                command.locale,
                checks,
                inventory_ids={item.resource_id for item in player.inventory} if player else set(),
                knowledge_ids=knowledge_ids,
            ),
        )

    event_id = f"committed_{command.id}"
    inventory_after = []
    for item in player.inventory:
        if item.id != inventory_item.id:
            inventory_after.append(item)
        elif item.quantity > option.quantity:
            inventory_after.append(
                item.model_copy(update={"quantity": item.quantity - option.quantity})
            )
    removed = set(option.removed_target_tags)
    target_tags_after = tuple(
        dict.fromkeys(
            (*(tag for tag in target.tags if tag not in removed), *option.result_target_tags)
        )
    )
    knowledge = _knowledge(
        option.knowledge_id,
        content_package.text(option.knowledge_statement_key, command.locale),
        target.id,
        (target.id,),
        event_id,
    )
    committed = _committed_event(
        command=command,
        candidate=candidate,
        verification=verification,
        summary=f"{command.actor_id} completed the calibration of {target.name}.",
        tags=option.committed_event_tags,
        patches=(
            StatePatch(
                operation=PatchOperation.UPDATE,
                target_type=PatchTargetType.WORLD,
                target_id=world_state.world_id,
                path="/player/inventory",
                before=_dump(player.inventory),
                after=_dump(inventory_after),
                reason="The calibration key was installed into the regulator core.",
            ),
            StatePatch(
                operation=PatchOperation.UPDATE,
                target_type=PatchTargetType.ENTITY,
                target_id=target.id,
                path=f"/entity/{target.id}/tags",
                before=list(target.tags),
                after=list(target_tags_after),
                reason="Definitive repair replaced temporary containment with stable state.",
            ),
            _knowledge_patch(
                world_state, player.knowledge, knowledge, "The final repair was verified."
            ),
        ),
    )
    return _applied(
        world_state,
        committed,
        proposal,
        candidate,
        verification,
        content_package.text(option.response_key, command.locale),
        (
            f"Key item installed: {inventory_item.resource_id}",
            f"Ending reached: {option.outcome_id}",
        ),
    )


def _final_repair_recovery_message(
    package: ProductContentPackage | None,
    locale: str,
    checks: tuple[VerificationCheck, ...],
    *,
    inventory_ids: set[str],
    knowledge_ids: set[str],
) -> str:
    failed = {check.name for check in checks if not check.passed}
    if "intermediate_repair_is_complete" in failed:
        recovery_id = (
            "recovery_repair_access"
            if "stabilizer_parts" in inventory_ids
            else "recovery_missing_parts"
        )
        return _recovery_text(
            package,
            recovery_id,
            locale,
            "Complete the first regulator repair before attempting final calibration.",
        )
    if "calibration_key_is_held" in failed or (
        "repair_evidence_is_complete" in failed
        and "knowledge_calibration_key_secured" not in knowledge_ids
    ):
        return _recovery_text(
            package,
            "recovery_missing_key",
            locale,
            "Secure the calibration key from Ivo's workshop before final calibration.",
        )
    if "repair_evidence_is_complete" in failed:
        return _recovery_text(
            package,
            "recovery_missing_lens",
            locale,
            "Validate the gate lens at the Old Aqueduct before final calibration.",
        )
    return "The Dawn Regulator cannot be fully stabilized with the current evidence."


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


def _governance_records(
    *, command, intent, location_id, target_ids, summary, expected, verifier, checks
):
    proposal = ActionProposal(
        id=f"proposal_{command.id}",
        proposer_agent_id=command.actor_id,
        intent=intent,
        rationale=summary,
        target_location_id=location_id,
        target_entity_ids=target_ids,
        expected_outcome=expected,
    )
    candidate = EventCandidate(
        id=f"candidate_{command.id}",
        source_action_proposal_id=proposal.id,
        actor_agent_id=command.actor_id,
        summary=summary,
        status=EventCandidateStatus.UNDER_REVIEW,
        involved_location_ids=(location_id,) if location_id else (),
        involved_entity_ids=target_ids,
    )
    decision = (
        VerificationDecision.COMMIT
        if all(item.passed for item in checks)
        else VerificationDecision.REJECT
    )
    verification = VerificationResult(
        id=f"verification_{command.id}",
        event_candidate_id=candidate.id,
        decision=decision,
        verifier=verifier,
        checks=checks,
        reasons=(
            "All content and world-state gates passed."
            if decision == VerificationDecision.COMMIT
            else "One or more progression gates failed.",
        ),
        risk_flags=() if decision == VerificationDecision.COMMIT else ("progression_rejected",),
    )
    return proposal, candidate, verification


def _committed_event(*, command, candidate, verification, summary, tags, patches):
    return CommittedEvent(
        id=f"committed_{command.id}",
        event_candidate_id=candidate.id,
        verification_result_id=verification.id,
        summary=summary,
        tags=tags,
        state_diff=StateDiff(
            id=f"diff_{command.id}",
            source_event_candidate_id=candidate.id,
            committed_event_id=f"committed_{command.id}",
            patches=patches,
        ),
    )


def _applied(world_state, committed, proposal, candidate, verification, message, consequences):
    resulting_world, report = ControlledStateDiffApplier().apply(
        world_state=world_state,
        committed_event=committed,
        verification_result=verification,
    )
    if not report.applied:
        raise RuntimeError("verified progression StateDiff failed atomic application")
    return GovernedWorldOutcome(
        proposal=proposal,
        candidate=candidate.model_copy(update={"status": EventCandidateStatus.VERIFIED}),
        verification=verification,
        committed_event=committed,
        resulting_world_state=resulting_world,
        apply_report=report,
        player_message=message,
        consequences=consequences,
    )


def _rejected(proposal, candidate, verification, message):
    return GovernedWorldOutcome(
        proposal=proposal,
        candidate=candidate,
        verification=verification,
        committed_event=None,
        resulting_world_state=None,
        apply_report=None,
        player_message=message,
        consequences=(),
    )


def _knowledge(identifier, statement, source_id, subject_ids, event_id):
    return PlayerKnowledgeRecord(
        id=identifier,
        kind=PlayerKnowledgeKind.CONFIRMED_FACT,
        statement=statement,
        source_entity_id=source_id,
        subject_ids=subject_ids,
        confidence="high",
        committed_event_id=event_id,
    )


def _knowledge_patch(world_state, records, knowledge, reason):
    before = _dump(records)
    return StatePatch(
        operation=PatchOperation.APPEND,
        target_type=PatchTargetType.WORLD,
        target_id=world_state.world_id,
        path="/player/knowledge",
        before=before,
        after=[*before, knowledge.model_dump(mode="json")],
        reason=reason,
    )


def _release_option(package, location_id, character_id) -> ProductResourceReleaseOption | None:
    if not package:
        return None
    return next(
        (
            item
            for item in package.blueprint.resource_release_options
            if item.location_id == location_id and item.character_id == character_id
        ),
        None,
    )


def _validation_option(package, location_id, resource_id) -> ProductResourceValidationOption | None:
    if not package:
        return None
    return next(
        (
            item
            for item in package.blueprint.resource_validation_options
            if item.location_id == location_id and item.resource_id == resource_id
        ),
        None,
    )


def _final_repair_option(package, location_id, target_id) -> ProductFinalRepairOption | None:
    if not package:
        return None
    return next(
        (
            item
            for item in package.blueprint.final_repair_options
            if item.location_id == location_id and item.target_entity_id == target_id
        ),
        None,
    )


def _local_entity(world_state, entity_id, location_id, required_tag=None):
    return next(
        (
            item
            for item in world_state.entities
            if item.id == entity_id
            and item.location_id == location_id
            and (required_tag is None or required_tag in item.tags)
        ),
        None,
    )


def _resource(world_state, resource_id):
    return next((item for item in world_state.resources if item.id == resource_id), None)


def _check(name: str, passed: bool) -> VerificationCheck:
    return VerificationCheck(name=name, passed=passed, message=name.replace("_", " "))


def _dump(records: Iterable) -> list[dict[str, object]]:
    return [record.model_dump(mode="json") for record in records]
