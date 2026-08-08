from __future__ import annotations

from aethelis.product.command_contracts import (
    CommandInputMode,
    ParsedPlayerIntent,
    PlayerCommand,
)
from aethelis.product.commitment_resolution import (
    govern_commitment_breach,
    govern_repair,
)
from aethelis.product.content_contracts import (
    ProductContentPackage,
    ProductDialogueOption,
    ProductRoute,
)
from aethelis.product.dialogue_expression import (
    DialogueExpressionService,
    resolve_dialogue_expression,
)
from aethelis.product.governance_contracts import GovernedWorldOutcome
from aethelis.product.living_world import govern_living_world
from aethelis.product.progression_resolution import (
    govern_final_repair,
    govern_resource_release,
    govern_resource_validation,
)
from aethelis.product.resource_exchange import govern_resource_exchange
from aethelis.product.social_dialogue import govern_social_dialogue
from aethelis.product.world_narrative import govern_world_narrative
from aethelis.product.world_response import govern_world_response
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
    PlayerDialogueTurn,
    PlayerKnowledgeKind,
    PlayerKnowledgeRecord,
    PlayerRelationshipState,
    WorldResource,
    WorldState,
)


class ProductWorldEngine:
    """Govern supported product actions without bypassing engine truth contracts."""

    def __init__(self, dialogue_expression: DialogueExpressionService | None = None) -> None:
        self._dialogue_expression = dialogue_expression

    def govern(
        self,
        *,
        command: PlayerCommand,
        intent: ParsedPlayerIntent,
        world_state: WorldState,
        content_package: ProductContentPackage | None = None,
    ) -> GovernedWorldOutcome:
        if intent.normalized_action == "move_to_location":
            return self._govern_move(
                command=command,
                intent=intent,
                world_state=world_state,
                content_package=content_package,
            )
        if intent.normalized_action == "ask_character":
            if command.input_mode == CommandInputMode.NATURAL_LANGUAGE_INTENT:
                return govern_social_dialogue(
                    command=command,
                    intent=intent,
                    world_state=world_state,
                    content_package=content_package,
                    dialogue_expression=self._dialogue_expression,
                )
            return self._govern_dialogue(
                command=command,
                intent=intent,
                world_state=world_state,
                content_package=content_package,
            )
        if intent.normalized_action == "ask_world":
            return govern_world_narrative(
                command=command,
                intent=intent,
                world_state=world_state,
                content_package=content_package,
            )
        if intent.normalized_action == "advance_world":
            return govern_living_world(
                command=command,
                intent=intent,
                world_state=world_state,
                content_package=content_package,
            )
        if intent.normalized_action == "negotiate_resource":
            return govern_resource_exchange(
                command=command,
                intent=intent,
                world_state=world_state,
                content_package=content_package,
                dialogue_expression=self._dialogue_expression,
            )
        if intent.normalized_action == "repair_regulator":
            return govern_repair(
                command=command,
                intent=intent,
                world_state=world_state,
                content_package=content_package,
            )
        if intent.normalized_action == "break_commitment":
            return govern_commitment_breach(
                command=command,
                intent=intent,
                world_state=world_state,
                content_package=content_package,
            )
        if intent.normalized_action == "request_calibration_key":
            return govern_resource_release(
                command=command,
                intent=intent,
                world_state=world_state,
                content_package=content_package,
            )
        if intent.normalized_action == "validate_gate_lens":
            return govern_resource_validation(
                command=command,
                intent=intent,
                world_state=world_state,
                content_package=content_package,
            )
        if intent.normalized_action == "stabilize_regulator":
            return govern_final_repair(
                command=command,
                intent=intent,
                world_state=world_state,
                content_package=content_package,
            )
        if intent.normalized_action == "wait_for_world_response":
            return govern_world_response(
                command=command,
                intent=intent,
                world_state=world_state,
                content_package=content_package,
            )
        return self._govern_investigation(
            command=command,
            intent=intent,
            world_state=world_state,
        )

    def _govern_dialogue(
        self,
        *,
        command: PlayerCommand,
        intent: ParsedPlayerIntent,
        world_state: WorldState,
        content_package: ProductContentPackage | None,
    ) -> GovernedWorldOutcome:
        current_location = world_state.player.current_location_id if world_state.player else None
        character_id = intent.target_ids[0] if len(intent.target_ids) == 1 else None
        option = _find_dialogue_option(content_package, current_location, character_id)
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
        known_ids = (
            {record.id for record in world_state.player.knowledge} if world_state.player else set()
        )
        knowledge_is_new = option is not None and option.knowledge_id not in known_ids
        source_canon_valid = _dialogue_canon_source_is_valid(option, world_state)
        checks = (
            VerificationCheck(
                name="supported_product_action",
                passed=intent.normalized_action == "ask_character",
                message="Action must be ask_character.",
            ),
            VerificationCheck(
                name="intent_actor_matches_command",
                passed=intent.actor_id == command.actor_id,
                message="Normalized intent cannot replace the authorized player actor.",
            ),
            VerificationCheck(
                name="content_dialogue_option_exists",
                passed=option is not None,
                message="Dialogue must use a versioned option for this character and location.",
            ),
            VerificationCheck(
                name="character_is_present",
                passed=character is not None,
                message="The selected character must be present at the player's location.",
            ),
            VerificationCheck(
                name="player_location_matches_world",
                passed=command.location_id == current_location and current_location is not None,
                message="Command location must match the player's current world location.",
            ),
            VerificationCheck(
                name="knowledge_boundary_is_valid",
                passed=source_canon_valid,
                message="Confirmed knowledge must cite public Canon; rumors must not claim Canon.",
            ),
            VerificationCheck(
                name="knowledge_is_new",
                passed=knowledge_is_new,
                message="This bounded dialogue option must not be replayed as a new event.",
            ),
        )
        decision = (
            VerificationDecision.COMMIT
            if (
                option is not None
                and character is not None
                and all(check.passed for check in checks)
            )
            else VerificationDecision.REJECT
        )
        proposal = ActionProposal(
            id=f"proposal_{command.id}",
            proposer_agent_id=command.actor_id,
            intent=ActionIntent.DIALOGUE,
            rationale=(
                "Player selected a content-defined dialogue option with explicit knowledge scope."
            ),
            target_location_id=current_location,
            target_entity_ids=(character_id,) if character_id else (),
            expected_outcome=(
                "Record the bounded utterance, disclosed knowledge, and relationship effect."
            ),
        )
        candidate = EventCandidate(
            id=f"candidate_{command.id}",
            source_action_proposal_id=proposal.id,
            actor_agent_id=command.actor_id,
            summary="Player asks a present Mistgate character about a bounded topic.",
            status=EventCandidateStatus.UNDER_REVIEW,
            involved_location_ids=(current_location,) if current_location else (),
            involved_entity_ids=(character_id,) if character_id else (),
        )
        verification = VerificationResult(
            id=f"verification_{command.id}",
            event_candidate_id=candidate.id,
            decision=decision,
            verifier="product_dialogue_governance_v1",
            checks=checks,
            reasons=(
                "The authored dialogue act preserves location, identity, and knowledge boundaries."
                if decision == VerificationDecision.COMMIT
                else "The dialogue act could not be committed under current world state.",
            ),
            risk_flags=() if decision == VerificationDecision.COMMIT else ("dialogue_rejected",),
        )
        if (
            decision != VerificationDecision.COMMIT
            or option is None
            or character is None
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
                player_message="This conversation is not available in the current world state.",
                consequences=(),
            )

        event_id = f"committed_{command.id}"
        knowledge = PlayerKnowledgeRecord(
            id=option.knowledge_id,
            kind=PlayerKnowledgeKind(option.knowledge_kind),
            statement=content_package.text(option.knowledge_statement_key, command.locale),
            source_entity_id=character.id,
            subject_ids=option.topic_ids,
            confidence=option.knowledge_confidence,
            committed_event_id=event_id,
        )
        relationships = list(world_state.player.relationships)
        relationship_index = next(
            (
                index
                for index, relationship in enumerate(relationships)
                if relationship.character_id == character.id
            ),
            None,
        )
        if relationship_index is None:
            relationships.append(
                PlayerRelationshipState(
                    character_id=character.id,
                    trust=option.relationship_delta,
                    interaction_count=1,
                    last_committed_event_id=event_id,
                )
            )
        else:
            current_relationship = relationships[relationship_index]
            relationships[relationship_index] = current_relationship.model_copy(
                update={
                    "trust": max(
                        -5,
                        min(5, current_relationship.trust + option.relationship_delta),
                    ),
                    "interaction_count": current_relationship.interaction_count + 1,
                    "last_committed_event_id": event_id,
                }
            )
        authored_response = content_package.text(option.response_key, command.locale)
        expression = resolve_dialogue_expression(
            self._dialogue_expression,
            policy=content_package.blueprint.dialogue_expression_policy,
            locale=command.locale,
            character_id=character.id,
            character_name=character.name,
            character_summary=character.summary,
            dialogue_option_id=option.id,
            authored_utterance=authored_response,
            allowed_knowledge={knowledge.id: knowledge.statement},
        )
        response = expression.utterance
        dialogue_turn = PlayerDialogueTurn(
            id=f"dialogue_{command.id}",
            interaction_id=command.dialogue_interaction_id or f"interaction_{command.id}",
            character_id=character.id,
            dialogue_option_id=option.id,
            utterance=response,
            knowledge_record_ids=(knowledge.id,),
            committed_event_id=event_id,
            expression_evidence=expression.evidence,
        )
        before_knowledge = _dump_records(world_state.player.knowledge)
        before_relationships = _dump_records(world_state.player.relationships)
        before_dialogue = _dump_records(world_state.player.dialogue_history)
        committed = CommittedEvent(
            id=event_id,
            event_candidate_id=candidate.id,
            verification_result_id=verification.id,
            summary=f"{command.actor_id} spoke with {character.name} about {option.id}.",
            state_diff=StateDiff(
                id=f"diff_{command.id}",
                source_event_candidate_id=candidate.id,
                committed_event_id=event_id,
                patches=(
                    StatePatch(
                        operation=PatchOperation.APPEND,
                        target_type=PatchTargetType.WORLD,
                        target_id=world_state.world_id,
                        path="/player/knowledge",
                        before=before_knowledge,
                        after=[*before_knowledge, knowledge.model_dump(mode="json")],
                        reason="Verified dialogue disclosed one bounded knowledge record.",
                    ),
                    StatePatch(
                        operation=PatchOperation.APPEND,
                        target_type=PatchTargetType.RELATIONSHIP,
                        target_id=character.id,
                        path="/player/relationships",
                        before=before_relationships,
                        after=_dump_records(relationships),
                        reason="Verified dialogue updated the event-linked player relationship.",
                    ),
                    StatePatch(
                        operation=PatchOperation.APPEND,
                        target_type=PatchTargetType.WORLD,
                        target_id=world_state.world_id,
                        path="/player/dialogue_history",
                        before=before_dialogue,
                        after=[*before_dialogue, dialogue_turn.model_dump(mode="json")],
                        reason="Verified dialogue was retained for save and fork continuity.",
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
            raise RuntimeError("verified dialogue StateDiff failed atomic application")
        return GovernedWorldOutcome(
            proposal=proposal,
            candidate=candidate.model_copy(update={"status": EventCandidateStatus.VERIFIED}),
            verification=verification,
            committed_event=committed,
            resulting_world_state=resulting_world,
            apply_report=report,
            player_message=response,
            consequences=(
                f"Knowledge recorded: {knowledge.statement}",
                f"Relationship changed: {character.name} {option.relationship_delta:+d}",
            ),
        )

    def _govern_investigation(
        self,
        *,
        command: PlayerCommand,
        intent: ParsedPlayerIntent,
        world_state: WorldState,
    ) -> GovernedWorldOutcome:
        proposal = ActionProposal(
            id=f"proposal_{command.id}",
            proposer_agent_id=command.actor_id,
            intent=ActionIntent.INVESTIGATE,
            rationale="Player requested a bounded local investigation.",
            target_location_id=command.location_id,
            expected_outcome="Reveal one locally discoverable resource if governance permits.",
        )
        candidate = EventCandidate(
            id=f"candidate_{command.id}",
            source_action_proposal_id=proposal.id,
            actor_agent_id=command.actor_id,
            summary="Player investigates the current area for a discoverable resource.",
            status=EventCandidateStatus.UNDER_REVIEW,
            involved_location_ids=(command.location_id,) if command.location_id else (),
        )
        resource, checks = self._verify_investigation(
            command=command,
            intent=intent,
            world_state=world_state,
        )
        decision = (
            VerificationDecision.COMMIT
            if resource is not None and all(check.passed for check in checks)
            else VerificationDecision.REJECT
        )
        verification = VerificationResult(
            id=f"verification_{command.id}",
            event_candidate_id=candidate.id,
            decision=decision,
            verifier="product_investigation_governance_v1",
            checks=checks,
            reasons=(
                "The local investigation satisfies the bounded product action contract."
                if decision == VerificationDecision.COMMIT
                else "The investigation could not be committed under current world state.",
            ),
            risk_flags=() if decision == VerificationDecision.COMMIT else ("action_rejected",),
        )
        if decision != VerificationDecision.COMMIT or resource is None:
            return GovernedWorldOutcome(
                proposal=proposal,
                candidate=candidate,
                verification=verification,
                committed_event=None,
                resulting_world_state=None,
                apply_report=None,
                player_message="Nothing new could be discovered here.",
                consequences=(),
            )

        current = list(resource.discovery_state.discovered_by_agent_ids)
        after = list(dict.fromkeys([*current, command.actor_id]))
        event_id = f"committed_{command.id}"
        committed = CommittedEvent(
            id=event_id,
            event_candidate_id=candidate.id,
            verification_result_id=verification.id,
            summary=f"{command.actor_id} discovered {resource.name}.",
            state_diff=StateDiff(
                id=f"diff_{command.id}",
                source_event_candidate_id=candidate.id,
                committed_event_id=event_id,
                patches=(
                    StatePatch(
                        operation=PatchOperation.APPEND,
                        target_type=PatchTargetType.RESOURCE,
                        target_id=resource.id,
                        path=(f"/resource/{resource.id}/discovery_state/discovered_by_agent_ids"),
                        before=current,
                        after=after,
                        reason="Verified local investigation revealed the resource.",
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
            raise RuntimeError("verified investigation StateDiff failed atomic application")
        return GovernedWorldOutcome(
            proposal=proposal,
            candidate=candidate.model_copy(update={"status": EventCandidateStatus.VERIFIED}),
            verification=verification,
            committed_event=committed,
            resulting_world_state=resulting_world,
            apply_report=report,
            player_message=f"You discovered {resource.name}.",
            consequences=(f"Discovered resource: {resource.name}",),
        )

    def _govern_move(
        self,
        *,
        command: PlayerCommand,
        intent: ParsedPlayerIntent,
        world_state: WorldState,
        content_package: ProductContentPackage | None,
    ) -> GovernedWorldOutcome:
        current_location = world_state.player.current_location_id if world_state.player else None
        destination_id = intent.target_ids[0] if len(intent.target_ids) == 1 else None
        route = _find_route(content_package, current_location, destination_id)
        destination = next(
            (location for location in world_state.locations if location.id == destination_id),
            None,
        )
        checks = (
            VerificationCheck(
                name="supported_product_action",
                passed=intent.normalized_action == "move_to_location",
                message="Action must be move_to_location.",
            ),
            VerificationCheck(
                name="intent_actor_matches_command",
                passed=intent.actor_id == command.actor_id,
                message="Normalized intent cannot replace the authorized player actor.",
            ),
            VerificationCheck(
                name="content_package_available",
                passed=content_package is not None,
                message="Movement requires a versioned product content package.",
            ),
            VerificationCheck(
                name="player_location_matches_world",
                passed=command.location_id == current_location and current_location is not None,
                message="Command source location must match the player world location.",
            ),
            VerificationCheck(
                name="destination_exists",
                passed=destination is not None,
                message="Movement destination must exist in the current world.",
            ),
            VerificationCheck(
                name="route_is_known",
                passed=route is not None and route.initially_known,
                message="Movement must follow a player-known content route.",
            ),
            VerificationCheck(
                name="route_access_is_open",
                passed=route is not None and not route.required_access_tags,
                message="The P2 entry movement route must not require unresolved access tags.",
            ),
        )
        if route is not None and route.required_access_tags:
            decision = VerificationDecision.PENDING_GATE
        elif all(check.passed for check in checks):
            decision = VerificationDecision.COMMIT
        else:
            decision = VerificationDecision.REJECT
        proposal = ActionProposal(
            id=f"proposal_{command.id}",
            proposer_agent_id=command.actor_id,
            intent=ActionIntent.MOVE,
            rationale="Player requested movement along a product-defined route.",
            target_location_id=destination_id,
            expected_outcome="Move to the selected connected location if governance permits.",
        )
        candidate = EventCandidate(
            id=f"candidate_{command.id}",
            source_action_proposal_id=proposal.id,
            actor_agent_id=command.actor_id,
            summary="Player attempts to move between two connected Mistgate locations.",
            status=EventCandidateStatus.UNDER_REVIEW,
            involved_location_ids=tuple(
                location_id
                for location_id in (current_location, destination_id)
                if location_id is not None
            ),
        )
        verification = VerificationResult(
            id=f"verification_{command.id}",
            event_candidate_id=candidate.id,
            decision=decision,
            verifier="product_route_governance_v1",
            checks=checks,
            reasons=(
                "The selected route is known, connected, and currently open."
                if decision == VerificationDecision.COMMIT
                else "Movement cannot commit under the current route and access state.",
            ),
            risk_flags=() if decision == VerificationDecision.COMMIT else ("movement_blocked",),
        )
        if decision != VerificationDecision.COMMIT or destination is None:
            message = (
                "This route requires access that has not been granted."
                if decision == VerificationDecision.PENDING_GATE
                else "You cannot move to that location from here."
            )
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

        event_id = f"committed_{command.id}"
        committed = CommittedEvent(
            id=event_id,
            event_candidate_id=candidate.id,
            verification_result_id=verification.id,
            summary=f"{command.actor_id} moved to {destination.name}.",
            state_diff=StateDiff(
                id=f"diff_{command.id}",
                source_event_candidate_id=candidate.id,
                committed_event_id=event_id,
                patches=(
                    StatePatch(
                        operation=PatchOperation.UPDATE,
                        target_type=PatchTargetType.WORLD,
                        target_id=world_state.world_id,
                        path="/player/current_location_id",
                        before=current_location,
                        after=destination.id,
                        reason="Verified route movement changed the player location.",
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
            raise RuntimeError("verified movement StateDiff failed atomic application")
        return GovernedWorldOutcome(
            proposal=proposal,
            candidate=candidate.model_copy(update={"status": EventCandidateStatus.VERIFIED}),
            verification=verification,
            committed_event=committed,
            resulting_world_state=resulting_world,
            apply_report=report,
            player_message=f"You arrived at {destination.name}.",
            consequences=(f"Moved to: {destination.name}",),
        )

    @staticmethod
    def _verify_investigation(
        *,
        command: PlayerCommand,
        intent: ParsedPlayerIntent,
        world_state: WorldState,
    ) -> tuple[WorldResource | None, tuple[VerificationCheck, ...]]:
        action_valid = intent.normalized_action in {"investigate_area", "inspect_resource"}
        current_location = world_state.player.current_location_id if world_state.player else None
        location_valid = command.location_id is not None and command.location_id == current_location
        candidates = tuple(
            resource
            for resource in world_state.resources
            if resource.location_id == current_location
            and command.actor_id not in resource.discovery_state.discovered_by_agent_ids
            and (intent.normalized_action == "investigate_area" or resource.id in intent.target_ids)
        )
        resource = sorted(candidates, key=lambda item: item.id)[0] if candidates else None
        checks = (
            VerificationCheck(
                name="supported_product_action",
                passed=action_valid,
                message="Action must be investigate_area or inspect_resource.",
            ),
            VerificationCheck(
                name="intent_actor_matches_command",
                passed=intent.actor_id == command.actor_id,
                message="Normalized intent cannot replace the authorized player actor.",
            ),
            VerificationCheck(
                name="player_location_matches_world",
                passed=location_valid,
                message="Command location must match the player's current world location.",
            ),
            VerificationCheck(
                name="undiscovered_local_resource_exists",
                passed=resource is not None,
                message="A requested local resource must remain undiscovered by this player actor.",
            ),
        )
        return resource, checks


def _find_route(
    package: ProductContentPackage | None,
    current_location_id: str | None,
    destination_id: str | None,
) -> ProductRoute | None:
    if package is None or current_location_id is None or destination_id is None:
        return None
    for route in package.blueprint.routes:
        if route.from_location_id == current_location_id and route.to_location_id == destination_id:
            return route
        if (
            route.bidirectional
            and route.to_location_id == current_location_id
            and route.from_location_id == destination_id
        ):
            return route
    return None


def _find_dialogue_option(
    package: ProductContentPackage | None,
    location_id: str | None,
    character_id: str | None,
) -> ProductDialogueOption | None:
    if package is None or location_id is None or character_id is None:
        return None
    matches = tuple(
        option
        for option in package.blueprint.dialogue_options
        if option.action_id == "ask_character"
        and option.location_id == location_id
        and option.character_id == character_id
    )
    return matches[0] if len(matches) == 1 else None


def _dialogue_canon_source_is_valid(
    option: ProductDialogueOption | None,
    world_state: WorldState,
) -> bool:
    if option is None:
        return False
    if option.knowledge_kind == "rumor":
        return option.source_canon_fact_id is None
    return any(
        fact.id == option.source_canon_fact_id and fact.visibility == CanonVisibility.PUBLIC
        for fact in world_state.canon_facts
    )


def _dump_records(records) -> list[dict[str, object]]:
    return [record.model_dump(mode="json") for record in records]
