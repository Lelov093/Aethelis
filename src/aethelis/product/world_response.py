from __future__ import annotations

from aethelis.product.command_contracts import ParsedPlayerIntent, PlayerCommand
from aethelis.product.content_contracts import ProductContentPackage, ProductWorldResponseOption
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
from aethelis.schemas.world import PlayerWorldResponse, WorldState


def govern_world_response(
    *,
    command: PlayerCommand,
    intent: ParsedPlayerIntent,
    world_state: WorldState,
    content_package: ProductContentPackage | None,
) -> GovernedWorldOutcome:
    player = world_state.player
    location_id = player.current_location_id if player else None
    option = _eligible_option(content_package, world_state)
    actor = next(
        (
            entity
            for entity in world_state.entities
            if option is not None and entity.id == option.actor_entity_id
        ),
        None,
    )
    commitment = (
        next(
            (
                item
                for item in player.commitments
                if option is not None and item.id == option.commitment_id
            ),
            None,
        )
        if player
        else None
    )
    relationship = (
        next(
            (
                item
                for item in player.relationships
                if option is not None and item.character_id == option.actor_entity_id
            ),
            None,
        )
        if player
        else None
    )
    existing_option_ids = (
        {item.response_option_id for item in player.world_responses} if player else set()
    )
    outcome = (
        next(
            (
                item
                for item in content_package.blueprint.outcomes
                if option is not None and item.id == option.outcome_id
            ),
            None,
        )
        if content_package
        else None
    )
    persisted_tags = {tag for entity in world_state.entities for tag in entity.tags}
    checks = (
        _check("supported_wait_action", intent.normalized_action == "wait_for_world_response"),
        _check("intent_actor_matches_command", intent.actor_id == command.actor_id),
        _check("wait_has_no_invented_target", not intent.target_ids),
        _check("player_location_matches_world", command.location_id == location_id),
        _check("content_response_option_exists", option is not None),
        _check("response_actor_exists", actor is not None),
        _check(
            "branch_commitment_matches",
            bool(commitment and option and commitment.status.value == option.commitment_status),
        ),
        _check("response_relationship_exists", relationship is not None),
        _check(
            "required_outcome_is_persisted",
            bool(outcome and set(outcome.required_committed_event_tags).issubset(persisted_tags)),
        ),
        _check(
            "response_is_new",
            bool(option and option.id not in existing_option_ids),
        ),
    )
    decision = (
        VerificationDecision.COMMIT
        if all(check.passed for check in checks)
        else VerificationDecision.REJECT
    )
    proposal = ActionProposal(
        id=f"proposal_{command.id}",
        proposer_agent_id=actor.id if actor else command.actor_id,
        intent=ActionIntent.OBSERVE,
        rationale=(
            "A bounded character response is selected from committed ending and social state."
        ),
        target_location_id=location_id,
        target_entity_ids=(actor.id,) if actor else (),
        expected_outcome="Persist one branch-specific civic response after the player yields time.",
    )
    candidate = EventCandidate(
        id=f"candidate_{command.id}",
        source_action_proposal_id=proposal.id,
        actor_agent_id=actor.id if actor else command.actor_id,
        summary="A Mistgate character responds to the completed repair and social history.",
        status=EventCandidateStatus.UNDER_REVIEW,
        involved_location_ids=(location_id,) if location_id else (),
        involved_entity_ids=(actor.id,) if actor else (),
    )
    verification = VerificationResult(
        id=f"verification_{command.id}",
        event_candidate_id=candidate.id,
        decision=decision,
        verifier="product_bounded_world_response_governance_v1",
        checks=checks,
        reasons=(
            "Committed outcome and social state select exactly one authored character response."
            if decision == VerificationDecision.COMMIT
            else "No bounded character response can commit under the current state.",
        ),
        risk_flags=() if decision == VerificationDecision.COMMIT else ("response_rejected",),
    )
    if (
        decision != VerificationDecision.COMMIT
        or option is None
        or actor is None
        or commitment is None
        or relationship is None
        or player is None
        or content_package is None
    ):
        return GovernedWorldOutcome(
            proposal=proposal,
            candidate=candidate,
            verification=verification,
            committed_event=None,
            resulting_world_state=None,
            apply_report=None,
            player_message="No further city response is ready for this timeline.",
            consequences=(),
        )

    event_id = f"committed_{command.id}"
    actor_tags_after = tuple(dict.fromkeys((*actor.tags, *option.result_actor_tags)))
    relationships_after = list(player.relationships)
    relationship_index = next(
        index
        for index, item in enumerate(relationships_after)
        if item.character_id == relationship.character_id
    )
    relationships_after[relationship_index] = relationship.model_copy(
        update={
            "trust": max(-5, min(5, relationship.trust + option.relationship_delta)),
            "interaction_count": relationship.interaction_count + 1,
            "last_committed_event_id": event_id,
        }
    )
    response = PlayerWorldResponse(
        id=f"world_response_{command.id}",
        response_option_id=option.id,
        actor_entity_id=actor.id,
        response_kind=option.response_kind,
        summary=content_package.text(option.response_key, command.locale),
        committed_event_id=event_id,
    )
    before_relationships = _dump(player.relationships)
    before_responses = _dump(player.world_responses)
    committed = CommittedEvent(
        id=event_id,
        event_candidate_id=candidate.id,
        verification_result_id=verification.id,
        summary=f"{actor.name} issued a branch-specific response to the repaired city.",
        tags=option.committed_event_tags,
        state_diff=StateDiff(
            id=f"diff_{command.id}",
            source_event_candidate_id=candidate.id,
            committed_event_id=event_id,
            patches=(
                StatePatch(
                    operation=PatchOperation.UPDATE,
                    target_type=PatchTargetType.ENTITY,
                    target_id=actor.id,
                    path=f"/entity/{actor.id}/tags",
                    before=list(actor.tags),
                    after=list(actor_tags_after),
                    reason="The responding character's civic stance became durable world state.",
                ),
                StatePatch(
                    operation=PatchOperation.APPEND,
                    target_type=PatchTargetType.RELATIONSHIP,
                    target_id=actor.id,
                    path="/player/relationships",
                    before=before_relationships,
                    after=_dump(relationships_after),
                    reason="The autonomous response applied its bounded relationship consequence.",
                ),
                StatePatch(
                    operation=PatchOperation.APPEND,
                    target_type=PatchTargetType.WORLD,
                    target_id=world_state.world_id,
                    path="/player/world_responses",
                    before=before_responses,
                    after=[*before_responses, response.model_dump(mode="json")],
                    reason="The player-visible city response remains available after reload.",
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
        raise RuntimeError("verified world-response StateDiff failed atomic application")
    return GovernedWorldOutcome(
        proposal=proposal,
        candidate=candidate.model_copy(update={"status": EventCandidateStatus.VERIFIED}),
        verification=verification,
        committed_event=committed,
        resulting_world_state=resulting_world,
        apply_report=report,
        player_message=response.summary,
        consequences=(
            f"World response: {option.response_kind}",
            f"Relationship changed: {actor.name} {option.relationship_delta:+d}",
        ),
    )


def _eligible_option(
    package: ProductContentPackage | None,
    world_state: WorldState,
) -> ProductWorldResponseOption | None:
    if package is None or world_state.player is None:
        return None
    commitment_by_id = {item.id: item for item in world_state.player.commitments}
    existing = {item.response_option_id for item in world_state.player.world_responses}
    matches = tuple(
        option
        for option in package.blueprint.world_response_options
        if option.id not in existing
        and option.commitment_id in commitment_by_id
        and commitment_by_id[option.commitment_id].status.value == option.commitment_status
    )
    return matches[0] if len(matches) == 1 else None


def _check(name: str, passed: bool) -> VerificationCheck:
    return VerificationCheck(name=name, passed=passed, message=name.replace("_", " "))


def _dump(records) -> list[dict[str, object]]:
    return [record.model_dump(mode="json") for record in records]
