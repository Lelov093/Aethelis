from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from aethelis.product.command_contracts import (
    CommandInputMode,
    ParsedPlayerIntent,
    PlayerCommand,
)
from aethelis.product.content_loader import ProductContentPackageLoader
from aethelis.product.world_engine import ProductWorldEngine
from aethelis.schemas.events import VerificationDecision
from aethelis.schemas.world import (
    DialogueActKind,
    Location,
    PlayerContext,
    ResourceKind,
    WorldResource,
    WorldState,
)

NOW = datetime(2026, 8, 1, 13, 0, tzinfo=UTC)


ROOT = Path(__file__).resolve().parents[2]


def command(
    *,
    location_id: str = "plaza",
    action_id: str = "investigate_area",
    target_ids: tuple[str, ...] = (),
) -> PlayerCommand:
    return PlayerCommand(
        id="command_engine",
        idempotency_key="engine-request-0001",
        principal_id="principal_1",
        player_profile_id="profile_1",
        world_instance_id="instance_1",
        play_session_id="session_1",
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id=action_id,
        actor_id="profile_1",
        target_ids=target_ids,
        location_id=location_id,
        expected_world_version=0,
        locale="en",
        submitted_at=NOW,
        updated_at=NOW,
    )


def world() -> WorldState:
    return WorldState(
        schema_version="1.0",
        world_id="world_1",
        name="World",
        summary="A governed world.",
        locations=(Location(id="plaza", name="Plaza", summary="A plaza."),),
        factions=(),
        entities=(),
        resources=(
            WorldResource(
                id="token",
                name="Token",
                kind=ResourceKind.KEY_ITEM,
                quantity=1,
                location_id="plaza",
                summary="A hidden token.",
            ),
        ),
        canon_facts=(),
        player=PlayerContext(summary="Player", current_location_id="plaza"),
    )


def intent() -> ParsedPlayerIntent:
    return ParsedPlayerIntent(
        normalized_action="investigate_area",
        actor_id="profile_1",
        confidence=1,
        safety_classification="requires_governance",
    )


def test_investigation_uses_verified_committed_event_and_applier() -> None:
    original = world()
    outcome = ProductWorldEngine().govern(command=command(), intent=intent(), world_state=original)

    assert outcome.verification.decision == VerificationDecision.COMMIT
    assert outcome.committed_event is not None
    assert outcome.apply_report is not None and outcome.apply_report.applied
    assert outcome.resulting_world_state is not original
    assert original.resources[0].discovery_state.discovered_by_agent_ids == ()
    assert outcome.resulting_world_state.resources[0].discovery_state.discovered_by_agent_ids == (
        "profile_1",
    )


def test_wrong_location_is_rejected_without_world_mutation() -> None:
    outcome = ProductWorldEngine().govern(
        command=command(location_id="elsewhere"), intent=intent(), world_state=world()
    )

    assert outcome.verification.decision == VerificationDecision.REJECT
    assert outcome.committed_event is None
    assert outcome.resulting_world_state is None
    assert outcome.apply_report is None


def test_normalized_intent_cannot_replace_authorized_player_actor() -> None:
    outcome = ProductWorldEngine().govern(
        command=command(),
        intent=intent().model_copy(update={"actor_id": "intruder"}),
        world_state=world(),
    )

    assert outcome.verification.decision == VerificationDecision.REJECT
    assert outcome.committed_event is None


def test_inspect_resource_only_discovers_the_requested_local_target() -> None:
    original = world()
    inspect_command = command(action_id="inspect_resource", target_ids=("token",))
    inspect_intent = ParsedPlayerIntent(
        normalized_action="inspect_resource",
        actor_id="profile_1",
        target_ids=("token",),
        confidence=1,
        safety_classification="requires_governance",
    )

    outcome = ProductWorldEngine().govern(
        command=inspect_command,
        intent=inspect_intent,
        world_state=original,
    )

    assert outcome.verification.decision == VerificationDecision.COMMIT
    assert outcome.resulting_world_state.resources[0].discovery_state.discovered_by_agent_ids == (
        "profile_1",
    )


def test_movement_uses_product_route_and_controlled_world_patch() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={"player": package.initial_world_state.player.model_copy(update={"id": "profile_1"})}
    )
    movement = command(
        location_id="council_square",
        action_id="move_to_location",
        target_ids=("central_archive",),
    )
    parsed = ParsedPlayerIntent(
        normalized_action="move_to_location",
        actor_id="profile_1",
        target_ids=("central_archive",),
        confidence=1,
        safety_classification="requires_governance",
    )

    outcome = ProductWorldEngine().govern(
        command=movement,
        intent=parsed,
        world_state=original,
        content_package=package,
    )

    assert outcome.verification.decision == VerificationDecision.COMMIT
    assert outcome.apply_report is not None and outcome.apply_report.applied
    assert original.player.current_location_id == "council_square"
    assert outcome.resulting_world_state.player.current_location_id == "central_archive"


def test_movement_rejects_non_adjacent_destination() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={
            "player": package.initial_world_state.player.model_copy(
                update={"id": "profile_1", "current_location_id": "central_archive"}
            )
        }
    )
    movement = command(
        location_id="central_archive",
        action_id="move_to_location",
        target_ids=("market_row",),
    )
    parsed = ParsedPlayerIntent(
        normalized_action="move_to_location",
        actor_id="profile_1",
        target_ids=("market_row",),
        confidence=1,
        safety_classification="requires_governance",
    )

    outcome = ProductWorldEngine().govern(
        command=movement,
        intent=parsed,
        world_state=original,
        content_package=package,
    )

    assert outcome.verification.decision == VerificationDecision.REJECT
    assert outcome.committed_event is None
    assert outcome.resulting_world_state is None


def test_dialogue_commits_bounded_knowledge_relationship_and_history() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={
            "player": package.initial_world_state.player.model_copy(
                update={"id": "profile_1", "current_location_id": "market_row"}
            )
        }
    )
    dialogue = command(
        location_id="market_row",
        action_id="ask_character",
        target_ids=("selka",),
    )
    parsed = ParsedPlayerIntent(
        normalized_action="ask_character",
        actor_id="profile_1",
        target_ids=("selka",),
        confidence=1,
        safety_classification="requires_governance",
    )

    outcome = ProductWorldEngine().govern(
        command=dialogue,
        intent=parsed,
        world_state=original,
        content_package=package,
    )

    assert outcome.verification.decision == VerificationDecision.COMMIT
    assert outcome.apply_report is not None and outcome.apply_report.applied_patch_count == 3
    assert original.player.knowledge == ()
    player = outcome.resulting_world_state.player
    assert player.knowledge[0].kind == "confirmed_fact"
    assert player.knowledge[0].source_entity_id == "selka"
    assert player.relationships[0].trust == 1
    assert player.relationships[0].interaction_count == 1
    assert player.dialogue_history[0].knowledge_record_ids == (
        "knowledge_stabilizer_parts_limited",
    )
    assert "零件确实紧张" in outcome.player_message


def test_dialogue_keeps_nara_claim_as_low_confidence_rumor() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={
            "player": package.initial_world_state.player.model_copy(
                update={"id": "profile_1", "current_location_id": "market_row"}
            )
        }
    )
    dialogue = command(
        location_id="market_row",
        action_id="ask_character",
        target_ids=("nara",),
    )
    parsed = ParsedPlayerIntent(
        normalized_action="ask_character",
        actor_id="profile_1",
        target_ids=("nara",),
        confidence=1,
        safety_classification="requires_governance",
    )

    outcome = ProductWorldEngine().govern(
        command=dialogue,
        intent=parsed,
        world_state=original,
        content_package=package,
    )

    knowledge = outcome.resulting_world_state.player.knowledge[0]
    assert knowledge.kind == "rumor"
    assert knowledge.confidence == "low"
    assert "尚未验证" in knowledge.statement
    assert "钥匙在工坊保险柜" not in outcome.resulting_world_state.model_dump_json()


def test_dialogue_rejects_replay_without_mutation() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={
            "player": package.initial_world_state.player.model_copy(
                update={"id": "profile_1", "current_location_id": "market_row"}
            )
        }
    )
    dialogue = command(
        location_id="market_row",
        action_id="ask_character",
        target_ids=("selka",),
    )
    parsed = ParsedPlayerIntent(
        normalized_action="ask_character",
        actor_id="profile_1",
        target_ids=("selka",),
        confidence=1,
        safety_classification="requires_governance",
    )
    first = ProductWorldEngine().govern(
        command=dialogue,
        intent=parsed,
        world_state=original,
        content_package=package,
    )

    replay = ProductWorldEngine().govern(
        command=dialogue.model_copy(update={"id": "command_replay"}),
        intent=parsed,
        world_state=first.resulting_world_state,
        content_package=package,
    )

    assert replay.verification.decision == VerificationDecision.REJECT
    assert replay.committed_event is None
    assert replay.resulting_world_state is None


def test_free_greeting_commits_a_character_response_without_authored_topic() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={"player": package.initial_world_state.player.model_copy(update={"id": "profile_1"})}
    )
    free_command = command(
        location_id="council_square",
        action_id="ask_character",
        target_ids=("rowan",),
    ).model_copy(
        update={
            "input_mode": CommandInputMode.NATURAL_LANGUAGE_INTENT,
            "action_id": None,
            "text": "你好",
        }
    )
    parsed = ParsedPlayerIntent(
        normalized_action="ask_character",
        actor_id="profile_1",
        target_ids=("rowan",),
        confidence=1,
        safety_classification="requires_governance",
        dialogue_act=DialogueActKind.GREETING,
    )

    outcome = ProductWorldEngine().govern(
        command=free_command,
        intent=parsed,
        world_state=original,
        content_package=package,
    )

    assert outcome.verification.decision == VerificationDecision.COMMIT
    turn = outcome.resulting_world_state.player.dialogue_history[-1]
    assert turn.dialogue_act == DialogueActKind.GREETING
    assert turn.dialogue_option_id is None
    assert turn.player_utterance == "你好"


def test_free_dialogue_reuses_bounded_current_interaction_context() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={"player": package.initial_world_state.player.model_copy(update={"id": "profile_1"})}
    )
    interaction_id = "dialogue_context_1"
    first_command = command(
        location_id="council_square",
        action_id="ask_character",
        target_ids=("rowan",),
    ).model_copy(
        update={
            "id": "command_context_1",
            "input_mode": CommandInputMode.NATURAL_LANGUAGE_INTENT,
            "action_id": None,
            "text": "你好",
            "dialogue_interaction_id": interaction_id,
        }
    )
    first = ProductWorldEngine().govern(
        command=first_command,
        intent=ParsedPlayerIntent(
            normalized_action="ask_character",
            actor_id="profile_1",
            target_ids=("rowan",),
            confidence=1,
            safety_classification="requires_governance",
            dialogue_act=DialogueActKind.GREETING,
        ),
        world_state=original,
        content_package=package,
    )
    second = ProductWorldEngine().govern(
        command=first_command.model_copy(
            update={"id": "command_context_2", "text": "你刚才那句话是什么意思？"}
        ),
        intent=ParsedPlayerIntent(
            normalized_action="ask_character",
            actor_id="profile_1",
            target_ids=("rowan",),
            confidence=1,
            safety_classification="requires_governance",
            dialogue_act=DialogueActKind.QUESTION,
        ),
        world_state=first.resulting_world_state,
        content_package=package,
    )
    third = ProductWorldEngine().govern(
        command=first_command.model_copy(
            update={"id": "command_context_3", "text": "那为什么要这样提醒我？"}
        ),
        intent=ParsedPlayerIntent(
            normalized_action="ask_character",
            actor_id="profile_1",
            target_ids=("rowan",),
            confidence=1,
            safety_classification="requires_governance",
            dialogue_act=DialogueActKind.QUESTION,
        ),
        world_state=second.resulting_world_state,
        content_package=package,
    )

    turns = third.resulting_world_state.player.dialogue_history[-3:]
    assert {turn.interaction_id for turn in turns} == {interaction_id}
    assert len(turns) == 3
    assert turns[1].player_utterance in turns[-1].utterance


def test_player_claim_grows_listener_cognition_then_propagates_on_world_turn() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={"player": package.initial_world_state.player.model_copy(update={"id": "profile_1"})}
    )
    claim_command = command(
        location_id="council_square",
        action_id="ask_character",
        target_ids=("rowan",),
    ).model_copy(
        update={
            "id": "command_claim",
            "input_mode": CommandInputMode.NATURAL_LANGUAGE_INTENT,
            "action_id": None,
            "text": "我听说校准钥匙在市场。",
        }
    )
    claim = ProductWorldEngine().govern(
        command=claim_command,
        intent=ParsedPlayerIntent(
            normalized_action="ask_character",
            actor_id="profile_1",
            target_ids=("rowan",),
            confidence=1,
            safety_classification="requires_governance",
            dialogue_act=DialogueActKind.CLAIM,
            claim_text="我听说校准钥匙在市场。",
        ),
        world_state=original,
        content_package=package,
    )

    assert len(claim.resulting_world_state.agent_claims) == 1
    listener_belief = claim.resulting_world_state.agent_beliefs[-1]
    assert listener_belief.owner_agent_id == "rowan"
    assert listener_belief.truth_status == "unknown"
    advance = ProductWorldEngine().govern(
        command=command(action_id="advance_world").model_copy(update={"id": "command_advance"}),
        intent=ParsedPlayerIntent(
            normalized_action="advance_world",
            actor_id="profile_1",
            confidence=1,
            safety_classification="requires_governance",
        ),
        world_state=claim.resulting_world_state,
        content_package=package,
    )

    assert advance.verification.decision == VerificationDecision.COMMIT
    assert advance.resulting_world_state.clock.turn == 1
    assert advance.resulting_world_state.clock.elapsed_minutes == 15
    assert (
        advance.resulting_world_state.world_activities[-1].activity_kind
        == "knowledge_propagation"
    )
    assert advance.resulting_world_state.agent_beliefs[-1].owner_agent_id != "rowan"


def test_world_narrative_answers_from_visible_state_without_hidden_canon() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={
            "player": package.initial_world_state.player.model_copy(update={"id": "profile_1"}),
            "agent_profiles": (),
        }
    )
    narrative_command = command(location_id="council_square").model_copy(
        update={
            "id": "command_narrative",
            "input_mode": CommandInputMode.NATURAL_LANGUAGE_INTENT,
            "action_id": None,
            "text": "这里现在发生了什么？",
            "target_ids": ("world_narrative",),
        }
    )
    outcome = ProductWorldEngine().govern(
        command=narrative_command,
        intent=ParsedPlayerIntent(
            normalized_action="ask_world",
            actor_id="profile_1",
            confidence=1,
            safety_classification="requires_governance",
            dialogue_act=DialogueActKind.WORLD_OBSERVATION,
        ),
        world_state=original,
        content_package=package,
    )

    assert outcome.verification.decision == VerificationDecision.COMMIT
    assert "workshop_safe" not in outcome.player_message
    assert (
        outcome.resulting_world_state.player.dialogue_history[-1].target_kind
        == "world_narrative"
    )

    attempted = ProductWorldEngine().govern(
        command=narrative_command.model_copy(
            update={"id": "command_narrative_action", "text": "我尝试直接打开那扇门。"}
        ),
        intent=ParsedPlayerIntent(
            normalized_action="ask_world",
            actor_id="profile_1",
            confidence=1,
            safety_classification="requires_governance",
            dialogue_act=DialogueActKind.WORLD_ACTION,
        ),
        world_state=outcome.resulting_world_state,
        content_package=package,
    )
    turn = attempted.resulting_world_state.player.dialogue_history[-1]
    assert turn.requested_effect_status == "needs_clarification"
    assert attempted.resulting_world_state.resources == original.resources


def test_resource_exchange_commits_stock_inventory_commitment_and_social_history() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={
            "player": package.initial_world_state.player.model_copy(
                update={"id": "profile_1", "current_location_id": "market_row"}
            )
        }
    )
    dialogue = ProductWorldEngine().govern(
        command=command(
            location_id="market_row",
            action_id="ask_character",
            target_ids=("selka",),
        ),
        intent=ParsedPlayerIntent(
            normalized_action="ask_character",
            actor_id="profile_1",
            target_ids=("selka",),
            confidence=1,
            safety_classification="requires_governance",
        ),
        world_state=original,
        content_package=package,
    )
    exchange_command = command(
        location_id="market_row",
        action_id="negotiate_resource",
        target_ids=("selka",),
    ).model_copy(update={"id": "command_exchange"})
    exchange_intent = ParsedPlayerIntent(
        normalized_action="negotiate_resource",
        actor_id="profile_1",
        target_ids=("selka",),
        confidence=1,
        safety_classification="requires_governance",
    )

    outcome = ProductWorldEngine().govern(
        command=exchange_command,
        intent=exchange_intent,
        world_state=dialogue.resulting_world_state,
        content_package=package,
    )

    assert outcome.verification.decision == VerificationDecision.COMMIT
    assert outcome.apply_report is not None and outcome.apply_report.applied_patch_count == 5
    assert original.player.inventory == ()
    player = outcome.resulting_world_state.player
    assert player.inventory[0].resource_id == "stabilizer_parts"
    assert player.inventory[0].quantity == 1
    assert player.commitments[0].id == "commitment_provide_repair_evidence"
    assert player.commitments[0].status == "active"
    assert player.relationships[0].trust == 1
    assert player.relationships[0].interaction_count == 2
    assert len(player.dialogue_history) == 2
    stock = next(
        resource
        for resource in outcome.resulting_world_state.resources
        if resource.id == "stabilizer_parts"
    )
    assert stock.quantity == 2

    replay = ProductWorldEngine().govern(
        command=exchange_command.model_copy(update={"id": "command_exchange_replay"}),
        intent=exchange_intent,
        world_state=outcome.resulting_world_state,
        content_package=package,
    )
    assert replay.verification.decision == VerificationDecision.REJECT
    assert replay.committed_event is None
    assert replay.resulting_world_state is None


def test_resource_exchange_rejects_without_prerequisite_player_state() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    original = package.initial_world_state.model_copy(
        update={
            "player": package.initial_world_state.player.model_copy(
                update={"id": "profile_1", "current_location_id": "market_row"}
            )
        }
    )

    outcome = ProductWorldEngine().govern(
        command=command(
            location_id="market_row",
            action_id="negotiate_resource",
            target_ids=("selka",),
        ),
        intent=ParsedPlayerIntent(
            normalized_action="negotiate_resource",
            actor_id="profile_1",
            target_ids=("selka",),
            confidence=1,
            safety_classification="requires_governance",
        ),
        world_state=original,
        content_package=package,
    )

    assert outcome.verification.decision == VerificationDecision.REJECT
    assert outcome.committed_event is None
    assert outcome.resulting_world_state is None
    assert next(
        resource for resource in original.resources if resource.id == "stabilizer_parts"
    ).quantity == 3
