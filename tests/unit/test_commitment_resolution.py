from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from aethelis.product.command_contracts import (
    CommandInputMode,
    ParsedPlayerIntent,
    PlayerCommand,
)
from aethelis.product.content_loader import ProductContentPackageLoader
from aethelis.product.projections import _achieved_outcomes, _scene_actions
from aethelis.product.world_engine import ProductWorldEngine
from aethelis.schemas.events import VerificationDecision

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 8, 3, 17, 0, tzinfo=UTC)


def test_repair_consumes_parts_fulfills_commitment_and_reaches_continuation() -> None:
    package, exchanged = _exchanged_world()
    repair_world = exchanged.model_copy(
        update={
            "player": exchanged.player.model_copy(update={"current_location_id": "old_aqueduct"})
        }
    )
    scene_actions = _scene_actions(
        package=package,
        locale="zh-CN",
        world=repair_world,
        location_id="old_aqueduct",
        has_undiscovered=False,
    )
    assert any(action.action_id == "repair_regulator" for action in scene_actions)

    outcome = ProductWorldEngine().govern(
        command=_command(
            "command_repair",
            action_id="repair_regulator",
            location_id="old_aqueduct",
            target_id="dawn_regulator",
        ),
        intent=_intent("repair_regulator", "dawn_regulator"),
        world_state=repair_world,
        content_package=package,
    )

    assert outcome.verification.decision == VerificationDecision.COMMIT
    assert outcome.apply_report is not None and outcome.apply_report.applied_patch_count == 4
    assert outcome.committed_event.tags == (
        "regulator_pressure_contained",
        "civic_response_active",
    )
    player = outcome.resulting_world_state.player
    assert player.inventory == ()
    assert player.commitments[0].status.value == "fulfilled"
    assert player.commitments[0].resolved_event_id == "committed_command_repair"
    assert player.knowledge[-1].id == "knowledge_regulator_pressure_contained"
    regulator = next(
        item for item in outcome.resulting_world_state.entities if item.id == "dawn_regulator"
    )
    assert "unstable" not in regulator.tags
    assert "regulator_pressure_contained" in regulator.tags
    achieved = _achieved_outcomes(package, outcome.resulting_world_state, "zh-CN")
    assert tuple(item.id for item in achieved) == ("outcome_city_holds",)

    replay = ProductWorldEngine().govern(
        command=_command(
            "command_repair_replay",
            action_id="repair_regulator",
            location_id="old_aqueduct",
            target_id="dawn_regulator",
        ),
        intent=_intent("repair_regulator", "dawn_regulator"),
        world_state=outcome.resulting_world_state,
        content_package=package,
    )
    assert replay.verification.decision == VerificationDecision.REJECT
    assert replay.committed_event is None


def test_blocked_repair_returns_a_player_recovery_route() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    world = package.initial_world_state.model_copy(
        update={
            "player": package.initial_world_state.player.model_copy(
                update={"id": "profile_1", "current_location_id": "old_aqueduct"}
            )
        }
    )

    outcome = ProductWorldEngine().govern(
        command=_command(
            "command_blocked_repair",
            action_id="repair_regulator",
            location_id="old_aqueduct",
            target_id="dawn_regulator",
        ),
        intent=_intent("repair_regulator", "dawn_regulator"),
        world_state=world,
        content_package=package,
    )

    assert outcome.verification.decision == VerificationDecision.REJECT
    assert outcome.player_message == (
        "若缺少稳定器零件，可在集市街调查供应与商会条件，而不是让世界陷入死锁。"
    )


def test_explicit_breach_persists_social_cost_but_does_not_block_later_repair() -> None:
    package, exchanged = _exchanged_world()
    market_actions = _scene_actions(
        package=package,
        locale="zh-CN",
        world=exchanged,
        location_id="market_row",
        has_undiscovered=False,
    )
    assert any(action.action_id == "break_commitment" for action in market_actions)

    breach = ProductWorldEngine().govern(
        command=_command(
            "command_breach",
            action_id="break_commitment",
            location_id="market_row",
            target_id="selka",
        ),
        intent=_intent("break_commitment", "selka"),
        world_state=exchanged,
        content_package=package,
    )

    assert breach.verification.decision == VerificationDecision.COMMIT
    assert breach.apply_report is not None and breach.apply_report.applied_patch_count == 3
    player = breach.resulting_world_state.player
    assert player.commitments[0].status.value == "broken"
    assert player.commitments[0].resolved_event_id == "committed_command_breach"
    assert player.relationships[0].trust == 0
    assert player.inventory[0].quantity == 1
    assert player.dialogue_history[-1].expression_evidence.source == "authored"

    repair_world = breach.resulting_world_state.model_copy(
        update={
            "player": breach.resulting_world_state.player.model_copy(
                update={"current_location_id": "old_aqueduct"}
            )
        }
    )
    repair = ProductWorldEngine().govern(
        command=_command(
            "command_repair_after_breach",
            action_id="repair_regulator",
            location_id="old_aqueduct",
            target_id="dawn_regulator",
        ),
        intent=_intent("repair_regulator", "dawn_regulator"),
        world_state=repair_world,
        content_package=package,
    )

    assert repair.verification.decision == VerificationDecision.COMMIT
    assert repair.resulting_world_state.player.commitments[0].status.value == "broken"
    assert repair.resulting_world_state.player.inventory == ()
    assert tuple(
        item.id for item in _achieved_outcomes(package, repair.resulting_world_state, "zh-CN")
    ) == ("outcome_city_holds",)


def _exchanged_world():
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    initial = package.initial_world_state.model_copy(
        update={
            "player": package.initial_world_state.player.model_copy(
                update={"id": "profile_1", "current_location_id": "market_row"}
            )
        }
    )
    dialogue = ProductWorldEngine().govern(
        command=_command(
            "command_dialogue",
            action_id="ask_character",
            location_id="market_row",
            target_id="selka",
        ),
        intent=_intent("ask_character", "selka"),
        world_state=initial,
        content_package=package,
    )
    exchange = ProductWorldEngine().govern(
        command=_command(
            "command_exchange",
            action_id="negotiate_resource",
            location_id="market_row",
            target_id="selka",
        ),
        intent=_intent("negotiate_resource", "selka"),
        world_state=dialogue.resulting_world_state,
        content_package=package,
    )
    assert exchange.verification.decision == VerificationDecision.COMMIT
    return package, exchange.resulting_world_state


def _command(
    command_id: str,
    *,
    action_id: str,
    location_id: str,
    target_id: str,
) -> PlayerCommand:
    return PlayerCommand(
        id=command_id,
        idempotency_key=f"{command_id}-request",
        principal_id="principal_1",
        player_profile_id="profile_1",
        world_instance_id="instance_1",
        play_session_id="session_1",
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id=action_id,
        actor_id="profile_1",
        target_ids=(target_id,),
        location_id=location_id,
        expected_world_version=0,
        locale="zh-CN",
        submitted_at=NOW,
        updated_at=NOW,
    )


def _intent(action_id: str, target_id: str) -> ParsedPlayerIntent:
    return ParsedPlayerIntent(
        normalized_action=action_id,
        actor_id="profile_1",
        target_ids=(target_id,),
        confidence=1,
        safety_classification="requires_governance",
    )
