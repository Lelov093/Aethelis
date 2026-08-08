from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from aethelis.product.command_contracts import CommandInputMode, ParsedPlayerIntent, PlayerCommand
from aethelis.product.content_loader import ProductContentPackageLoader
from aethelis.product.projections import _achieved_outcomes, _scene_actions
from aethelis.product.world_engine import ProductWorldEngine
from aethelis.schemas.events import VerificationDecision
from aethelis.schemas.world import PlayerCommitment, PlayerCommitmentStatus

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)


def test_key_lens_and_final_repair_form_a_persisted_governed_ending() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    world = package.initial_world_state
    resources = tuple(
        item.model_copy(
            update={
                "discovery_state": item.discovery_state.model_copy(
                    update={"discovered_by_agent_ids": ("profile_1",)}
                )
            }
        )
        if item.id in {"calibration_key", "gate_lens"}
        else item
        for item in world.resources
    )
    player = world.player.model_copy(
        update={"id": "profile_1", "current_location_id": "workshop_lane"}
    )
    world = world.model_copy(update={"player": player, "resources": resources})
    workshop_actions = _scene_actions(
        package=package,
        locale="zh-CN",
        world=world,
        location_id="workshop_lane",
        has_undiscovered=False,
    )
    assert any(action.action_id == "request_calibration_key" for action in workshop_actions)

    released = _govern(package, world, "release", "request_calibration_key", "ivo")
    assert released.verification.decision == VerificationDecision.COMMIT
    assert released.apply_report.applied_patch_count == 4
    key = next(
        item for item in released.resulting_world_state.resources if item.id == "calibration_key"
    )
    assert key.quantity == 0
    assert released.resulting_world_state.player.inventory[0].resource_id == "calibration_key"
    safe = next(
        item for item in released.resulting_world_state.entities if item.id == "workshop_safe"
    )
    assert "locked" not in safe.tags and "key_released" in safe.tags

    aqueduct_player = released.resulting_world_state.player.model_copy(
        update={"current_location_id": "old_aqueduct"}
    )
    aqueduct_world = released.resulting_world_state.model_copy(update={"player": aqueduct_player})
    aqueduct_actions = _scene_actions(
        package=package,
        locale="zh-CN",
        world=aqueduct_world,
        location_id="old_aqueduct",
        has_undiscovered=False,
    )
    assert any(action.action_id == "validate_gate_lens" for action in aqueduct_actions)
    validated = _govern(package, aqueduct_world, "validate", "validate_gate_lens", "gate_lens")
    assert validated.verification.decision == VerificationDecision.COMMIT
    assert validated.apply_report.applied_patch_count == 1

    entities = tuple(
        item.model_copy(
            update={
                "tags": (
                    "repair_target",
                    "high_impact",
                    "repair_progressed",
                    "regulator_pressure_contained",
                    "civic_response_active",
                )
            }
        )
        if item.id == "dawn_regulator"
        else item
        for item in validated.resulting_world_state.entities
    )
    broken = PlayerCommitment(
        id="commitment_provide_repair_evidence",
        counterparty_entity_id="selka",
        description="Return evidence.",
        status=PlayerCommitmentStatus.BROKEN,
        related_resource_ids=("stabilizer_parts",),
        committed_event_id="committed_exchange",
        resolved_event_id="committed_breach",
    )
    final_player = validated.resulting_world_state.player.model_copy(
        update={"commitments": (broken,)}
    )
    ready_world = validated.resulting_world_state.model_copy(
        update={"entities": entities, "player": final_player}
    )
    final_actions = _scene_actions(
        package=package,
        locale="zh-CN",
        world=ready_world,
        location_id="old_aqueduct",
        has_undiscovered=False,
    )
    assert any(action.action_id == "stabilize_regulator" for action in final_actions)
    stabilized = _govern(package, ready_world, "stabilize", "stabilize_regulator", "dawn_regulator")

    assert stabilized.verification.decision == VerificationDecision.COMMIT
    assert stabilized.apply_report.applied_patch_count == 3
    assert stabilized.resulting_world_state.player.inventory == ()
    assert stabilized.resulting_world_state.player.commitments[0].status.value == "broken"
    regulator = next(
        item for item in stabilized.resulting_world_state.entities if item.id == "dawn_regulator"
    )
    assert "repair_progressed" not in regulator.tags
    assert "regulator_pressure_contained" not in regulator.tags
    assert "regulator_repaired" in regulator.tags
    assert tuple(
        item.id for item in _achieved_outcomes(package, stabilized.resulting_world_state, "zh-CN")
    ) == ("outcome_regulator_stabilized",)


def test_progression_actions_reject_without_discovery_and_intermediate_repair() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    player = package.initial_world_state.player.model_copy(
        update={"id": "profile_1", "current_location_id": "workshop_lane"}
    )
    world = package.initial_world_state.model_copy(update={"player": player})

    release = _govern(package, world, "release_rejected", "request_calibration_key", "ivo")
    assert release.verification.decision == VerificationDecision.REJECT
    assert release.committed_event is None
    assert release.player_message == (
        "最终校准缺少校准钥匙；前往工坊巷调查保险柜，并请伊沃依法交付。"
    )


def _govern(package, world, suffix: str, action_id: str, target_id: str):
    command = PlayerCommand(
        id=f"command_{suffix}",
        idempotency_key=f"command-{suffix}-request",
        principal_id="principal_1",
        player_profile_id="profile_1",
        world_instance_id="instance_1",
        play_session_id="session_1",
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id=action_id,
        actor_id="profile_1",
        target_ids=(target_id,),
        location_id=world.player.current_location_id,
        expected_world_version=0,
        locale="zh-CN",
        submitted_at=NOW,
        updated_at=NOW,
    )
    intent = ParsedPlayerIntent(
        normalized_action=action_id,
        actor_id="profile_1",
        target_ids=(target_id,),
        confidence=1,
        safety_classification="requires_governance",
    )
    return ProductWorldEngine().govern(
        command=command,
        intent=intent,
        world_state=world,
        content_package=package,
    )
