from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from aethelis.product.command_contracts import CommandInputMode, ParsedPlayerIntent, PlayerCommand
from aethelis.product.content_loader import ProductContentPackageLoader
from aethelis.product.projections import _scene_actions
from aethelis.product.world_engine import ProductWorldEngine
from aethelis.schemas.events import VerificationDecision
from aethelis.schemas.world import (
    PlayerCommitment,
    PlayerCommitmentStatus,
    PlayerRelationshipState,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 8, 3, 22, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("status", "expected_kind", "expected_tag", "trust_before", "trust_after"),
    (
        ("fulfilled", "civic_support", "civic_supply_coordinator", 1, 2),
        ("broken", "social_withdrawal", "repair_support_withdrawn", 0, -1),
    ),
)
def test_wait_commits_one_actor_owned_branch_response(
    status: str,
    expected_kind: str,
    expected_tag: str,
    trust_before: int,
    trust_after: int,
) -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    world = package.initial_world_state
    commitment = PlayerCommitment(
        id="commitment_provide_repair_evidence",
        counterparty_entity_id="selka",
        description="Return repair evidence.",
        status=PlayerCommitmentStatus(status),
        related_resource_ids=("stabilizer_parts",),
        committed_event_id="committed_exchange",
        resolved_event_id=f"committed_{status}",
    )
    relationship = PlayerRelationshipState(
        character_id="selka",
        trust=trust_before,
        interaction_count=2,
        last_committed_event_id=f"committed_{status}",
    )
    player = world.player.model_copy(
        update={
            "id": "profile_1",
            "current_location_id": "old_aqueduct",
            "commitments": (commitment,),
            "relationships": (relationship,),
        }
    )
    entities = tuple(
        entity.model_copy(
            update={
                "tags": (
                    *entity.tags,
                    "calibration_key_secured",
                    "stabilizer_parts_secured",
                    "regulator_repaired",
                )
            }
        )
        if entity.id == "dawn_regulator"
        else entity
        for entity in world.entities
    )
    world = world.model_copy(update={"player": player, "entities": entities})
    actions = _scene_actions(
        package=package,
        locale="zh-CN",
        world=world,
        location_id="old_aqueduct",
        has_undiscovered=False,
    )
    assert sum(action.action_id == "wait_for_world_response" for action in actions) == 1

    outcome = ProductWorldEngine().govern(
        command=_command(f"response_{status}"),
        intent=_intent(),
        world_state=world,
        content_package=package,
    )

    assert outcome.verification.decision == VerificationDecision.COMMIT
    assert outcome.proposal.proposer_agent_id == "selka"
    assert outcome.candidate.actor_agent_id == "selka"
    assert outcome.apply_report.applied_patch_count == 3
    response = outcome.resulting_world_state.player.world_responses[0]
    assert response.response_kind == expected_kind
    assert response.actor_entity_id == "selka"
    assert outcome.resulting_world_state.player.relationships[0].trust == trust_after
    selka = next(item for item in outcome.resulting_world_state.entities if item.id == "selka")
    assert expected_tag in selka.tags

    replay = ProductWorldEngine().govern(
        command=_command(f"response_{status}_replay"),
        intent=_intent(),
        world_state=outcome.resulting_world_state,
        content_package=package,
    )
    assert replay.verification.decision == VerificationDecision.REJECT
    assert replay.committed_event is None


def test_wait_rejects_before_the_definitive_ending() -> None:
    package = ProductContentPackageLoader(ROOT).load(Path("content/mistgate/v1"))
    world = package.initial_world_state.model_copy(
        update={
            "player": package.initial_world_state.player.model_copy(
                update={"id": "profile_1", "current_location_id": "old_aqueduct"}
            )
        }
    )
    outcome = ProductWorldEngine().govern(
        command=_command("response_too_early"),
        intent=_intent(),
        world_state=world,
        content_package=package,
    )
    assert outcome.verification.decision == VerificationDecision.REJECT
    assert outcome.committed_event is None


def _command(suffix: str) -> PlayerCommand:
    return PlayerCommand(
        id=f"command_{suffix}",
        idempotency_key=f"command-{suffix}-request",
        principal_id="principal_1",
        player_profile_id="profile_1",
        world_instance_id="instance_1",
        play_session_id="session_1",
        input_mode=CommandInputMode.CONTEXTUAL_ACTION,
        action_id="wait_for_world_response",
        actor_id="profile_1",
        location_id="old_aqueduct",
        expected_world_version=0,
        locale="zh-CN",
        submitted_at=NOW,
        updated_at=NOW,
    )


def _intent() -> ParsedPlayerIntent:
    return ParsedPlayerIntent(
        normalized_action="wait_for_world_response",
        actor_id="profile_1",
        confidence=1,
        safety_classification="requires_governance",
    )
