from datetime import UTC, datetime

from aethelis.db.command_repository import _command, _command_record
from aethelis.product.command_contracts import CommandInputMode, PlayerCommand


def test_dialogue_interaction_id_round_trips_inside_existing_target_json() -> None:
    now = datetime(2026, 8, 4, 16, 0, tzinfo=UTC)
    command = PlayerCommand(
        id="command_dialogue_round_trip",
        idempotency_key="dialogue-round-trip-0001",
        principal_id="principal_1",
        player_profile_id="profile_1",
        world_instance_id="world_1",
        play_session_id="session_1",
        input_mode=CommandInputMode.NATURAL_LANGUAGE_INTENT,
        text="你好",
        actor_id="profile_1",
        target_ids=("rowan",),
        target_hints={"rowan": "罗文·凯斯特"},
        dialogue_interaction_id="dialogue_interaction_1",
        location_id="council_square",
        expected_world_version=0,
        locale="zh-CN",
        submitted_at=now,
        updated_at=now,
    )

    row = _command_record(command)
    restored = _command(row)

    assert row.target_ids_json["dialogue_interaction_id"] == "dialogue_interaction_1"
    assert restored == command
