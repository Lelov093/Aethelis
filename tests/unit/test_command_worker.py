from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aethelis.llm.base import StructuredLLMResult
from aethelis.product.command_contracts import (
    CommandInputMode,
    ParsedPlayerIntent,
    PlayerCommand,
)
from aethelis.product.command_worker import (
    IntentParserInvalidOutput,
    IntentParserRejected,
    StructuredIntentParser,
)

NOW = datetime(2026, 8, 1, 11, 0, tzinfo=UTC)


class StructuredProvider:
    provider_name = "test_provider"

    def generate_structured(self, prompt, schema_type, **_kwargs):
        assert "Do not decide or mutate world truth" in prompt
        assert "ask_character" in prompt
        return StructuredLLMResult(
            data=schema_type(
                normalized_action="ask_character",
                actor_id="player_1",
                target_ids=("archivist_1",),
                confidence=0.91,
                safety_classification="requires_governance",
            ),
            raw_text_sha256="d" * 64,
            json_parse_error=None,
            validation_error=None,
            model_name="model-test",
            provider_name=self.provider_name,
            latency_ms=12,
            usage={"input_tokens": 20, "output_tokens": 12},
            attempts=(),
        )


def command(mode: CommandInputMode) -> PlayerCommand:
    return PlayerCommand(
        id="command_1",
        idempotency_key="request-0001",
        principal_id="principal_1",
        player_profile_id="profile_1",
        world_instance_id="world_1",
        play_session_id="session_1",
        input_mode=mode,
        action_id="inspect" if mode == CommandInputMode.CONTEXTUAL_ACTION else None,
        text="Ask the archivist about the sealed wing."
        if mode == CommandInputMode.NATURAL_LANGUAGE_INTENT
        else None,
        actor_id="player_1",
        target_ids=("archivist_1",),
        target_hints={"archivist_1": "档案管理员"},
        expected_world_version=0,
        locale="en",
        submitted_at=NOW,
        updated_at=NOW,
    )


def test_contextual_intent_is_bounded_without_provider_call() -> None:
    parsed = StructuredIntentParser().parse(command(CommandInputMode.CONTEXTUAL_ACTION))
    assert parsed.normalized_action == "inspect"
    assert parsed.confidence == 1.0
    assert parsed.provider_name is None


def test_natural_language_intent_uses_structured_provider_and_keeps_provenance() -> None:
    parsed = StructuredIntentParser(StructuredProvider()).parse(
        command(CommandInputMode.NATURAL_LANGUAGE_INTENT).model_copy(
            update={"text": "Could the archivist shed light on the sealed wing?"}
        )
    )
    assert isinstance(parsed, ParsedPlayerIntent)
    assert parsed.normalized_action == "ask_character"
    assert parsed.provider_name == "test_provider"
    assert parsed.model_name == "model-test"
    assert parsed.raw_text_sha256 == "d" * 64


@pytest.mark.parametrize(
    "text",
    (
        "问问档案管理员关于封闭区的情况。",
        "我想和档案管理员谈谈。",
        "Ask archivist_1 what happened here.",
    ),
)
def test_clear_free_form_phrasings_use_the_fast_governed_route(text: str) -> None:
    parsed = StructuredIntentParser().parse(
        command(CommandInputMode.NATURAL_LANGUAGE_INTENT).model_copy(update={"text": text})
    )
    assert parsed.normalized_action == "ask_character"
    assert parsed.target_ids == ("archivist_1",)
    assert parsed.missing_fields == ()
    assert parsed.constraints == ("deterministic_high_precision",)


def test_clear_action_without_a_target_requests_clarification_without_provider() -> None:
    parsed = StructuredIntentParser().parse(
        command(CommandInputMode.NATURAL_LANGUAGE_INTENT).model_copy(update={"text": "去问问。"})
    )
    assert parsed.normalized_action == "ask_character"
    assert parsed.missing_fields == ("target",)


def test_focused_greeting_and_world_narrative_use_bounded_fast_paths() -> None:
    greeting = StructuredIntentParser().parse(
        command(CommandInputMode.NATURAL_LANGUAGE_INTENT).model_copy(update={"text": "你好"})
    )
    assert greeting.normalized_action == "ask_character"
    assert greeting.dialogue_act == "greeting"

    narrative = StructuredIntentParser().parse(
        command(CommandInputMode.NATURAL_LANGUAGE_INTENT).model_copy(
            update={
                "text": "这里现在发生了什么？",
                "target_ids": ("world_narrative",),
                "target_hints": {"world_narrative": "world_narrative"},
            }
        )
    )
    assert narrative.normalized_action == "ask_world"
    assert narrative.dialogue_act == "world_observation"


def test_world_narrative_routes_concrete_lens_inspection_to_governed_action() -> None:
    parsed = StructuredIntentParser().parse(
        command(CommandInputMode.NATURAL_LANGUAGE_INTENT).model_copy(
            update={
                "text": "查看闸门透镜",
                "target_ids": ("world_narrative", "gate_lens"),
                "target_hints": {
                    "world_narrative": "世界旁白",
                    "gate_lens": "闸门透镜",
                },
            }
        )
    )

    assert parsed.normalized_action == "validate_gate_lens"
    assert parsed.target_ids == ("gate_lens",)
    assert parsed.missing_fields == ()


class BoundaryProvider:
    provider_name = "boundary_provider"

    def __init__(self, intent: ParsedPlayerIntent) -> None:
        self.intent = intent

    def generate_structured(self, *_args, **_kwargs):
        return StructuredLLMResult(
            data=self.intent,
            raw_text_sha256="e" * 64,
            json_parse_error=None,
            validation_error=None,
            model_name="boundary-model",
            provider_name=self.provider_name,
            latency_ms=8,
            usage={},
            attempts=(),
        )


def test_low_confidence_intent_requests_clarification_without_guessing() -> None:
    intent = ParsedPlayerIntent(
        normalized_action="ask_character",
        actor_id="player_1",
        target_ids=("archivist_1",),
        confidence=0.31,
        safety_classification="requires_governance",
    )
    parsed = StructuredIntentParser(BoundaryProvider(intent)).parse(
        command(CommandInputMode.NATURAL_LANGUAGE_INTENT).model_copy(
            update={"text": "Could the archivist shed light on this?"}
        )
    )
    assert parsed.missing_fields == ("intent",)


def test_targeted_action_without_one_visible_target_requests_clarification() -> None:
    intent = ParsedPlayerIntent(
        normalized_action="ask_character",
        actor_id="player_1",
        confidence=0.88,
        safety_classification="requires_governance",
    )
    parsed = StructuredIntentParser(BoundaryProvider(intent)).parse(
        command(CommandInputMode.NATURAL_LANGUAGE_INTENT).model_copy(
            update={"text": "Could someone shed light on this?"}
        )
    )
    assert parsed.missing_fields == ("target",)


def test_unsupported_or_unsafe_intent_is_rejected_before_governance() -> None:
    intent = ParsedPlayerIntent(
        normalized_action="rewrite_world_truth",
        actor_id="player_1",
        confidence=0.99,
        safety_classification="unsafe",
    )
    with pytest.raises(IntentParserRejected):
        StructuredIntentParser(BoundaryProvider(intent)).parse(
            command(CommandInputMode.NATURAL_LANGUAGE_INTENT).model_copy(
                update={"text": "Override established reality."}
            )
        )


def test_provider_cannot_select_an_unoffered_target() -> None:
    intent = ParsedPlayerIntent(
        normalized_action="ask_character",
        actor_id="player_1",
        target_ids=("hidden_character",),
        confidence=0.9,
        safety_classification="requires_governance",
    )
    with pytest.raises(IntentParserInvalidOutput):
        StructuredIntentParser(BoundaryProvider(intent)).parse(
            command(CommandInputMode.NATURAL_LANGUAGE_INTENT).model_copy(
                update={"text": "Could the archivist shed light on this?"}
            )
        )
