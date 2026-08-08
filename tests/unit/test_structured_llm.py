from __future__ import annotations

from pydantic import BaseModel, Field

from aethelis.llm.structured import _structured_prompt, parse_structured_output
from aethelis.providers import ProviderAttempt


class ExampleAction(BaseModel):
    action_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)


ATTEMPTS = (ProviderAttempt(model="test-model", success=True, latency_ms=1),)


def test_structured_output_parses_and_validates() -> None:
    result = parse_structured_output(
        raw_text='{"action_id": "a1", "actor_id": "mira"}',
        schema_type=ExampleAction,
        model_name="test-model",
        provider_name="fake_test_provider",
        latency_ms=1,
        attempts=ATTEMPTS,
    )

    assert result.success
    assert result.data == ExampleAction(action_id="a1", actor_id="mira")
    assert result.raw_text_sha256
    assert not hasattr(result, "raw_text")


def test_structured_output_reports_json_parse_error() -> None:
    result = parse_structured_output(
        raw_text="{not-json",
        schema_type=ExampleAction,
        model_name="test-model",
        provider_name="fake_test_provider",
        latency_ms=1,
        attempts=ATTEMPTS,
    )

    assert not result.success
    assert result.json_parse_error is not None
    assert result.validation_error is None


def test_structured_output_reports_validation_error() -> None:
    result = parse_structured_output(
        raw_text='{"action_id": ""}',
        schema_type=ExampleAction,
        model_name="test-model",
        provider_name="fake_test_provider",
        latency_ms=1,
        attempts=ATTEMPTS,
    )

    assert not result.success
    assert result.json_parse_error is None
    assert result.validation_error is not None


def test_structured_prompt_uses_compact_json_only_guard() -> None:
    prompt = _structured_prompt("base prompt", ExampleAction)

    assert "JSON only" in prompt
    assert "No markdown" in prompt
    assert "No markdown, explanation, or extra keys" in prompt
    assert "StateDiff" in prompt
    assert "CanonFact" in prompt
    assert "```" not in prompt
    assert "\n  " not in prompt
