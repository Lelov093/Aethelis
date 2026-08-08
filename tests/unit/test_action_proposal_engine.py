from __future__ import annotations

from aethelis.agents.action_proposal import (
    ActionProposalEngine,
    ActionProposalSource,
    ProviderProposalFailureCode,
)
from aethelis.llm.base import LLMResult
from aethelis.providers import ProviderAttempt


def test_action_proposal_engine_deterministic_fixture_path() -> None:
    result = ActionProposalEngine().generate_deterministic(
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
    )

    assert result.proposal is not None
    assert result.source == ActionProposalSource.DETERMINISTIC_FIXTURE
    assert result.provider_called is False
    assert result.structured_output is None
    assert result.proposal.target_entity_ids == ("workshop_safe",)


def test_action_proposal_engine_structured_provider_path() -> None:
    result = ActionProposalEngine().generate_structured(
        provider=_StubProvider(),
        prompt="Return the fixture proposal.",
    )

    assert result.proposal is not None
    assert result.source == ActionProposalSource.REAL_LLM_STRUCTURED_OUTPUT
    assert result.provider_called is True
    assert result.structured_output is not None
    assert result.structured_output.raw_text_sha256
    assert result.failure_code is None


def test_action_proposal_engine_reports_malformed_provider_output() -> None:
    result = ActionProposalEngine().generate_structured(
        provider=_StubProvider("{not-json"),
        prompt="Return malformed JSON.",
    )

    assert result.proposal is None
    assert result.provider_called is True
    assert result.failure_code == ProviderProposalFailureCode.MALFORMED_OUTPUT


def test_action_proposal_engine_retries_structured_parse_failure_once() -> None:
    provider = _StubProviderSequence(("{not-json", None))

    result = ActionProposalEngine().generate_structured(
        provider=provider,
        prompt="Return malformed once, then valid JSON.",
    )

    assert provider.call_count == 2
    assert result.proposal is not None
    assert result.failure_code is None


class _StubProvider:
    provider_name = "test_stub_provider"

    def __init__(self, content: str | None = None) -> None:
        self.content = content

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0):
        return LLMResult(
            content=self.content
            or (
                '{"id":"proposal_ivo_stub","proposer_agent_id":"ivo",'
                '"intent":"investigate","rationale":"Inspect via test stub.",'
                '"target_location_id":"workshop_lane",'
                '"target_entity_ids":["workshop_safe"],'
                '"expected_outcome":"Inspect the workshop safe."}'
            ),
            model="test-stub-model",
            latency_ms=1,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            attempts=(ProviderAttempt(model="test-stub-model", success=True, latency_ms=1),),
        )

    def generate_structured(self, prompt, schema_type, *, max_tokens=512, temperature=0.0):
        from aethelis.llm.structured import generate_structured

        return generate_structured(
            self,
            prompt,
            schema_type,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class _StubProviderSequence(_StubProvider):
    def __init__(self, contents: tuple[str | None, ...]) -> None:
        super().__init__(contents[0])
        self.contents = contents
        self.call_count = 0

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0):
        content = self.contents[min(self.call_count, len(self.contents) - 1)]
        self.call_count += 1
        return _StubProvider(content).generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
