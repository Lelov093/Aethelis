import json
from pathlib import Path

import httpx

from aethelis.config.settings import load_settings
from aethelis.llm.openai_compatible import OpenAICompatibleLLMProvider


def write_env(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=https://llm.example.test/v1",
                "OPENAI_MODEL=primary-model",
                "OPENAI_MODEL_FALLBACKS=fallback-model",
                "OPENAI_API_KEY=sk-test-openai-secret",
                "OPENAI_TIMEOUT_SECONDS=10",
                "OPENAI_MAX_RETRIES=0",
                "EMBEDDING_PROVIDER=volcengine_ark",
                "EMBEDDING_BASE_URL=https://ark.example.test/api/v3",
                "EMBEDDING_MODEL=embedding-model",
                "EMBEDDING_API_KEY=sk-test-embedding-secret",
                "EMBEDDING_DIMENSIONS=3",
                "EMBEDDING_TIMEOUT_SECONDS=10",
                "EMBEDDING_MAX_RETRIES=0",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_llm_provider_uses_fallback_without_leaking_key(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = request.read().decode()
        if '"primary-model"' in body:
            return httpx.Response(404, json={"error": {"message": "model not found"}})
        return httpx.Response(
            200,
            json={
                "model": "fallback-model",
                "choices": [{"message": {"content": "AETHELIS_OK"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        )

    settings = load_settings(write_env(tmp_path / ".env"))
    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleLLMProvider(settings, client=client)

    result = provider.generate("connectivity check", max_tokens=8)

    assert result.model == "fallback-model"
    assert [attempt.model for attempt in result.attempts] == [
        "primary-model",
        "fallback-model",
    ]
    assert result.attempts[0].success is False
    assert result.attempts[1].success is True
    assert all("sk-test-openai-secret" not in repr(attempt) for attempt in result.attempts)
    assert len(requests) == 2


def test_llm_provider_can_include_configured_thinking_flag(tmp_path: Path) -> None:
    request_bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.read().decode()))
        return httpx.Response(
            200,
            json={
                "model": "primary-model",
                "choices": [{"message": {"content": "AETHELIS_OK"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        )

    env_file = write_env(tmp_path / ".env")
    env_file.write_text(
        f"{env_file.read_text(encoding='utf-8')}\nOPENAI_ENABLE_THINKING=false",
        encoding="utf-8",
    )
    settings = load_settings(env_file)
    provider = OpenAICompatibleLLMProvider(
        settings,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    provider.generate("connectivity check")

    assert request_bodies[0]["enable_thinking"] is False


def test_llm_provider_stops_fallback_on_authentication_error(tmp_path: Path) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    settings = load_settings(write_env(tmp_path / ".env"))
    provider = OpenAICompatibleLLMProvider(
        settings,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    try:
        provider.generate("connectivity check")
    except Exception as exc:
        assert "sk-test-openai-secret" not in str(exc)
    else:
        raise AssertionError("Expected provider failure")
    assert calls == 1


def test_llm_provider_uses_fallback_on_permission_error(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        if '"primary-model"' in body:
            calls.append("primary-model")
            return httpx.Response(403, json={"error": {"message": "model forbidden"}})
        calls.append("fallback-model")
        return httpx.Response(
            200,
            json={
                "model": "fallback-model",
                "choices": [{"message": {"content": "AETHELIS_OK"}}],
            },
        )

    settings = load_settings(write_env(tmp_path / ".env"))
    provider = OpenAICompatibleLLMProvider(
        settings,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = provider.generate("connectivity check")

    assert result.model == "fallback-model"
    assert calls == ["primary-model", "fallback-model"]
