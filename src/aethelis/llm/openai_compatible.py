from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import httpx

from aethelis.config.settings import Settings
from aethelis.llm.base import LLMResult, StructuredLLMResult, StructuredModelT
from aethelis.llm.structured import generate_structured
from aethelis.providers import ProviderAttempt, ProviderError
from aethelis.utils.redaction import redact_text


class OpenAICompatibleLLMProvider:
    """OpenAI-compatible chat provider with ordered model fallback."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=settings.openai_timeout_seconds,
            follow_redirects=False,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAICompatibleLLMProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> LLMResult:
        attempts: list[ProviderAttempt] = []
        models = _deduplicate((self._settings.openai_model, *self._settings.openai_model_fallbacks))

        for model in models:
            for retry_index in range(self._settings.openai_max_retries + 1):
                started = time.perf_counter()
                try:
                    response = self._client.post(
                        f"{self._settings.openai_base_url}/chat/completions",
                        headers={
                            "Authorization": (
                                f"Bearer {self._settings.openai_api_key.get_secret_value()}"
                            ),
                            "Content-Type": "application/json",
                        },
                        json=self._request_body(
                            model=model,
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                    )
                    latency_ms = _elapsed_ms(started)
                    response.raise_for_status()
                    payload = response.json()
                    content = _extract_content(payload)
                    attempts.append(
                        ProviderAttempt(model=model, success=True, latency_ms=latency_ms)
                    )
                    return LLMResult(
                        content=content,
                        model=str(payload.get("model") or model),
                        latency_ms=latency_ms,
                        usage=_extract_usage(payload.get("usage")),
                        attempts=tuple(attempts),
                    )
                except Exception as exc:
                    latency_ms = _elapsed_ms(started)
                    error_type = _classify_http_error(exc)
                    attempts.append(
                        ProviderAttempt(
                            model=model,
                            success=False,
                            latency_ms=latency_ms,
                            error_type=error_type,
                            error_summary=redact_text(exc),
                        )
                    )
                    if _is_authentication_failure(exc):
                        break
                    if retry_index < self._settings.openai_max_retries and _is_retryable(exc):
                        continue
                    break
            if attempts[-1].error_type == "authentication_error":
                break

        raise ProviderError(
            "All configured LLM model attempts failed.",
            provider=self.provider_name,
            attempts=tuple(attempts),
        )

    def generate_structured(
        self,
        prompt: str,
        schema_type: type[StructuredModelT],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> StructuredLLMResult[StructuredModelT]:
        return generate_structured(
            self,
            prompt,
            schema_type,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _request_body(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self._settings.openai_enable_thinking is not None:
            body["enable_thinking"] = self._settings.openai_enable_thinking
        return body


def _deduplicate(models: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(model for model in models if model))


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _extract_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM response does not contain choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content is empty")
    return content


def _extract_usage(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if key in {"prompt_tokens", "completion_tokens", "total_tokens"} and isinstance(item, int)
    }


def _classify_http_error(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "connection_error"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 401:
            return "authentication_error"
        if status == 403:
            return "permission_error"
        if status == 404:
            return "model_or_endpoint_not_found"
        if status == 429:
            return "rate_limit_error"
        if status >= 500:
            return "provider_server_error"
        return f"http_{status}"
    if isinstance(exc, ValueError):
        return "invalid_response"
    return type(exc).__name__


def _is_authentication_failure(exc: Exception) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 401


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code == 429 or exc.response.status_code >= 500
    )
