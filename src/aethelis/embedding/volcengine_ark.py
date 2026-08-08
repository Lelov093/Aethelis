from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from volcenginesdkarkruntime import Ark

from aethelis.config.settings import Settings
from aethelis.embedding.base import EmbeddingInput, EmbeddingResult, TextEmbeddingInput
from aethelis.providers import ProviderAttempt, ProviderError
from aethelis.utils.redaction import redact_text

ArkClientFactory = Callable[..., Any]


class VolcengineArkEmbeddingProvider:
    """Volcengine Ark multimodal embedding adapter."""

    provider_name = "volcengine_ark"

    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: ArkClientFactory = Ark,
    ) -> None:
        self._settings = settings
        self._client = client_factory(
            api_key=settings.resolved_embedding_api_key.get_secret_value(),
            base_url=settings.resolved_embedding_base_url,
            timeout=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries,
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        return self.embed_multimodal([TextEmbeddingInput(text=text)])

    def embed_multimodal(self, inputs: list[EmbeddingInput]) -> EmbeddingResult:
        if not inputs:
            raise ValueError("At least one embedding input is required")

        started = time.perf_counter()
        try:
            response = self._client.multimodal_embeddings.create(
                model=self._settings.embedding_model,
                input=[item.model_dump(mode="json") for item in inputs],
                dimensions=self._settings.embedding_dimensions,
                encoding_format="float",
            )
            latency_ms = max(0, round((time.perf_counter() - started) * 1000))
            vector = tuple(response.data.embedding)
            _validate_vector(vector, self._settings.embedding_dimensions)
            return EmbeddingResult(
                embedding=vector,
                model=response.model,
                latency_ms=latency_ms,
                usage=_extract_usage(response.usage),
            )
        except Exception as exc:
            latency_ms = max(0, round((time.perf_counter() - started) * 1000))
            attempt = ProviderAttempt(
                model=self._settings.embedding_model,
                success=False,
                latency_ms=latency_ms,
                error_type=_classify_ark_error(exc),
                error_summary=redact_text(exc),
            )
            raise ProviderError(
                "Ark multimodal embedding request failed.",
                provider=self.provider_name,
                attempts=(attempt,),
            ) from None


def _validate_vector(vector: tuple[float, ...], expected_dimensions: int) -> None:
    if not vector:
        raise ValueError("Embedding response contains an empty vector")
    if len(vector) != expected_dimensions:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dimensions}, actual {len(vector)}"
        )
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding response contains a non-finite value")


def _extract_usage(usage: object) -> dict[str, int]:
    return {
        key: value
        for key in ("prompt_tokens", "total_tokens")
        if isinstance((value := getattr(usage, key, None)), int)
    }


def _classify_ark_error(exc: Exception) -> str:
    name = type(exc).__name__
    mapping = {
        "ArkAuthenticationError": "authentication_error",
        "ArkPermissionDeniedError": "permission_error",
        "ArkNotFoundError": "model_or_endpoint_not_found",
        "ArkRateLimitError": "rate_limit_error",
        "ArkAPITimeoutError": "timeout",
        "ArkAPIConnectionError": "connection_error",
        "ArkAPIResponseValidationError": "invalid_response",
    }
    return mapping.get(name, name)
