from __future__ import annotations

from aethelis.config.settings import Settings
from aethelis.embedding.volcengine_ark import VolcengineArkEmbeddingProvider
from aethelis.llm.openai_compatible import OpenAICompatibleLLMProvider
from aethelis.providers import (
    ConnectivityReport,
    ProviderAttempt,
    ProviderError,
    url_domain,
)

LLM_CHECK_PROMPT = "Reply with exactly: AETHELIS_OK"
EMBEDDING_CHECK_TEXT = "Aethelis 是一个世界状态治理型多智能体运行时。"


def check_llm(settings: Settings) -> ConnectivityReport:
    try:
        with OpenAICompatibleLLMProvider(settings) as provider:
            result = provider.generate(
                LLM_CHECK_PROMPT,
                max_tokens=16,
                temperature=0.0,
            )
        return ConnectivityReport(
            provider=provider.provider_name,
            base_url_domain=url_domain(settings.openai_base_url),
            success=True,
            model=result.model,
            latency_ms=result.latency_ms,
            usage=result.usage,
            attempts=result.attempts,
        )
    except ProviderError as exc:
        return _failure_report(
            provider=exc.provider,
            base_url=settings.openai_base_url,
            attempts=exc.attempts,
            error=exc,
        )
    except Exception as exc:
        return _failure_report(
            provider="openai_compatible",
            base_url=settings.openai_base_url,
            attempts=(),
            error=exc,
        )


def check_embedding(settings: Settings) -> ConnectivityReport:
    try:
        provider = VolcengineArkEmbeddingProvider(settings)
        result = provider.embed_text(EMBEDDING_CHECK_TEXT)
        attempt = ProviderAttempt(
            model=result.model,
            success=True,
            latency_ms=result.latency_ms,
        )
        return ConnectivityReport(
            provider=provider.provider_name,
            base_url_domain=url_domain(settings.resolved_embedding_base_url),
            success=True,
            model=result.model,
            latency_ms=result.latency_ms,
            dimensions=result.dimensions,
            usage=result.usage,
            attempts=(attempt,),
        )
    except ProviderError as exc:
        return _failure_report(
            provider=exc.provider,
            base_url=settings.resolved_embedding_base_url,
            attempts=exc.attempts,
            error=exc,
        )
    except Exception as exc:
        return _failure_report(
            provider=settings.embedding_provider,
            base_url=settings.resolved_embedding_base_url,
            attempts=(),
            error=exc,
        )


def _failure_report(
    *,
    provider: str,
    base_url: str,
    attempts: tuple[ProviderAttempt, ...],
    error: Exception,
) -> ConnectivityReport:
    last_attempt = attempts[-1] if attempts else None
    return ConnectivityReport(
        provider=provider,
        base_url_domain=url_domain(base_url),
        success=False,
        model=last_attempt.model if last_attempt else None,
        latency_ms=last_attempt.latency_ms if last_attempt else None,
        attempts=attempts,
        error_type=last_attempt.error_type if last_attempt else type(error).__name__,
        error_summary=(
            last_attempt.error_summary if last_attempt else "Provider connectivity check failed."
        ),
    )
