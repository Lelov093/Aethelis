from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from aethelis.utils.redaction import redact_text


@dataclass(frozen=True)
class ProviderAttempt:
    model: str
    success: bool
    latency_ms: int
    error_type: str | None = None
    error_summary: str | None = None

    def safe_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "error_type": self.error_type,
            "error_summary": (
                redact_text(self.error_summary) if self.error_summary is not None else None
            ),
        }


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        attempts: tuple[ProviderAttempt, ...] = (),
    ) -> None:
        super().__init__(redact_text(message))
        self.provider = provider
        self.attempts = attempts


@dataclass(frozen=True)
class ConnectivityReport:
    provider: str
    base_url_domain: str
    success: bool
    model: str | None = None
    latency_ms: int | None = None
    dimensions: int | None = None
    usage: dict[str, int] = field(default_factory=dict)
    attempts: tuple[ProviderAttempt, ...] = ()
    error_type: str | None = None
    error_summary: str | None = None

    def safe_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url_domain": self.base_url_domain,
            "success": self.success,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "dimensions": self.dimensions,
            "usage": self.usage,
            "fallback_attempts": [attempt.safe_dict() for attempt in self.attempts],
            "error_type": self.error_type,
            "error_summary": (
                redact_text(self.error_summary) if self.error_summary is not None else None
            ),
        }


def url_domain(url: str) -> str:
    return urlparse(url).netloc
