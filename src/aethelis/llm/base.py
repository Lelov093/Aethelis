from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from aethelis.providers import ProviderAttempt

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    latency_ms: int
    usage: dict[str, int]
    attempts: tuple[ProviderAttempt, ...]


@dataclass(frozen=True)
class StructuredLLMResult(Generic[StructuredModelT]):
    data: StructuredModelT | None
    raw_text_sha256: str
    json_parse_error: str | None
    validation_error: str | None
    model_name: str
    provider_name: str
    latency_ms: int
    usage: dict[str, int]
    attempts: tuple[ProviderAttempt, ...]

    @property
    def success(self) -> bool:
        return self.data is not None


class LLMProvider(Protocol):
    provider_name: str

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> LLMResult: ...

    def generate_structured(
        self,
        prompt: str,
        schema_type: type[StructuredModelT],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> StructuredLLMResult[StructuredModelT]: ...
