from __future__ import annotations

import hashlib
import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from aethelis.llm.base import LLMProvider, StructuredLLMResult
from aethelis.utils.redaction import redact_text

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


def generate_structured(
    provider: LLMProvider,
    prompt: str,
    schema_type: type[StructuredModelT],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> StructuredLLMResult[StructuredModelT]:
    """Generate JSON and validate it through a Pydantic schema.

    This is a contract layer, not a mock fallback. If parsing or validation
    fails, the returned result contains no data.
    """

    llm_result = provider.generate(
        _structured_prompt(prompt, schema_type),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return parse_structured_output(
        raw_text=llm_result.content,
        schema_type=schema_type,
        model_name=llm_result.model,
        provider_name=provider.provider_name,
        latency_ms=llm_result.latency_ms,
        usage=llm_result.usage,
        attempts=llm_result.attempts,
    )


def parse_structured_output(
    *,
    raw_text: str,
    schema_type: type[StructuredModelT],
    model_name: str,
    provider_name: str,
    latency_ms: int,
    attempts: tuple[Any, ...],
    usage: dict[str, int] | None = None,
) -> StructuredLLMResult[StructuredModelT]:
    raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    try:
        payload = json.loads(_extract_json(raw_text))
    except json.JSONDecodeError as exc:
        return StructuredLLMResult(
            data=None,
            raw_text_sha256=raw_hash,
            json_parse_error=redact_text(f"{type(exc).__name__}: {exc.msg}"),
            validation_error=None,
            model_name=model_name,
            provider_name=provider_name,
            latency_ms=latency_ms,
            usage=usage or {},
            attempts=attempts,
        )

    try:
        data = schema_type.model_validate(payload)
    except ValidationError as exc:
        return StructuredLLMResult(
            data=None,
            raw_text_sha256=raw_hash,
            json_parse_error=None,
            validation_error=redact_text(str(exc)),
            model_name=model_name,
            provider_name=provider_name,
            latency_ms=latency_ms,
            usage=usage or {},
            attempts=attempts,
        )

    return StructuredLLMResult(
        data=data,
        raw_text_sha256=raw_hash,
        json_parse_error=None,
        validation_error=None,
        model_name=model_name,
        provider_name=provider_name,
        latency_ms=latency_ms,
        usage=usage or {},
        attempts=attempts,
    )


def _structured_prompt(prompt: str, schema_type: type[BaseModel]) -> str:
    schema = json.dumps(
        schema_type.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"{prompt}\n\n"
        "JSON only. No markdown, explanation, or extra keys. "
        "Must validate against this Pydantic JSON Schema. "
        "Do not output StateDiff, CanonFact, or world/canon mutation.\n"
        f"{schema}"
    )


def _extract_json(raw_text: str) -> str:
    stripped = raw_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return stripped
