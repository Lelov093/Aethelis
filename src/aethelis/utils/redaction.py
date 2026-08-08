from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "embedding_api_key",
    "openai_api_key",
    "password",
    "secret",
    "token",
}
SENSITIVE_PATTERNS = (
    re.compile(r"(?i)\b(Bearer)\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b((?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{3,}\b"),
    re.compile(r"(?i)(https?://[^/\s:@]+:)[^@\s/]+@"),
)


def redact_text(value: object) -> str:
    """Remove common credential forms from arbitrary text."""

    text = str(value)
    for pattern in SENSITIVE_PATTERNS:
        if pattern.pattern.startswith("(?i)\\b(Bearer)"):
            text = pattern.sub(r"\1 " + REDACTED, text)
        elif pattern.pattern.startswith("(?i)\\b((?:api"):
            text = pattern.sub(r"\1" + REDACTED, text)
        elif pattern.pattern.startswith("(?i)(https?"):
            text = pattern.sub(r"\1" + REDACTED + "@", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def redact_data(value: Any) -> Any:
    """Recursively redact sensitive mapping values and credential-like strings."""

    if isinstance(value, Mapping):
        return {
            key: REDACTED if str(key).lower() in SENSITIVE_KEYS else redact_data(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_data(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


class RedactingFilter(logging.Filter):
    """Logging filter that sanitizes messages and structured arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(record.getMessage())
        record.args = ()
        if record.exc_text:
            record.exc_text = redact_text(record.exc_text)
        return True
