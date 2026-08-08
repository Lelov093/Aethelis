from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from pydantic_settings.exceptions import SettingsError

from aethelis.config.errors import ConfigurationError
from aethelis.utils.redaction import redact_text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
PLACEHOLDER_VALUES = {
    "",
    "xxx",
    "changeme",
    "replace-me",
    "sk-xxx",
    "your-api-key",
    "your_api_key",
}


class Settings(BaseSettings):
    """Validated runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    openai_base_url: str
    openai_model: str
    openai_model_fallbacks: Annotated[tuple[str, ...], NoDecode] = ()
    openai_api_key: SecretStr
    openai_timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 60
    openai_max_retries: Annotated[int, Field(ge=0, le=10)] = 1
    openai_enable_thinking: bool | None = None

    embedding_provider: str = "volcengine_ark"
    embedding_base_url: str
    embedding_model: str
    embedding_api_key: SecretStr | None = None
    ark_base_url: str | None = None
    ark_api_key: SecretStr | None = None
    embedding_dimensions: Annotated[int, Field(gt=0, le=65536)]
    embedding_timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 60
    embedding_max_retries: Annotated[int, Field(ge=0, le=10)] = 1

    @field_validator("openai_model_fallbacks", mode="before")
    @classmethod
    def parse_fallback_models(cls, value: Any) -> tuple[str, ...]:
        if value is None or value == "":
            return ()
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return tuple(value)

    @field_validator(
        "openai_base_url",
        "embedding_base_url",
        "ark_base_url",
        "openai_model",
        "embedding_provider",
        "embedding_model",
    )
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("openai_base_url", "embedding_base_url", "ark_base_url")
    @classmethod
    def require_http_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.startswith(("https://", "http://")):
            raise ValueError("must be an HTTP(S) URL")
        return value.rstrip("/")

    @model_validator(mode="after")
    def reject_placeholder_secrets(self) -> Settings:
        for field_name, secret in (("OPENAI_API_KEY", self.openai_api_key),):
            value = secret.get_secret_value().strip()
            if _is_placeholder_secret(value):
                raise ValueError(f"{field_name} is empty or still uses a placeholder")
        if _is_placeholder_secret(self.resolved_embedding_api_key.get_secret_value()):
            raise ValueError(
                "ARK_API_KEY and EMBEDDING_API_KEY are empty or still use placeholders"
            )
        return self

    @property
    def resolved_embedding_api_key(self) -> SecretStr:
        if self.ark_api_key is not None and not _is_placeholder_secret(
            self.ark_api_key.get_secret_value()
        ):
            return self.ark_api_key
        if self.embedding_api_key is not None:
            return self.embedding_api_key
        return SecretStr("")

    @property
    def resolved_embedding_base_url(self) -> str:
        return self.ark_base_url or self.embedding_base_url

    def safe_summary(self) -> dict[str, object]:
        """Return non-sensitive configuration suitable for CLI output and logs."""

        return {
            "openai_base_url_domain": _url_domain(self.openai_base_url),
            "openai_model": self.openai_model,
            "openai_model_fallbacks": list(self.openai_model_fallbacks),
            "openai_api_key_present": True,
            "openai_timeout_seconds": self.openai_timeout_seconds,
            "openai_max_retries": self.openai_max_retries,
            "openai_enable_thinking": self.openai_enable_thinking,
            "embedding_provider": self.embedding_provider,
            "embedding_base_url_domain": _url_domain(self.resolved_embedding_base_url),
            "embedding_model": self.embedding_model,
            "embedding_api_key_present": not _is_placeholder_secret(
                self.resolved_embedding_api_key.get_secret_value()
            ),
            "embedding_key_source": (
                "ARK_API_KEY"
                if self.ark_api_key is not None
                and not _is_placeholder_secret(self.ark_api_key.get_secret_value())
                else "EMBEDDING_API_KEY"
            ),
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_timeout_seconds": self.embedding_timeout_seconds,
            "embedding_max_retries": self.embedding_max_retries,
        }


def _is_placeholder_secret(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in PLACEHOLDER_VALUES or normalized.startswith("your-")


def _url_domain(value: str) -> str:
    return urlparse(value).netloc


def load_settings(env_file: Path | str | None = DEFAULT_ENV_FILE) -> Settings:
    """Load settings and convert validation failures into a sanitized domain error."""

    resolved_env_file = Path(env_file) if env_file is not None else None
    if resolved_env_file is not None and not resolved_env_file.is_file():
        raise ConfigurationError(
            f"Configuration file not found: {resolved_env_file}. Check the project .env file."
        )

    try:
        return Settings(_env_file=resolved_env_file)
    except (ValidationError, SettingsError) as exc:
        sanitized = redact_text(str(exc))
        raise ConfigurationError(
            f"Invalid Aethelis configuration. Check required values in .env.\n{sanitized}"
        ) from None
