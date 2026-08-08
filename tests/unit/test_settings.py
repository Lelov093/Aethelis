from pathlib import Path

import pytest

from aethelis.config.errors import ConfigurationError
from aethelis.config.settings import load_settings


def write_env(path: Path, **overrides: str) -> Path:
    values = {
        "OPENAI_BASE_URL": "https://llm.example.test/v1",
        "OPENAI_MODEL": "primary-model",
        "OPENAI_MODEL_FALLBACKS": "fallback-a, fallback-b",
        "OPENAI_API_KEY": "sk-test-openai-secret",
        "OPENAI_TIMEOUT_SECONDS": "60",
        "OPENAI_MAX_RETRIES": "1",
        "EMBEDDING_PROVIDER": "volcengine_ark",
        "EMBEDDING_BASE_URL": "https://embedding.example.test/api/v3",
        "EMBEDDING_MODEL": "embedding-model",
        "EMBEDDING_API_KEY": "sk-test-embedding-secret",
        "EMBEDDING_DIMENSIONS": "1024",
        "EMBEDDING_TIMEOUT_SECONDS": "60",
        "EMBEDDING_MAX_RETRIES": "1",
    }
    values.update(overrides)
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()),
        encoding="utf-8",
    )
    return path


def test_load_settings_parses_fallbacks_and_numeric_values(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path / ".env"))

    assert settings.openai_model_fallbacks == ("fallback-a", "fallback-b")
    assert settings.openai_enable_thinking is None
    assert settings.embedding_provider == "volcengine_ark"
    assert settings.embedding_dimensions == 1024
    assert settings.embedding_timeout_seconds == 60
    assert settings.embedding_max_retries == 1


def test_load_settings_parses_optional_thinking_flag(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path / ".env", OPENAI_ENABLE_THINKING="false"))

    assert settings.openai_enable_thinking is False
    assert settings.safe_summary()["openai_enable_thinking"] is False


def test_safe_summary_never_exposes_secrets(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path / ".env"))
    summary = settings.safe_summary()
    rendered = repr(summary)

    assert summary["openai_api_key_present"] is True
    assert summary["embedding_api_key_present"] is True
    assert summary["openai_base_url_domain"] == "llm.example.test"
    assert summary["embedding_base_url_domain"] == "embedding.example.test"
    assert "sk-test-openai-secret" not in rendered
    assert "sk-test-embedding-secret" not in rendered


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("OPENAI_API_KEY", ""),
        ("OPENAI_API_KEY", "sk-xxx"),
        ("EMBEDDING_API_KEY", "your-api-key"),
    ],
)
def test_empty_or_placeholder_secrets_fail_fast(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    with pytest.raises(ConfigurationError, match="Invalid Aethelis configuration"):
        load_settings(write_env(tmp_path / ".env", **{field: value}))


def test_missing_env_file_fails_fast_without_secret_output(tmp_path: Path) -> None:
    missing = tmp_path / "missing.env"

    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_settings(missing)


def test_invalid_numeric_configuration_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="embedding_dimensions"):
        load_settings(write_env(tmp_path / ".env", EMBEDDING_DIMENSIONS="0"))


def test_ark_configuration_takes_precedence(tmp_path: Path) -> None:
    settings = load_settings(
        write_env(
            tmp_path / ".env",
            ARK_API_KEY="sk-test-ark-secret",
            ARK_BASE_URL="https://ark.example.test/api/v3",
        )
    )

    assert settings.resolved_embedding_api_key.get_secret_value() == "sk-test-ark-secret"
    assert settings.resolved_embedding_base_url == "https://ark.example.test/api/v3"
    assert settings.safe_summary()["embedding_key_source"] == "ARK_API_KEY"


def test_placeholder_ark_key_falls_back_to_embedding_key(tmp_path: Path) -> None:
    settings = load_settings(
        write_env(
            tmp_path / ".env",
            ARK_API_KEY="sk-xxx",
            ARK_BASE_URL="https://ark.example.test/api/v3",
        )
    )

    assert settings.resolved_embedding_api_key.get_secret_value() == "sk-test-embedding-secret"
    assert settings.safe_summary()["embedding_key_source"] == "EMBEDDING_API_KEY"
