from pathlib import Path

from typer.testing import CliRunner

from aethelis.cli.app import app

runner = CliRunner()


def write_valid_env(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=https://llm.example.test/v1",
                "OPENAI_MODEL=primary-model",
                "OPENAI_MODEL_FALLBACKS=fallback-a,fallback-b",
                "OPENAI_API_KEY=sk-test-openai-secret",
                "OPENAI_TIMEOUT_SECONDS=60",
                "OPENAI_MAX_RETRIES=1",
                "EMBEDDING_PROVIDER=volcengine_ark",
                "EMBEDDING_BASE_URL=https://embedding.example.test/api/v3",
                "EMBEDDING_MODEL=embedding-model",
                "EMBEDDING_API_KEY=sk-test-embedding-secret",
                "EMBEDDING_DIMENSIONS=1024",
                "EMBEDDING_TIMEOUT_SECONDS=60",
                "EMBEDDING_MAX_RETRIES=1",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "config-check" in result.stdout
    assert "r2-mistgate-long-horizon-db-run" in result.stdout


def test_r2_mistgate_long_horizon_db_help() -> None:
    result = runner.invoke(app, ["r2-mistgate-long-horizon-db-run", "--help"])

    assert result.exit_code == 0
    assert "--step-count" in result.stdout


def test_config_check_validates_without_external_call(tmp_path: Path) -> None:
    env_file = write_valid_env(tmp_path / ".env")

    result = runner.invoke(app, ["config-check", "--env-file", str(env_file)])

    assert result.exit_code == 0
    assert "Configuration valid" in result.stdout
    assert "No external provider call was made" in result.stdout
    assert "sk-test-openai-secret" not in result.stdout
    assert "sk-test-embedding-secret" not in result.stdout


def test_config_check_summary_is_redacted(tmp_path: Path) -> None:
    env_file = write_valid_env(tmp_path / ".env")

    result = runner.invoke(
        app,
        ["config-check", "--env-file", str(env_file), "--show-summary"],
    )

    assert result.exit_code == 0
    assert '"openai_api_key_present": true' in result.stdout
    assert '"embedding_api_key_present": true' in result.stdout
    assert "sk-test-openai-secret" not in result.stdout
    assert "sk-test-embedding-secret" not in result.stdout


def test_config_check_missing_file_returns_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["config-check", "--env-file", str(tmp_path / "missing.env")],
    )

    assert result.exit_code == 2
    assert "Configuration file not found" in result.stderr


def test_provider_check_requires_selection(tmp_path: Path) -> None:
    env_file = write_valid_env(tmp_path / ".env")

    result = runner.invoke(app, ["provider-check", "--env-file", str(env_file)])

    assert result.exit_code == 2
    assert "Select --llm, --embedding, or --all" in result.stderr
