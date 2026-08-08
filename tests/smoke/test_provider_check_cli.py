import importlib
from pathlib import Path

from typer.testing import CliRunner

from aethelis.cli.app import app
from aethelis.providers import ConnectivityReport, ProviderAttempt

runner = CliRunner()


def write_valid_env(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=https://llm.example.test/v1",
                "OPENAI_MODEL=primary-model",
                "OPENAI_API_KEY=sk-test-openai-secret",
                "EMBEDDING_PROVIDER=volcengine_ark",
                "EMBEDDING_BASE_URL=https://ark.example.test/api/v3",
                "EMBEDDING_MODEL=embedding-model",
                "EMBEDDING_API_KEY=sk-test-embedding-secret",
                "EMBEDDING_DIMENSIONS=1024",
                "EMBEDDING_TIMEOUT_SECONDS=10",
                "EMBEDDING_MAX_RETRIES=0",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_provider_check_all_outputs_safe_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = write_valid_env(tmp_path / ".env")
    llm_report = ConnectivityReport(
        provider="openai_compatible",
        base_url_domain="llm.example.test",
        success=True,
        model="primary-model",
        latency_ms=12,
        attempts=(ProviderAttempt("primary-model", True, 12),),
    )
    embedding_report = ConnectivityReport(
        provider="volcengine_ark",
        base_url_domain="ark.example.test",
        success=True,
        model="embedding-model",
        latency_ms=20,
        dimensions=1024,
        attempts=(ProviderAttempt("embedding-model", True, 20),),
    )
    cli_module = importlib.import_module("aethelis.cli.app")
    monkeypatch.setattr(cli_module, "check_llm", lambda _: llm_report)
    monkeypatch.setattr(cli_module, "check_embedding", lambda _: embedding_report)

    result = runner.invoke(
        app,
        ["provider-check", "--all", "--env-file", str(env_file)],
    )

    assert result.exit_code == 0
    assert '"dimensions": 1024' in result.stdout
    assert '"ssl_verification": "enabled"' in result.stdout
    assert '"ssl_cert_dir_has_certificate_files"' in result.stdout
    assert "sk-test-openai-secret" not in result.stdout
    assert "sk-test-embedding-secret" not in result.stdout
