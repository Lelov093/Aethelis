from pathlib import Path
from types import SimpleNamespace

import pytest

from aethelis.config.settings import load_settings
from aethelis.embedding.volcengine_ark import VolcengineArkEmbeddingProvider
from aethelis.providers import ProviderError


def write_env(path: Path, *, dimensions: int = 3) -> Path:
    path.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=https://llm.example.test/v1",
                "OPENAI_MODEL=primary-model",
                "OPENAI_API_KEY=sk-test-openai-secret",
                "EMBEDDING_PROVIDER=volcengine_ark",
                "EMBEDDING_BASE_URL=https://ark.example.test/api/v3",
                "EMBEDDING_MODEL=doubao-embedding-vision-test",
                "EMBEDDING_API_KEY=sk-test-embedding-secret",
                f"EMBEDDING_DIMENSIONS={dimensions}",
                "EMBEDDING_TIMEOUT_SECONDS=10",
                "EMBEDDING_MAX_RETRIES=0",
            ]
        ),
        encoding="utf-8",
    )
    return path


class FakeEmbeddings:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.response


class FakeArkClient:
    def __init__(self, response: object) -> None:
        self.multimodal_embeddings = FakeEmbeddings(response)


def test_ark_provider_constructs_text_only_request(tmp_path: Path) -> None:
    response = SimpleNamespace(
        data=SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
        model="doubao-embedding-vision-test",
        usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
    )
    fake_client = FakeArkClient(response)
    constructed: dict[str, object] = {}

    def factory(**kwargs: object) -> FakeArkClient:
        constructed.update(kwargs)
        return fake_client

    settings = load_settings(write_env(tmp_path / ".env"))
    provider = VolcengineArkEmbeddingProvider(settings, client_factory=factory)
    result = provider.embed_text("Aethelis")

    assert constructed["api_key"] == "sk-test-embedding-secret"
    assert constructed["base_url"] == "https://ark.example.test/api/v3"
    assert result.dimensions == 3
    assert fake_client.multimodal_embeddings.kwargs == {
        "model": "doubao-embedding-vision-test",
        "input": [{"type": "text", "text": "Aethelis"}],
        "dimensions": 3,
        "encoding_format": "float",
    }


def test_ark_provider_rejects_dimension_mismatch(tmp_path: Path) -> None:
    response = SimpleNamespace(
        data=SimpleNamespace(embedding=[0.1, 0.2]),
        model="doubao-embedding-vision-test",
        usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
    )
    fake_client = FakeArkClient(response)
    settings = load_settings(write_env(tmp_path / ".env", dimensions=3))
    provider = VolcengineArkEmbeddingProvider(
        settings,
        client_factory=lambda **_: fake_client,
    )

    with pytest.raises(ProviderError) as exc_info:
        provider.embed_text("Aethelis")

    attempt = exc_info.value.attempts[0]
    assert attempt.error_type == "ValueError"
    assert "expected 3, actual 2" in (attempt.error_summary or "")
