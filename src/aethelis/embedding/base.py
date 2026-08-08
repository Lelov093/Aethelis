from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, Field, HttpUrl


class TextEmbeddingInput(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class ImageURLPayload(BaseModel):
    url: HttpUrl


class ImageURLEmbeddingInput(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURLPayload


EmbeddingInput = Annotated[
    TextEmbeddingInput | ImageURLEmbeddingInput,
    Field(discriminator="type"),
]


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: tuple[float, ...]
    model: str
    latency_ms: int
    usage: dict[str, int]

    @property
    def dimensions(self) -> int:
        return len(self.embedding)


class EmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> EmbeddingResult: ...

    def embed_multimodal(self, inputs: list[EmbeddingInput]) -> EmbeddingResult: ...
