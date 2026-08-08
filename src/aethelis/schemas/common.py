from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Identifier = Annotated[str, Field(min_length=1, pattern=r"^[a-z][a-z0-9_:-]*$")]
SchemaVersion = Annotated[str, Field(min_length=1)]


class AethelisModel(BaseModel):
    """Base model for strict domain contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ReferenceScope(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class ConfidenceBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecordStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
