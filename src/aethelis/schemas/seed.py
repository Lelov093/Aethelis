from __future__ import annotations

from pydantic import Field

from aethelis.schemas.agents import AgentProfile, RelationshipRecord
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.ledger import BeliefRecord, MemoryRecord, SecretRecord
from aethelis.schemas.metadata import MetadataSeed
from aethelis.schemas.world import WorldState


class SeedManifest(AethelisModel):
    seed_id: Identifier
    schema_version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    files: dict[str, str] = Field(default_factory=dict)


class AgentsSeed(AethelisModel):
    schema_version: str = Field(min_length=1)
    agents: tuple[AgentProfile, ...]
    relationships: tuple[RelationshipRecord, ...] = ()


class BeliefsSeed(AethelisModel):
    schema_version: str = Field(min_length=1)
    beliefs: tuple[BeliefRecord, ...] = ()
    secrets: tuple[SecretRecord, ...] = ()


class MemoriesSeed(AethelisModel):
    schema_version: str = Field(min_length=1)
    memories: tuple[MemoryRecord, ...] = ()


class SeedBundle(AethelisModel):
    manifest: SeedManifest
    world: WorldState
    agents: AgentsSeed
    beliefs: BeliefsSeed
    memories: MemoriesSeed
    metadata: MetadataSeed | None = None
