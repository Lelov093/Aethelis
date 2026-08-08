from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from aethelis.schemas.common import AethelisModel, Identifier


class GoalStatus(StrEnum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class RelationshipKind(StrEnum):
    ALLY = "ally"
    TENSE = "tense"
    TRUSTS = "trusts"
    DISTRUSTS = "distrusts"
    TRANSACTIONAL = "transactional"
    PROFESSIONAL = "professional"


class Goal(AethelisModel):
    id: Identifier
    description: str = Field(min_length=1)
    status: GoalStatus = GoalStatus.ACTIVE
    priority: int = Field(default=1, ge=1, le=5)


class AgentCognitiveState(AethelisModel):
    goals: tuple[Goal, ...] = ()
    knowledge_boundaries: tuple[str, ...] = ()
    attention: tuple[str, ...] = ()


class AgentProfile(AethelisModel):
    id: Identifier
    name: str = Field(min_length=1)
    faction_id: Identifier | None = None
    role: str = Field(min_length=1)
    current_location_id: Identifier
    role_station: str | None = None
    public_summary: str = Field(min_length=1)
    private_summary: str | None = None
    cognitive_state: AgentCognitiveState


class RelationshipRecord(AethelisModel):
    id: Identifier
    source_agent_id: Identifier
    target_agent_id: Identifier
    kind: RelationshipKind
    summary: str = Field(min_length=1)
    trust: int = Field(ge=-5, le=5)
