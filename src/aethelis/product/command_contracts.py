from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from aethelis.product.content_contracts import ProductContentPackage
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.world import DialogueActKind, WorldState


class CommandInputMode(StrEnum):
    CONTEXTUAL_ACTION = "contextual_action"
    NATURAL_LANGUAGE_INTENT = "natural_language_intent"


class PlayerCommandStatus(StrEnum):
    SUBMITTED = "submitted"
    INTERPRETING = "interpreting"
    NEEDS_CLARIFICATION = "needs_clarification"
    READY_FOR_GOVERNANCE = "ready_for_governance"
    VERIFYING = "verifying"
    COMMITTED = "committed"
    REJECTED = "rejected"
    PROJECTING = "projecting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PlayerCommand(AethelisModel):
    id: Identifier
    idempotency_key: str = Field(min_length=8, max_length=200)
    principal_id: Identifier
    player_profile_id: Identifier
    world_instance_id: Identifier
    play_session_id: Identifier
    input_mode: CommandInputMode
    action_id: Identifier | None = None
    text: str | None = Field(default=None, max_length=2000)
    actor_id: Identifier
    target_ids: tuple[Identifier, ...] = ()
    target_hints: dict[Identifier, str] = Field(default_factory=dict)
    dialogue_interaction_id: Identifier | None = None
    location_id: Identifier | None = None
    client_scene_id: Identifier | None = None
    expected_world_version: int = Field(ge=0)
    locale: str = Field(min_length=2, max_length=35)
    status: PlayerCommandStatus = PlayerCommandStatus.SUBMITTED
    cancellation_requested: bool = False
    submitted_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def validate_input(self) -> PlayerCommand:
        if self.input_mode == CommandInputMode.CONTEXTUAL_ACTION and self.action_id is None:
            raise ValueError("contextual action requires action_id")
        if self.input_mode == CommandInputMode.NATURAL_LANGUAGE_INTENT and not self.text:
            raise ValueError("natural-language intent requires text")
        if not set(self.target_hints).issubset(self.target_ids):
            raise ValueError("target hints must reference submitted target ids")
        if any(not label.strip() or len(label) > 160 for label in self.target_hints.values()):
            raise ValueError("target hint labels must contain 1-160 characters")
        return self


class ParsedPlayerIntent(AethelisModel):
    normalized_action: str = Field(min_length=1, max_length=160)
    actor_id: Identifier
    target_ids: tuple[Identifier, ...] = ()
    constraints: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    missing_fields: tuple[str, ...] = ()
    safety_classification: str = Field(min_length=1, max_length=80)
    dialogue_act: DialogueActKind | None = None
    claim_text: str | None = Field(default=None, min_length=1, max_length=2000)
    provider_name: str | None = None
    model_name: str | None = None
    raw_text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_dialogue_claim(self) -> ParsedPlayerIntent:
        if self.dialogue_act == DialogueActKind.CLAIM and not self.claim_text:
            raise ValueError("claim dialogue act requires claim_text")
        if self.dialogue_act != DialogueActKind.CLAIM and self.claim_text is not None:
            raise ValueError("claim_text is only valid for a claim dialogue act")
        return self


class CommandExecution(AethelisModel):
    command_id: Identifier
    attempt_count: int = Field(ge=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    lease_owner: str | None = Field(default=None, max_length=160)
    lease_expires_at: AwareDatetime | None = None
    heartbeat_at: AwareDatetime | None = None
    parsed_intent: ParsedPlayerIntent | None = None
    error_code: str | None = Field(default=None, max_length=120)
    error_message: str | None = Field(default=None, max_length=500)
    retryable: bool = False
    created_at: AwareDatetime
    updated_at: AwareDatetime


class SubmitPlayerCommand(AethelisModel):
    idempotency_key: str = Field(min_length=8, max_length=200)
    player_profile_id: Identifier
    play_session_id: Identifier
    input_mode: CommandInputMode
    action_id: Identifier | None = None
    text: str | None = Field(default=None, max_length=2000)
    actor_id: Identifier
    target_ids: tuple[Identifier, ...] = ()
    target_hints: dict[Identifier, str] = Field(default_factory=dict)
    dialogue_interaction_id: Identifier | None = None
    location_id: Identifier | None = None
    client_scene_id: Identifier | None = None
    expected_world_version: int = Field(ge=0)
    locale: str = Field(default="en", min_length=2, max_length=35)

    @model_validator(mode="after")
    def validate_input(self) -> SubmitPlayerCommand:
        if self.input_mode == CommandInputMode.CONTEXTUAL_ACTION and self.action_id is None:
            raise ValueError("contextual action requires action_id")
        if self.input_mode == CommandInputMode.NATURAL_LANGUAGE_INTENT and not self.text:
            raise ValueError("natural-language intent requires text")
        if not set(self.target_hints).issubset(self.target_ids):
            raise ValueError("target hints must reference submitted target ids")
        if any(not label.strip() or len(label) > 160 for label in self.target_hints.values()):
            raise ValueError("target hint labels must contain 1-160 characters")
        return self


class CommandResultView(AethelisModel):
    command_id: Identifier
    status: PlayerCommandStatus
    message: str = Field(min_length=1, max_length=500)
    source_world_version: int = Field(ge=0)
    resulting_world_version: int | None = Field(default=None, ge=0)
    snapshot_id: Identifier | None = None
    consequences: tuple[str, ...] = ()
    available_actions: tuple[str, ...] = ()
    created_at: AwareDatetime


class GovernanceWorkItem(AethelisModel):
    command: PlayerCommand
    execution: CommandExecution
    world_state: WorldState
    source_snapshot_id: Identifier
    current_world_version: int = Field(ge=0)
    content_version_id: Identifier
    content_package: ProductContentPackage | None = None


class CommandReceipt(AethelisModel):
    command: PlayerCommand
    execution: CommandExecution
    status_url: str
    result: CommandResultView | None = None
