from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.world import WorldState


class ProductRoute(AethelisModel):
    id: Identifier
    from_location_id: Identifier
    to_location_id: Identifier
    bidirectional: bool = True
    initially_known: bool = True
    required_access_tags: tuple[str, ...] = ()


class ProductActionDefinition(AethelisModel):
    action_id: Identifier
    intent: Literal["move", "observe", "investigate", "dialogue", "negotiate", "repair"]
    label_key: str = Field(min_length=1)
    target_type: Literal["location", "resource", "scene", "character", "entity"]
    mutates_world: bool


class ProductDialogueOption(AethelisModel):
    id: Identifier
    action_id: Identifier
    character_id: Identifier
    location_id: Identifier
    topic_ids: tuple[Identifier, ...] = Field(min_length=1)
    label_key: str = Field(min_length=1)
    response_key: str = Field(min_length=1)
    knowledge_id: Identifier
    knowledge_kind: Literal["confirmed_fact", "rumor"]
    knowledge_statement_key: str = Field(min_length=1)
    knowledge_confidence: Literal["low", "medium", "high"]
    source_canon_fact_id: Identifier | None = None
    relationship_delta: int = Field(default=0, ge=-1, le=1)

    @model_validator(mode="after")
    def validate_knowledge_source(self) -> ProductDialogueOption:
        if self.knowledge_kind == "confirmed_fact" and self.source_canon_fact_id is None:
            raise ValueError("confirmed dialogue knowledge requires a source Canon fact")
        if self.knowledge_kind == "rumor" and self.source_canon_fact_id is not None:
            raise ValueError("rumor dialogue knowledge cannot cite a Canon fact")
        return self


class ProductResourceExchangeOption(AethelisModel):
    id: Identifier
    action_id: Identifier
    character_id: Identifier
    location_id: Identifier
    resource_id: Identifier
    quantity: int = Field(ge=1)
    prerequisite_knowledge_ids: tuple[Identifier, ...] = Field(min_length=1)
    minimum_trust: int = Field(ge=-5, le=5)
    commitment_id: Identifier
    label_key: str = Field(min_length=1)
    response_key: str = Field(min_length=1)
    commitment_description_key: str = Field(min_length=1)


class ProductRepairOption(AethelisModel):
    id: Identifier
    action_id: Identifier
    location_id: Identifier
    target_entity_id: Identifier
    resource_id: Identifier
    quantity: int = Field(ge=1)
    commitment_id: Identifier
    outcome_id: Identifier
    required_target_tags: tuple[Identifier, ...] = Field(min_length=1)
    committed_event_tags: tuple[Identifier, ...] = Field(min_length=1)
    result_target_tags: tuple[Identifier, ...] = Field(min_length=1)
    knowledge_id: Identifier
    knowledge_statement_key: str = Field(min_length=1)
    label_key: str = Field(min_length=1)
    response_key: str = Field(min_length=1)


class ProductCommitmentBreachOption(AethelisModel):
    id: Identifier
    action_id: Identifier
    location_id: Identifier
    character_id: Identifier
    commitment_id: Identifier
    relationship_delta: int = Field(ge=-5, le=-1)
    label_key: str = Field(min_length=1)
    response_key: str = Field(min_length=1)


class ProductResourceReleaseOption(AethelisModel):
    id: Identifier
    action_id: Identifier
    location_id: Identifier
    character_id: Identifier
    container_entity_id: Identifier
    resource_id: Identifier
    quantity: int = Field(ge=1)
    required_discovery: bool = True
    knowledge_id: Identifier
    knowledge_statement_key: str = Field(min_length=1)
    label_key: str = Field(min_length=1)
    response_key: str = Field(min_length=1)


class ProductResourceValidationOption(AethelisModel):
    id: Identifier
    action_id: Identifier
    location_id: Identifier
    resource_id: Identifier
    required_discovery: bool = True
    knowledge_id: Identifier
    knowledge_statement_key: str = Field(min_length=1)
    label_key: str = Field(min_length=1)
    response_key: str = Field(min_length=1)


class ProductFinalRepairOption(AethelisModel):
    id: Identifier
    action_id: Identifier
    location_id: Identifier
    target_entity_id: Identifier
    consumed_resource_id: Identifier
    quantity: int = Field(ge=1)
    prerequisite_knowledge_ids: tuple[Identifier, ...] = Field(min_length=1)
    required_target_tags: tuple[Identifier, ...] = Field(min_length=1)
    removed_target_tags: tuple[Identifier, ...] = ()
    committed_event_tags: tuple[Identifier, ...] = Field(min_length=1)
    result_target_tags: tuple[Identifier, ...] = Field(min_length=1)
    outcome_id: Identifier
    knowledge_id: Identifier
    knowledge_statement_key: str = Field(min_length=1)
    label_key: str = Field(min_length=1)
    response_key: str = Field(min_length=1)


class ProductWorldResponseOption(AethelisModel):
    id: Identifier
    action_id: Identifier
    outcome_id: Identifier
    commitment_id: Identifier
    commitment_status: Literal["fulfilled", "broken"]
    actor_entity_id: Identifier
    response_kind: Literal["civic_support", "social_withdrawal"]
    relationship_delta: int = Field(ge=-1, le=1)
    committed_event_tags: tuple[Identifier, ...] = Field(min_length=1)
    result_actor_tags: tuple[Identifier, ...] = Field(min_length=1)
    label_key: str = Field(min_length=1)
    response_key: str = Field(min_length=1)


class ProductDialogueExpressionPolicy(AethelisModel):
    enabled: bool = False
    schema_version: Literal["dialogue_expression_v1"] = "dialogue_expression_v1"
    max_utterance_characters: int = Field(default=280, ge=80, le=500)
    max_total_latency_ms: int = Field(default=60_000, ge=1_000, le=180_000)
    max_total_tokens: int = Field(default=6_000, ge=256, le=20_000)
    semantic_review_required: bool = True

    @model_validator(mode="after")
    def require_review_when_enabled(self) -> ProductDialogueExpressionPolicy:
        if self.enabled and not self.semantic_review_required:
            raise ValueError("enabled dialogue expression requires semantic review")
        return self


class OnboardingObjective(AethelisModel):
    id: Identifier
    title_key: str = Field(min_length=1)
    description_key: str = Field(min_length=1)
    action_id: Identifier
    target_id: Identifier | None = None


class ObjectPresentation(AethelisModel):
    object_type: Literal["location", "character", "entity", "resource"]
    object_id: Identifier
    name_key: str = Field(min_length=1)
    summary_key: str = Field(min_length=1)
    accessibility_label_key: str = Field(min_length=1)
    visual_asset_id: Identifier | None = None


class ProductArtDirection(AethelisModel):
    style: str = Field(min_length=1)
    palette: tuple[str, ...] = Field(min_length=3)
    camera: str = Field(min_length=1)
    motion: str = Field(min_length=1)
    prohibited_treatments: tuple[str, ...] = ()


class ProductAssetPolicy(AethelisModel):
    ai_assisted_allowed: bool
    provenance_required: bool
    licensing_review_required: bool
    human_visual_approval_required: bool
    living_artist_imitation_allowed: bool = False


class ProductAssetSlot(AethelisModel):
    id: Identifier
    kind: Literal["location_scene", "character_portrait", "object_detail"]
    subject_id: Identifier
    accessibility_label_key: str = Field(min_length=1)
    status: Literal["planned", "produced", "approved"] = "planned"
    required_for_work_block: int = Field(ge=1, le=4)
    uri: str | None = None
    provenance_ref: str | None = None

    @model_validator(mode="after")
    def validate_produced_asset(self) -> ProductAssetSlot:
        if self.status in {"produced", "approved"} and not self.uri:
            raise ValueError("produced asset slot requires a URI")
        if self.status == "approved" and not self.provenance_ref:
            raise ValueError("approved asset slot requires provenance")
        return self


class ProductOpportunity(AethelisModel):
    id: Identifier
    title_key: str = Field(min_length=1)
    description_key: str = Field(min_length=1)
    location_id: Identifier
    action_id: Identifier
    target_id: Identifier | None = None
    is_optional: bool = False
    completion_knowledge_ids: tuple[Identifier, ...] = ()
    completion_inventory_resource_ids: tuple[Identifier, ...] = ()
    completion_discovered_resource_ids: tuple[Identifier, ...] = ()
    completion_target_tags: tuple[str, ...] = ()


class ProductPacingRule(AethelisModel):
    id: Identifier
    pressure_type: Identifier
    warning_level: int = Field(ge=0, le=10)
    crisis_level: int = Field(ge=0, le=10)
    cooldown_turns: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_levels(self) -> ProductPacingRule:
        if self.warning_level >= self.crisis_level:
            raise ValueError("pacing warning level must be lower than crisis level")
        return self


class ProductRecoveryPath(AethelisModel):
    id: Identifier
    blocked_action_id: Identifier
    alternative_action_id: Identifier
    target_location_id: Identifier
    guidance_key: str = Field(min_length=1)


class ProductOutcomeDefinition(AethelisModel):
    id: Identifier
    outcome_type: Literal["ending", "stable_continuation"]
    title_key: str = Field(min_length=1)
    description_key: str = Field(min_length=1)
    required_committed_event_tags: tuple[str, ...] = Field(min_length=1)


class ProductContentBlueprint(AethelisModel):
    package_schema_version: str = Field(min_length=1)
    package_id: Identifier
    world_definition_id: Identifier
    world_name_key: str = Field(min_length=1)
    player_summary_key: str = Field(min_length=1)
    content_version_id: Identifier
    engine_schema_version: str = Field(min_length=1)
    source_seed_path: str = Field(min_length=1)
    source_seed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    default_locale: str = Field(min_length=2, max_length=35)
    supported_locales: tuple[str, ...] = Field(min_length=1)
    entry_location_id: Identifier
    routes: tuple[ProductRoute, ...] = Field(min_length=1)
    actions: tuple[ProductActionDefinition, ...] = Field(min_length=1)
    dialogue_options: tuple[ProductDialogueOption, ...] = ()
    resource_exchange_options: tuple[ProductResourceExchangeOption, ...] = ()
    repair_options: tuple[ProductRepairOption, ...] = ()
    commitment_breach_options: tuple[ProductCommitmentBreachOption, ...] = ()
    resource_release_options: tuple[ProductResourceReleaseOption, ...] = ()
    resource_validation_options: tuple[ProductResourceValidationOption, ...] = ()
    final_repair_options: tuple[ProductFinalRepairOption, ...] = ()
    world_response_options: tuple[ProductWorldResponseOption, ...] = ()
    dialogue_expression_policy: ProductDialogueExpressionPolicy = Field(
        default_factory=ProductDialogueExpressionPolicy
    )
    onboarding: tuple[OnboardingObjective, ...] = Field(min_length=1)
    presentations: tuple[ObjectPresentation, ...] = Field(min_length=1)
    canon_text_keys: dict[Identifier, str] = Field(default_factory=dict)
    journal_seed_keys: tuple[str, ...] = ()
    art_direction: ProductArtDirection
    asset_policy: ProductAssetPolicy
    asset_slots: tuple[ProductAssetSlot, ...] = Field(min_length=1)
    opportunities: tuple[ProductOpportunity, ...] = Field(min_length=1)
    pacing_rules: tuple[ProductPacingRule, ...] = Field(min_length=1)
    recovery_paths: tuple[ProductRecoveryPath, ...] = Field(min_length=1)
    outcomes: tuple[ProductOutcomeDefinition, ...] = Field(min_length=1)
    compatible_from_content_versions: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_blueprint(self) -> ProductContentBlueprint:
        if self.default_locale not in self.supported_locales:
            raise ValueError("default locale must be supported")
        if len({route.id for route in self.routes}) != len(self.routes):
            raise ValueError("route ids must be unique")
        if len({action.action_id for action in self.actions}) != len(self.actions):
            raise ValueError("action ids must be unique")
        if len({option.id for option in self.dialogue_options}) != len(self.dialogue_options):
            raise ValueError("dialogue option ids must be unique")
        if len({option.knowledge_id for option in self.dialogue_options}) != len(
            self.dialogue_options
        ):
            raise ValueError("dialogue knowledge ids must be unique")
        if len({option.id for option in self.resource_exchange_options}) != len(
            self.resource_exchange_options
        ):
            raise ValueError("resource exchange option ids must be unique")
        if len({option.commitment_id for option in self.resource_exchange_options}) != len(
            self.resource_exchange_options
        ):
            raise ValueError("resource exchange commitment ids must be unique")
        if len({option.id for option in self.repair_options}) != len(self.repair_options):
            raise ValueError("repair option ids must be unique")
        if len({option.id for option in self.commitment_breach_options}) != len(
            self.commitment_breach_options
        ):
            raise ValueError("commitment breach option ids must be unique")
        if len({option.id for option in self.resource_release_options}) != len(
            self.resource_release_options
        ):
            raise ValueError("resource release option ids must be unique")
        if len({option.id for option in self.resource_validation_options}) != len(
            self.resource_validation_options
        ):
            raise ValueError("resource validation option ids must be unique")
        if len({option.id for option in self.final_repair_options}) != len(
            self.final_repair_options
        ):
            raise ValueError("final repair option ids must be unique")
        if len({option.id for option in self.world_response_options}) != len(
            self.world_response_options
        ):
            raise ValueError("world response option ids must be unique")
        if len({slot.id for slot in self.asset_slots}) != len(self.asset_slots):
            raise ValueError("asset slot ids must be unique")
        return self


class ProductContentPackage(AethelisModel):
    blueprint: ProductContentBlueprint
    localized_text: dict[str, dict[str, str]]
    initial_world_state: WorldState

    @model_validator(mode="after")
    def validate_package(self) -> ProductContentPackage:
        blueprint = self.blueprint
        if self.initial_world_state.world_id != blueprint.world_definition_id:
            raise ValueError("content package world does not match its definition")
        if self.initial_world_state.schema_version != blueprint.engine_schema_version:
            raise ValueError("content package engine schema version mismatch")
        location_ids = {location.id for location in self.initial_world_state.locations}
        entity_ids = {entity.id for entity in self.initial_world_state.entities}
        resource_ids = {resource.id for resource in self.initial_world_state.resources}
        canon_ids = {fact.id for fact in self.initial_world_state.canon_facts}
        if blueprint.entry_location_id not in location_ids:
            raise ValueError("entry location does not exist")
        for route in blueprint.routes:
            if (
                route.from_location_id not in location_ids
                or route.to_location_id not in location_ids
            ):
                raise ValueError(f"route {route.id} refers to an unknown location")
            if route.from_location_id == route.to_location_id:
                raise ValueError(f"route {route.id} cannot connect a location to itself")
        action_by_id = {action.action_id: action for action in blueprint.actions}
        action_ids = set(action_by_id)
        required_actions = {"move_to_location", "observe_scene", "investigate_area"}
        if not required_actions.issubset(action_ids):
            raise ValueError("P2 entry package is missing a required player action")
        for option in blueprint.dialogue_options:
            if option.action_id not in action_ids:
                raise ValueError(f"dialogue option {option.id} uses an unknown action")
            if option.character_id not in entity_ids:
                raise ValueError(f"dialogue option {option.id} uses an unknown character")
            if option.location_id not in location_ids:
                raise ValueError(f"dialogue option {option.id} uses an unknown location")
            if option.source_canon_fact_id and option.source_canon_fact_id not in canon_ids:
                raise ValueError(f"dialogue option {option.id} uses unknown Canon")
            action = action_by_id[option.action_id]
            if action.intent != "dialogue" or action.target_type != "character":
                raise ValueError(
                    f"dialogue option {option.id} requires a character dialogue action"
                )
        dialogue_knowledge_ids = {option.knowledge_id for option in blueprint.dialogue_options}
        exchange_commitment_ids = {
            option.commitment_id for option in blueprint.resource_exchange_options
        }
        for option in blueprint.resource_exchange_options:
            if option.action_id not in action_ids:
                raise ValueError(f"resource exchange {option.id} uses an unknown action")
            if option.character_id not in entity_ids:
                raise ValueError(f"resource exchange {option.id} uses an unknown character")
            if option.location_id not in location_ids:
                raise ValueError(f"resource exchange {option.id} uses an unknown location")
            if option.resource_id not in resource_ids:
                raise ValueError(f"resource exchange {option.id} uses an unknown resource")
            if not set(option.prerequisite_knowledge_ids).issubset(dialogue_knowledge_ids):
                raise ValueError(
                    f"resource exchange {option.id} uses unknown prerequisite knowledge"
                )
            action = action_by_id[option.action_id]
            if action.intent != "negotiate" or action.target_type != "character":
                raise ValueError(
                    f"resource exchange {option.id} requires a character negotiation action"
                )
        outcome_ids = {outcome.id for outcome in blueprint.outcomes}
        for option in blueprint.repair_options:
            if option.action_id not in action_ids:
                raise ValueError(f"repair option {option.id} uses an unknown action")
            if option.location_id not in location_ids:
                raise ValueError(f"repair option {option.id} uses an unknown location")
            if option.target_entity_id not in entity_ids:
                raise ValueError(f"repair option {option.id} uses an unknown target")
            if option.resource_id not in resource_ids:
                raise ValueError(f"repair option {option.id} uses an unknown resource")
            if option.commitment_id not in exchange_commitment_ids:
                raise ValueError(f"repair option {option.id} uses an unknown commitment")
            if option.outcome_id not in outcome_ids:
                raise ValueError(f"repair option {option.id} uses an unknown outcome")
            outcome = next(item for item in blueprint.outcomes if item.id == option.outcome_id)
            if not set(outcome.required_committed_event_tags).issubset(option.committed_event_tags):
                raise ValueError(f"repair option {option.id} does not emit its outcome tags")
            action = action_by_id[option.action_id]
            if action.intent != "repair" or action.target_type != "entity":
                raise ValueError(f"repair option {option.id} requires an entity repair action")
        for option in blueprint.commitment_breach_options:
            if option.action_id not in action_ids:
                raise ValueError(f"commitment breach {option.id} uses an unknown action")
            if option.location_id not in location_ids:
                raise ValueError(f"commitment breach {option.id} uses an unknown location")
            if option.character_id not in entity_ids:
                raise ValueError(f"commitment breach {option.id} uses an unknown character")
            if option.commitment_id not in exchange_commitment_ids:
                raise ValueError(f"commitment breach {option.id} uses an unknown commitment")
            action = action_by_id[option.action_id]
            if action.intent != "negotiate" or action.target_type != "character":
                raise ValueError(
                    f"commitment breach {option.id} requires a character negotiation action"
                )
        release_knowledge_ids = {
            option.knowledge_id for option in blueprint.resource_release_options
        }
        validation_knowledge_ids = {
            option.knowledge_id for option in blueprint.resource_validation_options
        }
        for option in blueprint.resource_release_options:
            if option.action_id not in action_ids:
                raise ValueError(f"resource release {option.id} uses an unknown action")
            if option.location_id not in location_ids:
                raise ValueError(f"resource release {option.id} uses an unknown location")
            if (
                option.character_id not in entity_ids
                or option.container_entity_id not in entity_ids
            ):
                raise ValueError(f"resource release {option.id} uses an unknown entity")
            if option.resource_id not in resource_ids:
                raise ValueError(f"resource release {option.id} uses an unknown resource")
            action = action_by_id[option.action_id]
            if action.intent != "negotiate" or action.target_type != "character":
                raise ValueError(
                    f"resource release {option.id} requires a character negotiation action"
                )
        for option in blueprint.resource_validation_options:
            if option.action_id not in action_ids:
                raise ValueError(f"resource validation {option.id} uses an unknown action")
            if option.location_id not in location_ids:
                raise ValueError(f"resource validation {option.id} uses an unknown location")
            if option.resource_id not in resource_ids:
                raise ValueError(f"resource validation {option.id} uses an unknown resource")
            action = action_by_id[option.action_id]
            if action.intent != "investigate" or action.target_type != "resource":
                raise ValueError(
                    f"resource validation {option.id} requires a resource investigation action"
                )
        progression_knowledge_ids = release_knowledge_ids | validation_knowledge_ids
        for option in blueprint.final_repair_options:
            if option.action_id not in action_ids:
                raise ValueError(f"final repair {option.id} uses an unknown action")
            if option.location_id not in location_ids or option.target_entity_id not in entity_ids:
                raise ValueError(f"final repair {option.id} uses an unknown target")
            if option.consumed_resource_id not in resource_ids:
                raise ValueError(f"final repair {option.id} uses an unknown resource")
            if not set(option.prerequisite_knowledge_ids).issubset(progression_knowledge_ids):
                raise ValueError(f"final repair {option.id} uses unknown prerequisite knowledge")
            if option.outcome_id not in outcome_ids:
                raise ValueError(f"final repair {option.id} uses an unknown outcome")
            outcome = next(item for item in blueprint.outcomes if item.id == option.outcome_id)
            if not set(outcome.required_committed_event_tags).issubset(option.committed_event_tags):
                raise ValueError(f"final repair {option.id} does not emit its outcome tags")
            action = action_by_id[option.action_id]
            if action.intent != "repair" or action.target_type != "entity":
                raise ValueError(f"final repair {option.id} requires an entity repair action")
        for option in blueprint.world_response_options:
            if option.action_id not in action_ids:
                raise ValueError(f"world response {option.id} uses an unknown action")
            if option.outcome_id not in outcome_ids:
                raise ValueError(f"world response {option.id} uses an unknown outcome")
            if option.commitment_id not in exchange_commitment_ids:
                raise ValueError(f"world response {option.id} uses an unknown commitment")
            if option.actor_entity_id not in entity_ids:
                raise ValueError(f"world response {option.id} uses an unknown actor")
            action = action_by_id[option.action_id]
            if action.intent != "observe" or action.target_type != "scene":
                raise ValueError(f"world response {option.id} requires a scene observation action")
        for objective in blueprint.onboarding:
            if objective.action_id not in action_ids:
                raise ValueError(f"onboarding objective {objective.id} uses an unknown action")
        for opportunity in blueprint.opportunities:
            if opportunity.location_id not in location_ids:
                raise ValueError(f"opportunity {opportunity.id} uses an unknown location")
            if opportunity.action_id not in action_ids:
                raise ValueError(f"opportunity {opportunity.id} uses an unknown action")
            completion_resource_ids = {
                *opportunity.completion_inventory_resource_ids,
                *opportunity.completion_discovered_resource_ids,
            }
            if not completion_resource_ids.issubset(resource_ids):
                raise ValueError(
                    f"opportunity {opportunity.id} uses an unknown completion resource"
                )
        for recovery in blueprint.recovery_paths:
            if recovery.target_location_id not in location_ids:
                raise ValueError(f"recovery path {recovery.id} uses an unknown location")
            if recovery.blocked_action_id not in action_ids:
                raise ValueError(f"recovery path {recovery.id} uses an unknown blocked action")
            if recovery.alternative_action_id not in action_ids:
                raise ValueError(f"recovery path {recovery.id} uses an unknown alternative")
        known_objects = {
            "location": location_ids,
            "character": entity_ids,
            "entity": entity_ids,
            "resource": resource_ids,
        }
        asset_subjects = location_ids | entity_ids | resource_ids
        for slot in blueprint.asset_slots:
            if slot.subject_id not in asset_subjects:
                raise ValueError(f"asset slot {slot.id} uses an unknown subject")
        for presentation in blueprint.presentations:
            if presentation.object_id not in known_objects[presentation.object_type]:
                raise ValueError(
                    f"presentation refers to unknown {presentation.object_type}: "
                    f"{presentation.object_id}"
                )
        if not set(blueprint.canon_text_keys).issubset(canon_ids):
            raise ValueError("canon localization refers to an unknown Canon fact")
        public_canon_ids = {
            fact.id
            for fact in self.initial_world_state.canon_facts
            if fact.visibility.value == "public"
        }
        if not public_canon_ids.issubset(blueprint.canon_text_keys):
            raise ValueError("every public Canon fact requires localized player text")
        for locale in blueprint.supported_locales:
            texts = self.localized_text.get(locale)
            if texts is None:
                raise ValueError(f"missing localization bundle: {locale}")
            for key in self.required_text_keys:
                if not texts.get(key):
                    raise ValueError(f"missing localized text {locale}:{key}")
        return self

    @property
    def required_text_keys(self) -> tuple[str, ...]:
        keys = {
            self.blueprint.world_name_key,
            self.blueprint.player_summary_key,
            *self.blueprint.journal_seed_keys,
        }
        for action in self.blueprint.actions:
            keys.add(action.label_key)
        for option in self.blueprint.dialogue_options:
            keys.update((option.label_key, option.response_key, option.knowledge_statement_key))
        for option in self.blueprint.resource_exchange_options:
            keys.update((option.label_key, option.response_key, option.commitment_description_key))
        for option in self.blueprint.repair_options:
            keys.update(
                (
                    option.label_key,
                    option.response_key,
                    option.knowledge_statement_key,
                )
            )
        for option in self.blueprint.commitment_breach_options:
            keys.update((option.label_key, option.response_key))
        for option in self.blueprint.resource_release_options:
            keys.update((option.label_key, option.response_key, option.knowledge_statement_key))
        for option in self.blueprint.resource_validation_options:
            keys.update((option.label_key, option.response_key, option.knowledge_statement_key))
        for option in self.blueprint.final_repair_options:
            keys.update((option.label_key, option.response_key, option.knowledge_statement_key))
        for option in self.blueprint.world_response_options:
            keys.update((option.label_key, option.response_key))
        for objective in self.blueprint.onboarding:
            keys.update((objective.title_key, objective.description_key))
        for presentation in self.blueprint.presentations:
            keys.update(
                (
                    presentation.name_key,
                    presentation.summary_key,
                    presentation.accessibility_label_key,
                )
            )
        keys.update(self.blueprint.canon_text_keys.values())
        for slot in self.blueprint.asset_slots:
            keys.add(slot.accessibility_label_key)
        for opportunity in self.blueprint.opportunities:
            keys.update((opportunity.title_key, opportunity.description_key))
        for recovery in self.blueprint.recovery_paths:
            keys.add(recovery.guidance_key)
        for outcome in self.blueprint.outcomes:
            keys.update((outcome.title_key, outcome.description_key))
        return tuple(sorted(keys))

    def text(self, key: str, locale: str) -> str:
        selected = locale if locale in self.localized_text else self.blueprint.default_locale
        return self.localized_text[selected][key]


class InstalledContentPackage(AethelisModel):
    content_version_id: Identifier
    package_id: Identifier
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    package: ProductContentPackage
    created_at: AwareDatetime


class AvailableWorldContent(AethelisModel):
    world_definition_id: Identifier
    world_name: str = Field(min_length=1)
    content_version_id: Identifier
    package_id: Identifier
    default_locale: str = Field(min_length=2, max_length=35)
    supported_locales: tuple[str, ...] = Field(min_length=1)


def product_content_hash(package: ProductContentPackage) -> str:
    encoded = json.dumps(
        package.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def product_content_hash_candidates(package: ProductContentPackage) -> tuple[str, ...]:
    """Return current and narrowly scoped compatibility hashes for published packages."""

    candidates = [product_content_hash(package)]
    progression_fields = (
        "resource_release_options",
        "resource_validation_options",
        "final_repair_options",
        "world_response_options",
    )
    if package.blueprint.content_version_id == "mistgate_product_v1_3_0":
        payload = package.model_dump(mode="json")
        _remove_opportunity_completion_fields(payload)
        _remove_living_world_state(payload)
        blueprint = payload.get("blueprint")
        policy = (
            blueprint.get("dialogue_expression_policy") if isinstance(blueprint, dict) else None
        )
        if isinstance(blueprint, dict) and isinstance(policy, dict):
            policy.pop("max_total_latency_ms", None)
            policy.pop("max_total_tokens", None)
            blueprint.pop("repair_options", None)
            blueprint.pop("commitment_breach_options", None)
            for field_name in progression_fields:
                blueprint.pop(field_name, None)
            _remove_player_world_responses(payload)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            candidates.append(sha256(encoded.encode("utf-8")).hexdigest())
    if package.blueprint.content_version_id == "mistgate_product_v1_4_0":
        payload = package.model_dump(mode="json")
        _remove_opportunity_completion_fields(payload)
        _remove_living_world_state(payload)
        blueprint = payload.get("blueprint")
        if isinstance(blueprint, dict):
            blueprint.pop("repair_options", None)
            blueprint.pop("commitment_breach_options", None)
            for field_name in progression_fields:
                blueprint.pop(field_name, None)
            _remove_player_world_responses(payload)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            candidates.append(sha256(encoded.encode("utf-8")).hexdigest())
    if package.blueprint.content_version_id == "mistgate_product_v1_5_0":
        payload = package.model_dump(mode="json")
        _remove_opportunity_completion_fields(payload)
        _remove_living_world_state(payload)
        blueprint = payload.get("blueprint")
        if isinstance(blueprint, dict):
            for field_name in progression_fields:
                blueprint.pop(field_name, None)
            _remove_player_world_responses(payload)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            candidates.append(sha256(encoded.encode("utf-8")).hexdigest())
    if package.blueprint.content_version_id == "mistgate_product_v1_6_0":
        payload = package.model_dump(mode="json")
        _remove_opportunity_completion_fields(payload)
        _remove_living_world_state(payload)
        blueprint = payload.get("blueprint")
        if isinstance(blueprint, dict):
            blueprint.pop("world_response_options", None)
            _remove_player_world_responses(payload)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            candidates.append(sha256(encoded.encode("utf-8")).hexdigest())
    if package.blueprint.content_version_id == "mistgate_product_v1_7_0":
        payload = package.model_dump(mode="json")
        _remove_opportunity_completion_fields(payload)
        _remove_living_world_state(payload)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        candidates.append(sha256(encoded.encode("utf-8")).hexdigest())
    if package.blueprint.content_version_id in {
        "mistgate_product_v1_8_0",
        "mistgate_product_v1_9_0",
    }:
        payload = package.model_dump(mode="json")
        _remove_opportunity_completion_fields(payload)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        candidates.append(sha256(encoded.encode("utf-8")).hexdigest())
    return tuple(dict.fromkeys(candidates))


def _remove_player_world_responses(payload: dict[str, object]) -> None:
    world = payload.get("initial_world_state")
    player = world.get("player") if isinstance(world, dict) else None
    if isinstance(player, dict):
        player.pop("world_responses", None)


def _remove_opportunity_completion_fields(payload: dict[str, object]) -> None:
    blueprint = payload.get("blueprint")
    opportunities = blueprint.get("opportunities") if isinstance(blueprint, dict) else None
    if not isinstance(opportunities, list):
        return
    for opportunity in opportunities:
        if not isinstance(opportunity, dict):
            continue
        for field_name in (
            "is_optional",
            "completion_knowledge_ids",
            "completion_inventory_resource_ids",
            "completion_discovered_resource_ids",
            "completion_target_tags",
        ):
            opportunity.pop(field_name, None)


def _remove_living_world_state(payload: dict[str, object]) -> None:
    world = payload.get("initial_world_state")
    if not isinstance(world, dict):
        return
    for field_name in (
        "clock",
        "agent_profiles",
        "agent_beliefs",
        "agent_memories",
        "agent_relationships",
        "agent_belief_candidates",
        "agent_claims",
        "world_activities",
    ):
        world.pop(field_name, None)


def installed_content_package(
    package: ProductContentPackage, *, created_at: datetime
) -> InstalledContentPackage:
    return InstalledContentPackage(
        content_version_id=package.blueprint.content_version_id,
        package_id=package.blueprint.package_id,
        content_hash=product_content_hash(package),
        package=package,
        created_at=created_at,
    )
