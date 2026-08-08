from __future__ import annotations

from collections.abc import Callable

from aethelis.product.content_contracts import ProductContentPackage, ProductRoute
from aethelis.product.contracts import (
    PrincipalContext,
    PrincipalStatus,
    WorldAccessLevel,
)
from aethelis.product.errors import (
    ProductAccessDeniedError,
    ProductConflictError,
    ProductNotFoundError,
)
from aethelis.product.ports import ProductUnitOfWork
from aethelis.product.projection_contracts import (
    ContextualActionView,
    DialogueExchangeView,
    DialogueInteractionView,
    JournalActivityView,
    JournalCommitmentView,
    JournalKnowledgeView,
    JournalOutcomeView,
    JournalRelationshipView,
    JournalResourceView,
    JournalView,
    JournalWorldResponseView,
    MapLocationView,
    MapView,
    OpportunityView,
    ResumeSummaryView,
    SceneView,
    SituationView,
    VisibleEntityView,
    VisibleResourceView,
)
from aethelis.schemas.world import CanonVisibility, WorldState


class ProjectionService:
    def __init__(
        self,
        uow_factory: Callable[[], ProductUnitOfWork],
    ) -> None:
        self._uow_factory = uow_factory

    def scene(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
        player_profile_id: str,
    ) -> SceneView:
        with self._uow_factory() as uow:
            instance, profile = self._authorize(
                uow, principal, world_instance_id, player_profile_id
            )
            snapshot = uow.worlds.get_snapshot(instance.current_snapshot_id)
            if snapshot is None:
                raise ProductConflictError(
                    "world_head_snapshot_missing", "World head snapshot is missing."
                )
            world = snapshot.world_state
            package = _content_package(uow, instance.content_version_id)
            location_id = world.player.current_location_id if world.player else None
            location = next((item for item in world.locations if item.id == location_id), None)
            visible_resources = tuple(
                VisibleResourceView(
                    id=item.id,
                    name=item.name,
                    summary=item.summary,
                    quantity=item.quantity,
                )
                for item in world.resources
                if item.location_id == location_id
                and profile.id in item.discovery_state.discovered_by_agent_ids
            )
            visible_entities = tuple(
                VisibleEntityView(
                    id=item.id,
                    name=item.name,
                    summary=item.summary,
                    status=item.status.value,
                )
                for item in world.entities
                if item.location_id == location_id
            )
            has_undiscovered = any(
                item.location_id == location_id
                and profile.id not in item.discovery_state.discovered_by_agent_ids
                for item in world.resources
            )
            actions = _scene_actions(
                package=package,
                locale=profile.locale,
                world=world,
                location_id=location_id,
                has_undiscovered=has_undiscovered,
                visible_resources=visible_resources,
            )
            published_versions = tuple(
                version
                for version in uow.catalog.list_published_content_versions()
                if version.world_definition_id == instance.world_definition_id
            )
            recommended_version = (
                max(
                    published_versions,
                    key=lambda version: (version.published_at or version.created_at, version.id),
                ).id
                if published_versions
                else instance.content_version_id
            )
            action_ids = (
                {action.action_id for action in package.blueprint.actions} if package else set()
            )
            return SceneView(
                world_instance_id=instance.id,
                world_version=instance.current_world_version,
                world_turn=world.clock.turn,
                elapsed_minutes=world.clock.elapsed_minutes,
                location_id=location_id,
                location_name=location.name if location else None,
                visible_entities=visible_entities,
                visible_resources=visible_resources,
                public_facts=_public_facts(package, world, profile.locale),
                contextual_actions=actions,
                content_version_id=instance.content_version_id,
                supports_free_dialogue=bool(world.agent_profiles and "ask_character" in action_ids),
                supports_world_narrative="ask_world" in action_ids,
                recommended_content_version_id=(
                    recommended_version
                    if recommended_version != instance.content_version_id
                    else None
                ),
            )

    def map(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
        player_profile_id: str,
    ) -> MapView:
        with self._uow_factory() as uow:
            instance, profile = self._authorize(
                uow, principal, world_instance_id, player_profile_id
            )
            snapshot = uow.worlds.get_snapshot(instance.current_snapshot_id)
            package = _content_package(uow, instance.content_version_id)
            if snapshot is None or package is None:
                raise ProductConflictError(
                    "world_map_boundary_missing",
                    "World map requires a complete product content package.",
                )
            world = snapshot.world_state
            current = world.player.current_location_id if world.player else None
            known_ids = {
                location_id
                for route in package.blueprint.routes
                if route.initially_known
                for location_id in (route.from_location_id, route.to_location_id)
            }
            reachable_ids = set(_reachable_location_ids(package, current))
            presentations = {
                (item.object_type, item.object_id): item for item in package.blueprint.presentations
            }
            locations = []
            for location in world.locations:
                if location.id not in known_ids and location.id != current:
                    continue
                presentation = presentations.get(("location", location.id))
                accessibility = (
                    package.text(presentation.accessibility_label_key, profile.locale)
                    if presentation
                    else location.summary
                )
                locations.append(
                    MapLocationView(
                        id=location.id,
                        name=location.name,
                        summary=location.summary,
                        accessibility_label=accessibility,
                        is_current=location.id == current,
                        is_reachable=location.id in reachable_ids,
                    )
                )
            return MapView(
                world_instance_id=instance.id,
                world_version=instance.current_world_version,
                current_location_id=current,
                locations=tuple(locations),
            )

    def journal(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
        player_profile_id: str,
    ) -> JournalView:
        with self._uow_factory() as uow:
            instance, profile = self._authorize(
                uow, principal, world_instance_id, player_profile_id
            )
            snapshot = uow.worlds.get_snapshot(instance.current_snapshot_id)
            package = _content_package(uow, instance.content_version_id)
            if snapshot is None or package is None:
                raise ProductConflictError(
                    "world_journal_boundary_missing",
                    "World journal requires a complete product content package.",
                )
            world = snapshot.world_state
            observations = tuple(
                f"已发现：{resource.name}"
                for resource in world.resources
                if profile.id in resource.discovery_state.discovered_by_agent_ids
            )
            location_names = {location.id: location.name for location in world.locations}
            entity_names = {entity.id: entity.name for entity in world.entities}
            resources = tuple(
                JournalResourceView(
                    id=resource.id,
                    name=resource.name,
                    summary=resource.summary,
                    kind=resource.kind.value,
                    quantity=resource.quantity,
                    custody_label=_resource_custody_label(
                        resource,
                        location_names=location_names,
                        entity_names=entity_names,
                    ),
                    source_resource_id=resource.id,
                )
                for resource in world.resources
                if profile.id in resource.discovery_state.discovered_by_agent_ids
            )
            resource_by_id = {resource.id: resource for resource in world.resources}
            player_inventory = world.player.inventory if world.player else ()
            inventory_resources = tuple(
                JournalResourceView(
                    id=item.id,
                    name=resource_by_id[item.resource_id].name,
                    summary=resource_by_id[item.resource_id].summary,
                    kind=resource_by_id[item.resource_id].kind.value,
                    quantity=item.quantity,
                    custody_label="由你持有",
                    source_resource_id=item.resource_id,
                    is_player_owned=True,
                )
                for item in player_inventory
                if item.resource_id in resource_by_id
            )
            opportunities = _opportunity_views(
                package,
                world,
                profile.id,
                profile.locale,
                location_names,
            )
            player_knowledge = world.player.knowledge if world.player else ()
            knowledge = tuple(
                JournalKnowledgeView(
                    id=record.id,
                    kind=record.kind.value,
                    statement=record.statement,
                    source_name=entity_names.get(record.source_entity_id, record.source_entity_id),
                    confidence=record.confidence,
                )
                for record in player_knowledge
            )
            player_relationships = world.player.relationships if world.player else ()
            relationships = tuple(
                JournalRelationshipView(
                    character_id=record.character_id,
                    character_name=entity_names.get(record.character_id, record.character_id),
                    trust=record.trust,
                    standing_label=_relationship_label(record.trust),
                    interaction_count=record.interaction_count,
                )
                for record in player_relationships
            )
            player_commitments = world.player.commitments if world.player else ()
            commitments = tuple(
                JournalCommitmentView(
                    id=record.id,
                    counterparty_name=entity_names.get(
                        record.counterparty_entity_id,
                        record.counterparty_entity_id,
                    ),
                    description=record.description,
                    status=record.status.value,
                )
                for record in player_commitments
            )
            outcomes = _achieved_outcomes(package, world, profile.locale)
            player_world_responses = world.player.world_responses if world.player else ()
            world_responses = tuple(
                JournalWorldResponseView(
                    id=record.id,
                    actor_name=entity_names.get(record.actor_entity_id, record.actor_entity_id),
                    response_kind=record.response_kind,
                    summary=record.summary,
                )
                for record in player_world_responses
            )
            world_activities = tuple(
                JournalActivityView(
                    id=record.id,
                    turn=record.turn,
                    actor_names=tuple(
                        entity_names.get(actor_id, actor_id)
                        for actor_id in record.actor_agent_ids
                    ),
                    activity_kind=record.activity_kind,
                    summary=record.summary,
                )
                for record in world.world_activities[-12:]
            )
            dialogue_interactions = _dialogue_interactions(
                package,
                world.player.dialogue_history if world.player else (),
                entity_names,
                profile.locale,
            )
            return JournalView(
                world_instance_id=instance.id,
                world_version=instance.current_world_version,
                entries=tuple(
                    package.text(key, profile.locale) for key in package.blueprint.journal_seed_keys
                ),
                confirmed_facts=_public_facts(package, world, profile.locale),
                observations=observations,
                current_objectives=_current_objectives(
                    package, world, profile.locale, opportunities
                ),
                resources=(*inventory_resources, *resources),
                opportunities=opportunities,
                situation=_situation_view(package, world, profile.locale),
                knowledge=knowledge,
                relationships=relationships,
                commitments=commitments,
                outcomes=outcomes,
                world_responses=world_responses,
                world_activities=world_activities,
                dialogue_interactions=dialogue_interactions,
            )

    def resume_summary(
        self,
        *,
        principal: PrincipalContext,
        world_instance_id: str,
        player_profile_id: str,
    ) -> ResumeSummaryView:
        with self._uow_factory() as uow:
            instance, _profile = self._authorize(
                uow, principal, world_instance_id, player_profile_id
            )
            snapshot = uow.worlds.get_snapshot(instance.current_snapshot_id)
            save = uow.saves.get_latest(instance.id)
            if snapshot is None or save is None:
                raise ProductConflictError(
                    "world_resume_boundary_missing", "World resume boundary is incomplete."
                )
            world = snapshot.world_state
            location_id = world.player.current_location_id if world.player else None
            location = next((item for item in world.locations if item.id == location_id), None)
            session = uow.sessions.find_resumable(instance.id, player_profile_id)
            visible_count = sum(
                player_profile_id in item.discovery_state.discovered_by_agent_ids
                for item in world.resources
            )
            return ResumeSummaryView(
                world_instance_id=instance.id,
                world_version=instance.current_world_version,
                world_name=world.name,
                location_name=location.name if location else None,
                last_save_reason=save.reason.value,
                visible_resource_count=visible_count,
                resumable_session_id=session.id if session else None,
            )

    @staticmethod
    def _authorize(uow, principal, world_instance_id, player_profile_id):
        stored = uow.identities.get_principal(principal.principal_id)
        if stored is None or stored.status != PrincipalStatus.ACTIVE:
            raise ProductAccessDeniedError(
                "principal_not_active", "Authenticated principal is not active."
            )
        profile = uow.identities.get_profile(player_profile_id)
        if profile is None or profile.principal_id != principal.principal_id:
            raise ProductAccessDeniedError(
                "player_profile_forbidden", "Player profile is not authorized."
            )
        instance = uow.worlds.get_instance(world_instance_id)
        if instance is None:
            raise ProductNotFoundError("world_instance_not_found", "World instance was not found.")
        grant = uow.access.get_grant(principal.principal_id, world_instance_id)
        if grant is None or grant.access_level not in {
            WorldAccessLevel.PLAY,
            WorldAccessLevel.MANAGE,
        }:
            raise ProductAccessDeniedError(
                "world_access_forbidden", "Principal cannot view this world."
            )
        return instance, profile


def _content_package(uow, content_version_id: str) -> ProductContentPackage | None:
    record = uow.catalog.get_content_package(content_version_id)
    return record.package if record else None


def _dialogue_interactions(
    package: ProductContentPackage,
    turns,
    entity_names: dict[str, str],
    locale: str,
) -> tuple[DialogueInteractionView, ...]:
    labels = _dialogue_option_labels(package, locale)
    grouped: dict[str, list] = {}
    order: list[str] = []
    for turn in turns:
        interaction_id = turn.interaction_id or f"legacy_{turn.id}"
        if interaction_id not in grouped:
            grouped[interaction_id] = []
            order.append(interaction_id)
        grouped[interaction_id].append(turn)

    interactions = []
    for interaction_id in order[-40:]:
        interaction_turns = grouped[interaction_id]
        first = interaction_turns[0]
        target_name = (
            entity_names.get(first.character_id, first.character_id)
            if first.character_id
            else "世界旁白"
        )
        exchanges = tuple(
            DialogueExchangeView(
                id=turn.id,
                input_kind="free" if turn.player_utterance else "preset",
                player_text=(
                    turn.player_utterance
                    or labels.get(turn.dialogue_option_id, "选择了一个预设回应")
                ),
                response_text=turn.utterance,
                requested_effect_status=turn.requested_effect_status.value,
                visible_effects=_dialogue_visible_effects(turn),
                committed_event_id=turn.committed_event_id,
            )
            for turn in interaction_turns
        )
        interactions.append(
            DialogueInteractionView(
                id=interaction_id,
                target_kind=first.target_kind.value,
                target_id=first.character_id,
                target_name=target_name,
                contains_free_expression=any(turn.player_utterance for turn in interaction_turns),
                exchanges=exchanges,
            )
        )
    return tuple(interactions)


def _dialogue_option_labels(
    package: ProductContentPackage,
    locale: str,
) -> dict[str, str]:
    collections = (
        package.blueprint.dialogue_options,
        package.blueprint.resource_exchange_options,
        package.blueprint.repair_options,
        package.blueprint.commitment_breach_options,
        package.blueprint.resource_release_options,
        package.blueprint.resource_validation_options,
        package.blueprint.final_repair_options,
        package.blueprint.world_response_options,
    )
    return {
        option.id: package.text(option.label_key, locale)
        for options in collections
        for option in options
    }


def _dialogue_visible_effects(turn) -> tuple[str, ...]:
    effects = []
    if turn.knowledge_record_ids:
        effects.append("已记录玩家可见知识")
    if turn.belief_candidate_ids:
        effects.append("角色将这句话保留为待验证认知")
    if turn.requested_effect_status.value == "committed":
        effects.append("请求的世界效果已经提交")
    elif turn.requested_effect_status.value == "rejected":
        effects.append("请求的世界效果被拒绝")
    elif turn.requested_effect_status.value == "needs_clarification":
        effects.append("请求的世界效果仍需明确目标或行动")
    return tuple(effects)


def _opportunity_views(
    package: ProductContentPackage,
    world: WorldState,
    player_id: str,
    locale: str,
    location_names: dict[str, str],
) -> tuple[OpportunityView, ...]:
    current_location_id = world.player.current_location_id if world.player else None
    knowledge_ids = {item.id for item in world.player.knowledge} if world.player else set()
    inventory_ids = {item.resource_id for item in world.player.inventory} if world.player else set()
    discovered_ids = {
        item.id
        for item in world.resources
        if player_id in item.discovery_state.discovered_by_agent_ids
    }
    world_tags = {tag for entity in world.entities for tag in entity.tags}
    views = []
    for opportunity in package.blueprint.opportunities:
        evidence_checks = (
            bool(set(opportunity.completion_knowledge_ids) & knowledge_ids),
            bool(set(opportunity.completion_inventory_resource_ids) & inventory_ids),
            bool(set(opportunity.completion_discovered_resource_ids) & discovered_ids),
            bool(set(opportunity.completion_target_tags) & world_tags),
        )
        views.append(
            OpportunityView(
                id=opportunity.id,
                title=package.text(opportunity.title_key, locale),
                description=package.text(opportunity.description_key, locale),
                location_id=opportunity.location_id,
                location_name=location_names[opportunity.location_id],
                action_id=opportunity.action_id,
                target_id=opportunity.target_id,
                is_at_location=opportunity.location_id == current_location_id,
                is_completed=any(evidence_checks),
                is_optional=opportunity.is_optional,
            )
        )
    return tuple(views)


def _current_objectives(
    package: ProductContentPackage,
    world: WorldState,
    locale: str,
    opportunities: tuple[OpportunityView, ...],
) -> tuple[str, ...]:
    facts = _mistgate_progress_facts(world)
    action_labels = {
        action.action_id: package.text(action.label_key, locale)
        for action in package.blueprint.actions
    }
    if facts["repaired"]:
        if (
            world.player
            and not world.player.world_responses
            and package.blueprint.world_response_options
        ):
            return (action_labels["wait_for_world_response"],)
        ending = next(
            item for item in package.blueprint.outcomes if item.id == "outcome_regulator_stabilized"
        )
        return (package.text(ending.title_key, locale),)

    objectives = [
        item.title
        for item in opportunities
        if not item.is_completed and not item.is_optional
    ]
    if facts["parts_secured"] and not facts["contained"]:
        objectives.append(action_labels["repair_regulator"])
    if facts["contained"] and facts["key_secured"] and facts["lens_validated"]:
        objectives.append(action_labels["stabilize_regulator"])
    return tuple(dict.fromkeys(objectives))


def _situation_view(
    package: ProductContentPackage,
    world: WorldState,
    locale: str,
) -> SituationView:
    facts = _mistgate_progress_facts(world)
    if facts["repaired"]:
        outcome = next(
            item for item in package.blueprint.outcomes if item.id == "outcome_regulator_stabilized"
        )
        phase = "repaired"
        title = package.text(outcome.title_key, locale)
        summary = package.text(outcome.description_key, locale)
    elif facts["contained"]:
        outcome = next(
            item for item in package.blueprint.outcomes if item.id == "outcome_city_holds"
        )
        phase = "contained"
        title = package.text(outcome.title_key, locale)
        summary = package.text(outcome.description_key, locale)
    else:
        opening = package.blueprint.onboarding[0]
        phase = "unstable"
        title = package.text(opening.title_key, locale)
        summary = package.text(opening.description_key, locale)

    recovery_ids = []
    if not facts["parts_secured"]:
        recovery_ids.append("recovery_missing_parts")
    elif not facts["contained"]:
        recovery_ids.append("recovery_repair_access")
    if facts["contained"] and not facts["key_secured"]:
        recovery_ids.append("recovery_missing_key")
    if facts["contained"] and not facts["lens_validated"]:
        recovery_ids.append("recovery_missing_lens")
    recovery_by_id = {item.id: item for item in package.blueprint.recovery_paths}
    guidance = tuple(
        package.text(recovery_by_id[item].guidance_key, locale)
        for item in recovery_ids
        if item in recovery_by_id
    )
    completed_steps = sum(
        bool(facts[key])
        for key in ("parts_secured", "key_secured", "lens_validated", "repaired")
    )
    return SituationView(
        phase=phase,
        title=title,
        summary=summary,
        completed_steps=completed_steps,
        total_steps=4,
        recovery_guidance=guidance,
    )


def _mistgate_progress_facts(world: WorldState) -> dict[str, bool]:
    player = world.player
    inventory_ids = {item.resource_id for item in player.inventory} if player else set()
    knowledge_ids = {item.id for item in player.knowledge} if player else set()
    regulator = next((item for item in world.entities if item.id == "dawn_regulator"), None)
    target_tags = set(regulator.tags) if regulator else set()
    repaired = "regulator_repaired" in target_tags
    return {
        "parts_secured": bool(
            "stabilizer_parts" in inventory_ids
            or "repair_progressed" in target_tags
            or repaired
        ),
        "key_secured": bool(
            "calibration_key" in inventory_ids
            or "knowledge_calibration_key_secured" in knowledge_ids
            or repaired
        ),
        "lens_validated": bool(
            "knowledge_gate_lens_validated" in knowledge_ids or repaired
        ),
        "contained": bool("repair_progressed" in target_tags or repaired),
        "repaired": repaired,
    }


def _public_facts(package: ProductContentPackage | None, world, locale: str) -> tuple[str, ...]:
    return tuple(
        (
            package.text(package.blueprint.canon_text_keys[fact.id], locale)
            if package is not None and fact.id in package.blueprint.canon_text_keys
            else fact.statement
        )
        for fact in world.canon_facts
        if fact.visibility == CanonVisibility.PUBLIC
    )


def _resource_custody_label(resource, *, location_names, entity_names) -> str:
    if resource.owner_agent_id:
        owner_name = entity_names.get(resource.owner_agent_id, resource.owner_agent_id)
        return f"由{owner_name}持有"
    if resource.owner_entity_id:
        owner_name = entity_names.get(resource.owner_entity_id, resource.owner_entity_id)
        return f"由{owner_name}保管"
    if resource.location_id:
        return f"位于{location_names.get(resource.location_id, resource.location_id)}"
    return "保管位置未知"


def _scene_actions(
    *,
    package: ProductContentPackage | None,
    locale: str,
    world,
    location_id: str | None,
    has_undiscovered: bool,
    visible_resources: tuple[VisibleResourceView, ...] = (),
) -> tuple[ContextualActionView, ...]:
    if package is None or location_id is None:
        return (
            (
                ContextualActionView(
                    action_id="investigate_area",
                    label="Investigate the area",
                    location_id=location_id,
                ),
            )
            if has_undiscovered
            else ()
        )
    action_labels = {
        action.action_id: package.text(action.label_key, locale)
        for action in package.blueprint.actions
    }
    actions = [
        ContextualActionView(
            action_id="observe_scene",
            label=action_labels["observe_scene"],
            location_id=location_id,
            command_required=False,
        )
    ]
    if "advance_world" in action_labels:
        actions.append(
            ContextualActionView(
                action_id="advance_world",
                label=action_labels["advance_world"],
                location_id=location_id,
            )
        )
    location_names = {location.id: location.name for location in world.locations}
    for destination_id in _reachable_location_ids(package, location_id):
        actions.append(
            ContextualActionView(
                action_id="move_to_location",
                label=f"{action_labels['move_to_location']}：{location_names[destination_id]}",
                location_id=location_id,
                target_id=destination_id,
            )
        )
    if has_undiscovered:
        actions.append(
            ContextualActionView(
                action_id="investigate_area",
                label=action_labels["investigate_area"],
                location_id=location_id,
            )
        )
    for resource in visible_resources:
        actions.append(
            ContextualActionView(
                action_id="inspect_resource",
                label=f"{action_labels['inspect_resource']}：{resource.name}",
                location_id=location_id,
                target_id=resource.id,
            )
        )
    known_knowledge_ids = (
        {record.id for record in world.player.knowledge} if world.player else set()
    )
    present_character_ids = {
        entity.id
        for entity in world.entities
        if entity.location_id == location_id and "character" in entity.tags
    }
    for option in package.blueprint.dialogue_options:
        if (
            option.location_id == location_id
            and option.character_id in present_character_ids
            and option.knowledge_id not in known_knowledge_ids
        ):
            actions.append(
                ContextualActionView(
                    action_id=option.action_id,
                    label=package.text(option.label_key, locale),
                    location_id=location_id,
                    target_id=option.character_id,
                )
            )
    relationships = (
        {record.character_id: record for record in world.player.relationships}
        if world.player
        else {}
    )
    commitment_ids = {record.id for record in world.player.commitments} if world.player else set()
    commitments_by_id = (
        {record.id: record for record in world.player.commitments} if world.player else {}
    )
    inventory_resource_ids = (
        {record.resource_id for record in world.player.inventory} if world.player else set()
    )
    inventory_quantities = (
        {record.resource_id: record.quantity for record in world.player.inventory}
        if world.player
        else {}
    )
    resources_by_id = {resource.id: resource for resource in world.resources}
    for option in package.blueprint.resource_exchange_options:
        resource = resources_by_id.get(option.resource_id)
        relationship = relationships.get(option.character_id)
        if (
            option.location_id == location_id
            and option.character_id in present_character_ids
            and set(option.prerequisite_knowledge_ids).issubset(known_knowledge_ids)
            and relationship is not None
            and relationship.trust >= option.minimum_trust
            and resource is not None
            and resource.location_id == location_id
            and resource.quantity >= option.quantity
            and option.commitment_id not in commitment_ids
            and option.resource_id not in inventory_resource_ids
        ):
            actions.append(
                ContextualActionView(
                    action_id=option.action_id,
                    label=package.text(option.label_key, locale),
                    location_id=location_id,
                    target_id=option.character_id,
                )
            )
    entities_by_id = {entity.id: entity for entity in world.entities}
    for option in package.blueprint.repair_options:
        target = entities_by_id.get(option.target_entity_id)
        commitment = commitments_by_id.get(option.commitment_id)
        target_tags = set(target.tags) if target is not None else set()
        if (
            option.location_id == location_id
            and target is not None
            and target.location_id == location_id
            and inventory_quantities.get(option.resource_id, 0) >= option.quantity
            and commitment is not None
            and commitment.status.value in {"active", "broken"}
            and set(option.required_target_tags).issubset(target_tags)
            and not set(option.result_target_tags).issubset(target_tags)
        ):
            actions.append(
                ContextualActionView(
                    action_id=option.action_id,
                    label=package.text(option.label_key, locale),
                    location_id=location_id,
                    target_id=option.target_entity_id,
                )
            )
    for option in package.blueprint.commitment_breach_options:
        commitment = commitments_by_id.get(option.commitment_id)
        if (
            option.location_id == location_id
            and option.character_id in present_character_ids
            and commitment is not None
            and commitment.status.value == "active"
        ):
            actions.append(
                ContextualActionView(
                    action_id=option.action_id,
                    label=package.text(option.label_key, locale),
                    location_id=location_id,
                    target_id=option.character_id,
                )
            )
    player_id = world.player.id if world.player else None
    for option in package.blueprint.resource_release_options:
        resource = resources_by_id.get(option.resource_id)
        container = entities_by_id.get(option.container_entity_id)
        discovered = bool(
            resource
            and (
                not option.required_discovery
                or player_id in resource.discovery_state.discovered_by_agent_ids
            )
        )
        if (
            option.location_id == location_id
            and option.character_id in present_character_ids
            and container is not None
            and container.location_id == location_id
            and resource is not None
            and resource.quantity >= option.quantity
            and discovered
            and option.resource_id not in inventory_resource_ids
            and option.knowledge_id not in known_knowledge_ids
        ):
            actions.append(
                ContextualActionView(
                    action_id=option.action_id,
                    label=package.text(option.label_key, locale),
                    location_id=location_id,
                    target_id=option.character_id,
                )
            )
    for option in package.blueprint.resource_validation_options:
        resource = resources_by_id.get(option.resource_id)
        discovered = bool(
            resource
            and (
                not option.required_discovery
                or player_id in resource.discovery_state.discovered_by_agent_ids
            )
        )
        if (
            option.location_id == location_id
            and resource is not None
            and resource.location_id == location_id
            and resource.quantity > 0
            and discovered
            and option.knowledge_id not in known_knowledge_ids
        ):
            actions.append(
                ContextualActionView(
                    action_id=option.action_id,
                    label=package.text(option.label_key, locale),
                    location_id=location_id,
                    target_id=option.resource_id,
                )
            )
    for option in package.blueprint.final_repair_options:
        target = entities_by_id.get(option.target_entity_id)
        target_tags = set(target.tags) if target is not None else set()
        if (
            option.location_id == location_id
            and target is not None
            and target.location_id == location_id
            and inventory_quantities.get(option.consumed_resource_id, 0) >= option.quantity
            and set(option.prerequisite_knowledge_ids).issubset(known_knowledge_ids)
            and set(option.required_target_tags).issubset(target_tags)
            and not set(option.result_target_tags).issubset(target_tags)
        ):
            actions.append(
                ContextualActionView(
                    action_id=option.action_id,
                    label=package.text(option.label_key, locale),
                    location_id=location_id,
                    target_id=option.target_entity_id,
                )
            )
    achieved_outcome_ids = {outcome.id for outcome in _achieved_outcomes(package, world, locale)}
    world_response_option_ids = (
        {record.response_option_id for record in world.player.world_responses}
        if world.player
        else set()
    )
    for option in package.blueprint.world_response_options:
        commitment = commitments_by_id.get(option.commitment_id)
        if (
            option.outcome_id in achieved_outcome_ids
            and commitment is not None
            and commitment.status.value == option.commitment_status
            and option.id not in world_response_option_ids
        ):
            actions.append(
                ContextualActionView(
                    action_id=option.action_id,
                    label=package.text(option.label_key, locale),
                    location_id=location_id,
                )
            )
    return tuple(actions)


def _achieved_outcomes(
    package: ProductContentPackage,
    world,
    locale: str,
) -> tuple[JournalOutcomeView, ...]:
    persisted_tags = {tag for entity in world.entities for tag in entity.tags}
    return tuple(
        JournalOutcomeView(
            id=outcome.id,
            outcome_type=outcome.outcome_type,
            title=package.text(outcome.title_key, locale),
            description=package.text(outcome.description_key, locale),
        )
        for outcome in package.blueprint.outcomes
        if set(outcome.required_committed_event_tags).issubset(persisted_tags)
    )


def _relationship_label(trust: int) -> str:
    if trust >= 4:
        return "深度信任"
    if trust >= 2:
        return "逐渐信任"
    if trust >= 1:
        return "初步信任"
    if trust <= -3:
        return "明显敌对"
    if trust <= -1:
        return "有所戒备"
    return "尚未建立关系"


def _reachable_location_ids(
    package: ProductContentPackage,
    location_id: str | None,
) -> tuple[str, ...]:
    if location_id is None:
        return ()
    destinations = []
    for route in package.blueprint.routes:
        if not route.initially_known or route.required_access_tags:
            continue
        destination = _route_destination(route, location_id)
        if destination is not None:
            destinations.append(destination)
    return tuple(sorted(set(destinations)))


def _route_destination(route: ProductRoute, location_id: str) -> str | None:
    if route.from_location_id == location_id:
        return route.to_location_id
    if route.bidirectional and route.to_location_id == location_id:
        return route.from_location_id
    return None
