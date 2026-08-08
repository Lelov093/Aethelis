from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from aethelis.schemas.seed import SeedBundle

EXPECTED_SCHEMA_VERSION = "0.1"
PROHIBITED_CANON_TAGS = {"belief", "rumor", "secret", "rejected_claim", "private_belief"}
PRODUCT_REQUIRED_LOCATION_IDS = {
    "central_archive",
    "council_square",
    "market_row",
    "workshop_lane",
    "old_aqueduct",
}
PRODUCT_REQUIRED_AGENT_NAMES = {
    "mira": "Mira Vale",
    "rowan": "Rowan Kest",
    "selka": "Selka Orin",
    "ivo": "Ivo Renn",
    "nara": "Nara Vey",
    "taren": "Taren Sol",
}
MISTGATE_SEED_IDS = {"mistgate_v01", "mistgate_v01_civic_pressure_variant"}


@dataclass(frozen=True)
class SeedValidationError:
    file: str
    object_id: str | None
    field_path: str
    error_type: str
    message: str

    def safe_dict(self) -> dict[str, str | None]:
        return {
            "file": self.file,
            "object_id": self.object_id,
            "field_path": self.field_path,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class SeedValidationReport:
    seed_path: Path
    schema_version: str | None
    loaded_files: tuple[str, ...]
    object_counts: dict[str, int]
    success: bool
    errors: tuple[SeedValidationError, ...]

    def safe_dict(self) -> dict[str, object]:
        return {
            "seed_path": str(self.seed_path),
            "schema_version": self.schema_version,
            "loaded_files": list(self.loaded_files),
            "object_counts": self.object_counts,
            "success": self.success,
            "error_count": len(self.errors),
            "errors": [error.safe_dict() for error in self.errors],
        }


class SeedValidator:
    def validate(
        self,
        seed_path: Path,
        bundle: SeedBundle | None,
        load_errors: Iterable[SeedValidationError] = (),
        loaded_files: Iterable[str] = (),
    ) -> SeedValidationReport:
        errors = list(load_errors)
        if bundle is None:
            return SeedValidationReport(
                seed_path=seed_path,
                schema_version=None,
                loaded_files=tuple(loaded_files),
                object_counts={},
                success=False,
                errors=tuple(errors),
            )

        errors.extend(_validate_schema_versions(bundle))
        errors.extend(_validate_product_alignment(bundle))
        errors.extend(_validate_unique_ids(bundle))
        errors.extend(_validate_cross_references(bundle))
        errors.extend(_validate_canon_boundaries(bundle))

        return SeedValidationReport(
            seed_path=seed_path,
            schema_version=bundle.manifest.schema_version,
            loaded_files=tuple(loaded_files),
            object_counts=_object_counts(bundle),
            success=not errors,
            errors=tuple(errors),
        )


def _validate_schema_versions(bundle: SeedBundle) -> list[SeedValidationError]:
    errors: list[SeedValidationError] = []
    versions = {
        "seed_manifest.yaml": bundle.manifest.schema_version,
        "world.yaml": bundle.world.schema_version,
        "agents.yaml": bundle.agents.schema_version,
        "beliefs.yaml": bundle.beliefs.schema_version,
        "memories.yaml": bundle.memories.schema_version,
    }
    if bundle.metadata is not None:
        versions["metadata.yaml"] = bundle.metadata.schema_version
    for file, version in versions.items():
        if version != EXPECTED_SCHEMA_VERSION:
            errors.append(
                SeedValidationError(
                    file=file,
                    object_id=None,
                    field_path="schema_version",
                    error_type="schema_version_mismatch",
                    message=(f"Expected schema_version {EXPECTED_SCHEMA_VERSION}, got {version}."),
                )
            )
    return errors


def _validate_product_alignment(bundle: SeedBundle) -> list[SeedValidationError]:
    errors: list[SeedValidationError] = []
    location_ids = {item.id for item in bundle.world.locations}
    agent_names = {item.id: item.name for item in bundle.agents.agents}
    agent_ids = set(agent_names)

    if "player" in agent_ids:
        errors.append(
            SeedValidationError(
                file="agents.yaml",
                object_id="player",
                field_path="agents",
                error_type="player_must_not_be_seed_agent",
                message="player is a special actor and must not be a seed Agent.",
            )
        )
    if bundle.manifest.seed_id not in MISTGATE_SEED_IDS:
        return errors

    for location_id in sorted(PRODUCT_REQUIRED_LOCATION_IDS - location_ids):
        errors.append(
            SeedValidationError(
                file="world.yaml",
                object_id=location_id,
                field_path="locations",
                error_type="missing_product_location",
                message=f"Missing Product-aligned Mistgate location: {location_id}",
            )
        )
    if "guard_tower" in location_ids:
        errors.append(
            SeedValidationError(
                file="world.yaml",
                object_id="guard_tower",
                field_path="locations",
                error_type="guard_tower_must_be_facility",
                message="guard_tower must be modeled as a council_square facility, not a Location.",
            )
        )
    for agent_id, expected_name in PRODUCT_REQUIRED_AGENT_NAMES.items():
        if agent_id not in agent_names:
            errors.append(
                SeedValidationError(
                    file="agents.yaml",
                    object_id=agent_id,
                    field_path="agents",
                    error_type="missing_product_agent",
                    message=f"Missing Product-aligned Mistgate agent: {expected_name}",
                )
            )
        elif agent_names[agent_id] != expected_name:
            errors.append(
                SeedValidationError(
                    file="agents.yaml",
                    object_id=agent_id,
                    field_path="agents.name",
                    error_type="product_agent_name_mismatch",
                    message=f"Expected agent name {expected_name}, got {agent_names[agent_id]}",
                )
            )
    return errors


def _validate_unique_ids(bundle: SeedBundle) -> list[SeedValidationError]:
    errors: list[SeedValidationError] = []
    collections = {
        "world.yaml:locations": [item.id for item in bundle.world.locations],
        "world.yaml:factions": [item.id for item in bundle.world.factions],
        "world.yaml:entities": [item.id for item in bundle.world.entities],
        "world.yaml:resources": [item.id for item in bundle.world.resources],
        "world.yaml:canon_facts": [item.id for item in bundle.world.canon_facts],
        "agents.yaml:agents": [item.id for item in bundle.agents.agents],
        "agents.yaml:relationships": [item.id for item in bundle.agents.relationships],
        "beliefs.yaml:beliefs": [item.id for item in bundle.beliefs.beliefs],
        "beliefs.yaml:secrets": [item.id for item in bundle.beliefs.secrets],
        "memories.yaml:memories": [item.id for item in bundle.memories.memories],
    }
    if bundle.metadata is not None:
        collections.update(
            {
                "metadata.yaml:public_facts": [item.id for item in bundle.metadata.public_facts],
                "metadata.yaml:rumors": [item.id for item in bundle.metadata.rumors],
                "metadata.yaml:pressure_seeds": [
                    item.id for item in bundle.metadata.pressure_seeds
                ],
                "metadata.yaml:action_metadata": [
                    item.id for item in bundle.metadata.action_metadata
                ],
            }
        )
    for collection, ids in collections.items():
        file, field = collection.split(":")
        for duplicate_id in _duplicates(ids):
            errors.append(
                SeedValidationError(
                    file=file,
                    object_id=duplicate_id,
                    field_path=field,
                    error_type="duplicate_id",
                    message=f"Duplicate id in {field}: {duplicate_id}",
                )
            )
    return errors


def _validate_cross_references(bundle: SeedBundle) -> list[SeedValidationError]:
    errors: list[SeedValidationError] = []
    location_ids = {item.id for item in bundle.world.locations}
    faction_ids = {item.id for item in bundle.world.factions}
    entity_ids = {item.id for item in bundle.world.entities}
    resource_ids = {item.id for item in bundle.world.resources}
    canon_fact_ids = {item.id for item in bundle.world.canon_facts}
    agent_ids = {item.id for item in bundle.agents.agents}
    memory_ids = {item.id for item in bundle.memories.memories}
    metadata_public_fact_ids = (
        {item.id for item in bundle.metadata.public_facts} if bundle.metadata else set()
    )
    metadata_rumor_ids = {item.id for item in bundle.metadata.rumors} if bundle.metadata else set()
    permission_tag_ids = (
        {item.value for item in bundle.metadata.permission_tags} if bundle.metadata else set()
    )
    known_core_ids = (
        location_ids | faction_ids | entity_ids | resource_ids | canon_fact_ids | agent_ids
    )

    if bundle.world.player and bundle.world.player.current_location_id:
        _require_ref(
            errors,
            bundle.world.player.current_location_id,
            location_ids,
            file="world.yaml",
            object_id=bundle.world.player.id,
            field_path="player.current_location_id",
            error_type="missing_referenced_location",
        )

    for entity in bundle.world.entities:
        if entity.location_id:
            _require_ref(
                errors,
                entity.location_id,
                location_ids,
                file="world.yaml",
                object_id=entity.id,
                field_path="entities.location_id",
                error_type="missing_referenced_location",
            )

    for resource in bundle.world.resources:
        if resource.location_id:
            _require_ref(
                errors,
                resource.location_id,
                location_ids,
                file="world.yaml",
                object_id=resource.id,
                field_path="resources.location_id",
                error_type="missing_referenced_location",
            )
        if resource.owner_agent_id:
            _require_ref(
                errors,
                resource.owner_agent_id,
                agent_ids,
                file="world.yaml",
                object_id=resource.id,
                field_path="resources.owner_agent_id",
                error_type="missing_referenced_agent",
            )
        if resource.owner_entity_id:
            _require_ref(
                errors,
                resource.owner_entity_id,
                entity_ids,
                file="world.yaml",
                object_id=resource.id,
                field_path="resources.owner_entity_id",
                error_type="missing_referenced_entity",
            )
        for agent_id in resource.discovery_state.discovered_by_agent_ids:
            _require_ref(
                errors,
                agent_id,
                agent_ids,
                file="world.yaml",
                object_id=resource.id,
                field_path="resources.discovery_state.discovered_by_agent_ids",
                error_type="missing_referenced_agent",
            )

    for fact in bundle.world.canon_facts:
        if fact.location_id:
            _require_ref(
                errors,
                fact.location_id,
                location_ids,
                file="world.yaml",
                object_id=fact.id,
                field_path="canon_facts.location_id",
                error_type="missing_referenced_location",
            )
        allowed = known_core_ids | set(fact.external_ref_ids)
        for ref_id in (*fact.subject_ids, *fact.object_ids):
            _require_ref(
                errors,
                ref_id,
                allowed,
                file="world.yaml",
                object_id=fact.id,
                field_path="canon_facts.subject_ids/object_ids",
                error_type="missing_referenced_core_entity",
            )

    for agent in bundle.agents.agents:
        _require_ref(
            errors,
            agent.current_location_id,
            location_ids,
            file="agents.yaml",
            object_id=agent.id,
            field_path="agents.current_location_id",
            error_type="missing_referenced_location",
        )
        if agent.faction_id:
            _require_ref(
                errors,
                agent.faction_id,
                faction_ids,
                file="agents.yaml",
                object_id=agent.id,
                field_path="agents.faction_id",
                error_type="missing_referenced_faction",
            )

    for relationship in bundle.agents.relationships:
        _require_ref(
            errors,
            relationship.source_agent_id,
            agent_ids,
            file="agents.yaml",
            object_id=relationship.id,
            field_path="relationships.source_agent_id",
            error_type="missing_referenced_agent",
        )
        _require_ref(
            errors,
            relationship.target_agent_id,
            agent_ids,
            file="agents.yaml",
            object_id=relationship.id,
            field_path="relationships.target_agent_id",
            error_type="missing_referenced_agent",
        )

    for belief in bundle.beliefs.beliefs:
        _require_ref(
            errors,
            belief.owner_agent_id,
            agent_ids,
            file="beliefs.yaml",
            object_id=belief.id,
            field_path="beliefs.owner_agent_id",
            error_type="missing_referenced_agent",
        )
        for fact_id in belief.related_canon_fact_ids:
            _require_ref(
                errors,
                fact_id,
                canon_fact_ids,
                file="beliefs.yaml",
                object_id=belief.id,
                field_path="beliefs.related_canon_fact_ids",
                error_type="missing_referenced_canon_fact",
            )
        for memory_id in belief.source_memory_ids:
            _require_ref(
                errors,
                memory_id,
                memory_ids,
                file="beliefs.yaml",
                object_id=belief.id,
                field_path="beliefs.source_memory_ids",
                error_type="missing_referenced_memory",
            )

    for secret in bundle.beliefs.secrets:
        if secret.owner_agent_id:
            _require_ref(
                errors,
                secret.owner_agent_id,
                agent_ids,
                file="beliefs.yaml",
                object_id=secret.id,
                field_path="secrets.owner_agent_id",
                error_type="missing_referenced_agent",
            )
        if secret.owner_faction_id:
            _require_ref(
                errors,
                secret.owner_faction_id,
                faction_ids,
                file="beliefs.yaml",
                object_id=secret.id,
                field_path="secrets.owner_faction_id",
                error_type="missing_referenced_faction",
            )

    for memory in bundle.memories.memories:
        _require_ref(
            errors,
            memory.owner_agent_id,
            agent_ids,
            file="memories.yaml",
            object_id=memory.id,
            field_path="memories.owner_agent_id",
            error_type="missing_referenced_agent",
        )
        if memory.related_location_id:
            _require_ref(
                errors,
                memory.related_location_id,
                location_ids,
                file="memories.yaml",
                object_id=memory.id,
                field_path="memories.related_location_id",
                error_type="missing_referenced_location",
            )
        for agent_id in memory.related_agent_ids:
            _require_ref(
                errors,
                agent_id,
                agent_ids,
                file="memories.yaml",
                object_id=memory.id,
                field_path="memories.related_agent_ids",
                error_type="missing_referenced_agent",
            )
        for entity_id in memory.related_entity_ids:
            _require_ref(
                errors,
                entity_id,
                entity_ids,
                file="memories.yaml",
                object_id=memory.id,
                field_path="memories.related_entity_ids",
                error_type="missing_referenced_entity",
            )
        for resource_id in memory.related_resource_ids:
            _require_ref(
                errors,
                resource_id,
                resource_ids,
                file="memories.yaml",
                object_id=memory.id,
                field_path="memories.related_resource_ids",
                error_type="missing_referenced_resource",
            )

    if bundle.metadata is not None:
        metadata_allowed = (
            known_core_ids | metadata_public_fact_ids | metadata_rumor_ids | permission_tag_ids
        )
        for fact in bundle.metadata.public_facts:
            if fact.location_id:
                _require_ref(
                    errors,
                    fact.location_id,
                    location_ids,
                    file="metadata.yaml",
                    object_id=fact.id,
                    field_path="public_facts.location_id",
                    error_type="missing_referenced_location",
                )
            for ref_id in (*fact.subject_ids, *fact.object_ids):
                _require_ref(
                    errors,
                    ref_id,
                    metadata_allowed,
                    file="metadata.yaml",
                    object_id=fact.id,
                    field_path="public_facts.subject_ids/object_ids",
                    error_type="missing_referenced_core_entity",
                )
        for rumor in bundle.metadata.rumors:
            if rumor.source_agent_id:
                _require_ref(
                    errors,
                    rumor.source_agent_id,
                    agent_ids,
                    file="metadata.yaml",
                    object_id=rumor.id,
                    field_path="rumors.source_agent_id",
                    error_type="missing_referenced_agent",
                )
            if rumor.location_id:
                _require_ref(
                    errors,
                    rumor.location_id,
                    location_ids,
                    file="metadata.yaml",
                    object_id=rumor.id,
                    field_path="rumors.location_id",
                    error_type="missing_referenced_location",
                )
            for ref_id in (*rumor.subject_ids, *rumor.object_ids):
                _require_ref(
                    errors,
                    ref_id,
                    metadata_allowed,
                    file="metadata.yaml",
                    object_id=rumor.id,
                    field_path="rumors.subject_ids/object_ids",
                    error_type="missing_referenced_core_entity",
                )
        for pressure in bundle.metadata.pressure_seeds:
            if pressure.location_id:
                _require_ref(
                    errors,
                    pressure.location_id,
                    location_ids,
                    file="metadata.yaml",
                    object_id=pressure.id,
                    field_path="pressure_seeds.location_id",
                    error_type="missing_referenced_location",
                )
            if pressure.resource_id:
                _require_ref(
                    errors,
                    pressure.resource_id,
                    resource_ids,
                    file="metadata.yaml",
                    object_id=pressure.id,
                    field_path="pressure_seeds.resource_id",
                    error_type="missing_referenced_resource",
                )

    return errors


def _validate_canon_boundaries(bundle: SeedBundle) -> list[SeedValidationError]:
    errors: list[SeedValidationError] = []
    for fact in bundle.world.canon_facts:
        prohibited = PROHIBITED_CANON_TAGS.intersection(fact.tags)
        if prohibited:
            errors.append(
                SeedValidationError(
                    file="world.yaml",
                    object_id=fact.id,
                    field_path="canon_facts.tags",
                    error_type="belief_record_in_canon",
                    message=(
                        "CanonFact cannot be tagged as belief, rumor, secret, "
                        f"private_belief, or rejected_claim: {sorted(prohibited)}"
                    ),
                )
            )
    return errors


def _object_counts(bundle: SeedBundle) -> dict[str, int]:
    counts = {
        "locations": len(bundle.world.locations),
        "factions": len(bundle.world.factions),
        "entities": len(bundle.world.entities),
        "resources": len(bundle.world.resources),
        "canon_facts": len(bundle.world.canon_facts),
        "agents": len(bundle.agents.agents),
        "relationships": len(bundle.agents.relationships),
        "beliefs": len(bundle.beliefs.beliefs),
        "secrets": len(bundle.beliefs.secrets),
        "memories": len(bundle.memories.memories),
    }
    if bundle.metadata is not None:
        counts.update(
            {
                "public_facts": len(bundle.metadata.public_facts),
                "rumors": len(bundle.metadata.rumors),
                "pressure_seeds": len(bundle.metadata.pressure_seeds),
                "action_metadata": len(bundle.metadata.action_metadata),
                "permission_tags": len(bundle.metadata.permission_tags),
            }
        )
    return counts


def _duplicates(ids: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    return duplicates


def _require_ref(
    errors: list[SeedValidationError],
    ref_id: str,
    allowed_ids: set[str],
    *,
    file: str,
    object_id: str,
    field_path: str,
    error_type: str,
) -> None:
    if ref_id not in allowed_ids:
        errors.append(
            SeedValidationError(
                file=file,
                object_id=object_id,
                field_path=field_path,
                error_type=error_type,
                message=f"Referenced id does not exist: {ref_id}",
            )
        )
