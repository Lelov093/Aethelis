from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import yaml

from aethelis.product.content_contracts import (
    ObjectPresentation,
    ProductContentBlueprint,
    ProductContentPackage,
)
from aethelis.schemas.common import RecordStatus
from aethelis.schemas.world import Entity, EntityKind, PlayerContext
from aethelis.seeds.loader import SeedLoader


class ProductContentLoadError(ValueError):
    pass


class ProductContentPackageLoader:
    def __init__(self, repository_root: Path) -> None:
        self._root = repository_root.resolve()

    def load(self, package_directory: Path) -> ProductContentPackage:
        package_directory = self._contained_path(package_directory)
        blueprint = ProductContentBlueprint.model_validate(
            _read_yaml(package_directory / "package.yaml")
        )
        localized_text = {
            locale: _string_map(
                _read_yaml(package_directory / "locales" / f"{locale}.yaml"),
                filename=f"locales/{locale}.yaml",
            )
            for locale in blueprint.supported_locales
        }
        seed_path = self._contained_path(Path(blueprint.source_seed_path))
        seed_result = SeedLoader().load(seed_path)
        if seed_result.bundle is None or seed_result.errors:
            messages = "; ".join(error.message for error in seed_result.errors)
            raise ProductContentLoadError(f"source seed is invalid: {messages}")
        actual_seed_hash = _seed_hash(seed_result.bundle)
        if actual_seed_hash != blueprint.source_seed_sha256:
            raise ProductContentLoadError(
                "source seed hash mismatch; conversion input changed without a content version"
            )

        world = seed_result.bundle.world
        presentations = {
            (item.object_type, item.object_id): item for item in blueprint.presentations
        }
        locations = tuple(
            location.model_copy(
                update=_localized_fields(
                    presentations.get(("location", location.id)),
                    localized_text[blueprint.default_locale],
                )
            )
            for location in world.locations
        )
        resources = tuple(
            resource.model_copy(
                update=_localized_fields(
                    presentations.get(("resource", resource.id)),
                    localized_text[blueprint.default_locale],
                )
            )
            for resource in world.resources
        )
        existing_entity_ids = {entity.id for entity in world.entities}
        character_entities = []
        for agent in seed_result.bundle.agents.agents:
            if agent.id in existing_entity_ids:
                raise ProductContentLoadError(f"agent/entity id collision: {agent.id}")
            presentation = presentations.get(("character", agent.id))
            fields = _localized_fields(
                presentation,
                localized_text[blueprint.default_locale],
            )
            character_entities.append(
                Entity(
                    id=agent.id,
                    name=fields.get("name", agent.name),
                    kind=EntityKind.PERSON_REFERENCE,
                    location_id=agent.current_location_id,
                    status=RecordStatus.ACTIVE,
                    summary=fields.get("summary", agent.public_summary),
                    tags=(
                        "character",
                        f"role:{agent.role}",
                        *([f"faction:{agent.faction_id}"] if agent.faction_id else []),
                    ),
                )
            )
        entities = tuple(
            entity.model_copy(
                update=_localized_fields(
                    presentations.get(("entity", entity.id)),
                    localized_text[blueprint.default_locale],
                )
            )
            for entity in world.entities
        ) + tuple(character_entities)
        initial_world = world.model_copy(
            update={
                "name": localized_text[blueprint.default_locale][blueprint.world_name_key],
                "locations": locations,
                "entities": entities,
                "resources": resources,
                "agent_profiles": seed_result.bundle.agents.agents,
                "agent_beliefs": seed_result.bundle.beliefs.beliefs,
                "agent_memories": seed_result.bundle.memories.memories,
                "agent_relationships": seed_result.bundle.agents.relationships,
                "player": PlayerContext(
                    id="player_profile_template",
                    summary=localized_text[blueprint.default_locale][blueprint.player_summary_key],
                    current_location_id=blueprint.entry_location_id,
                    governance_notes=("玩家行动必须经过验证和事件提交后才能改变世界状态。",),
                ),
            }
        )
        return ProductContentPackage(
            blueprint=blueprint,
            localized_text=localized_text,
            initial_world_state=initial_world,
        )

    def _contained_path(self, path: Path) -> Path:
        resolved = path.resolve() if path.is_absolute() else (self._root / path).resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise ProductContentLoadError("content path escapes the repository root")
        return resolved


def _read_yaml(path: Path) -> object:
    if not path.is_file():
        raise ProductContentLoadError(f"required content file is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ProductContentLoadError(f"invalid YAML in {path}: {exc}") from exc


def _string_map(value: object, *, filename: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(text, str) and text.strip()
        for key, text in value.items()
    ):
        raise ProductContentLoadError(f"{filename} must contain non-empty string values")
    return value


def _seed_hash(bundle) -> str:  # type: ignore[no-untyped-def]
    encoded = json.dumps(
        bundle.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _localized_fields(
    presentation: ObjectPresentation | None,
    texts: dict[str, str],
) -> dict[str, str]:
    if presentation is None:
        return {}
    return {
        "name": texts[presentation.name_key],
        "summary": texts[presentation.summary_key],
    }
