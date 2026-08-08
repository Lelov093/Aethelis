from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
VARIANT_SEED = ROOT / "seeds" / "mistgate_v01_civic_pressure_variant"
HARBOR_SEED = ROOT / "seeds" / "harbor_lantern_v01"


def validate_seed(path: Path):
    load_result = SeedLoader().load(path)
    return SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )


def copy_seed(tmp_path: Path) -> Path:
    target = tmp_path / "seed"
    shutil.copytree(VALID_SEED, target)
    return target


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)


def error_types(report) -> set[str]:
    return {error.error_type for error in report.errors}


def test_mistgate_seed_is_valid() -> None:
    report = validate_seed(VALID_SEED)
    bundle = SeedLoader().load(VALID_SEED).bundle
    assert bundle is not None

    assert report.success
    assert report.schema_version == "0.1"
    assert report.object_counts["agents"] == 6
    assert report.object_counts["locations"] == 5
    assert report.object_counts["resources"] == 4
    assert report.object_counts["canon_facts"] == 8
    assert report.object_counts["public_facts"] == 4
    assert report.object_counts["rumors"] == 3
    assert report.object_counts["pressure_seeds"] == 4
    assert report.object_counts["action_metadata"] == 6
    assert {location.id for location in bundle.world.locations} == {
        "central_archive",
        "council_square",
        "market_row",
        "workshop_lane",
        "old_aqueduct",
    }
    assert "guard_tower" not in {location.id for location in bundle.world.locations}
    council_square = next(
        location for location in bundle.world.locations if location.id == "council_square"
    )
    assert "guard_tower" in council_square.facilities
    assert {agent.name for agent in bundle.agents.agents} == {
        "Mira Vale",
        "Rowan Kest",
        "Selka Orin",
        "Ivo Renn",
        "Nara Vey",
        "Taren Sol",
    }
    assert "player" not in {agent.id for agent in bundle.agents.agents}
    assert "Tamsin" not in {agent.name for agent in bundle.agents.agents}
    assert "Nera" not in {agent.name for agent in bundle.agents.agents}


def test_mistgate_civic_pressure_variant_seed_is_valid() -> None:
    report = validate_seed(VARIANT_SEED)
    bundle = SeedLoader().load(VARIANT_SEED).bundle
    assert bundle is not None

    assert report.success
    assert report.object_counts["agents"] == 6
    assert report.object_counts["locations"] == 5
    pressure_levels = {pressure.id: pressure.level for pressure in bundle.metadata.pressure_seeds}
    relationships = {
        relationship.id: relationship.trust for relationship in bundle.agents.relationships
    }
    beliefs = {belief.id: belief.confidence for belief in bundle.beliefs.beliefs}

    assert pressure_levels["pressure_civic_trust"] == 8
    assert pressure_levels["pressure_rumor_spread"] == 7
    assert relationships["rel_rowan_selka"] == -3
    assert beliefs["belief_mira_key_in_archive"].value == "high"


def test_harbor_lantern_seed_is_valid() -> None:
    report = validate_seed(HARBOR_SEED)
    bundle = SeedLoader().load(HARBOR_SEED).bundle
    assert bundle is not None

    assert report.success
    assert report.object_counts["agents"] == 4
    assert report.object_counts["locations"] == 3
    assert report.object_counts["resources"] == 3
    assert report.object_counts["canon_facts"] == 5
    assert report.object_counts["beliefs"] == 4
    assert report.object_counts["pressure_seeds"] == 3
    assert {agent.id for agent in bundle.agents.agents} == {"elin", "bram", "sora", "niven"}
    assert {location.id for location in bundle.world.locations} == {
        "ledger_house",
        "quay_gate",
        "lantern_room",
    }
    assert {resource.id for resource in bundle.world.resources} == {
        "harbor_pass",
        "signal_oil",
        "relief_crates",
    }
    assert "calibration_key" not in {resource.id for resource in bundle.world.resources}


def test_duplicate_id_is_reported(tmp_path: Path) -> None:
    seed = copy_seed(tmp_path)
    world_path = seed / "world.yaml"
    world = read_yaml(world_path)
    world["locations"][1]["id"] = world["locations"][0]["id"]
    write_yaml(world_path, world)

    report = validate_seed(seed)

    assert not report.success
    assert "duplicate_id" in error_types(report)


def test_missing_referenced_location_is_reported(tmp_path: Path) -> None:
    seed = copy_seed(tmp_path)
    agents_path = seed / "agents.yaml"
    agents = read_yaml(agents_path)
    agents["agents"][0]["current_location_id"] = "missing_location"
    write_yaml(agents_path, agents)

    report = validate_seed(seed)

    assert not report.success
    assert "missing_referenced_location" in error_types(report)


def test_belief_owner_invalid_is_reported(tmp_path: Path) -> None:
    seed = copy_seed(tmp_path)
    beliefs_path = seed / "beliefs.yaml"
    beliefs = read_yaml(beliefs_path)
    beliefs["beliefs"][0]["owner_agent_id"] = "missing_agent"
    write_yaml(beliefs_path, beliefs)

    report = validate_seed(seed)

    assert not report.success
    assert "missing_referenced_agent" in error_types(report)


def test_relationship_target_invalid_is_reported(tmp_path: Path) -> None:
    seed = copy_seed(tmp_path)
    agents_path = seed / "agents.yaml"
    agents = read_yaml(agents_path)
    agents["relationships"][0]["target_agent_id"] = "missing_agent"
    write_yaml(agents_path, agents)

    report = validate_seed(seed)

    assert not report.success
    assert "missing_referenced_agent" in error_types(report)


def test_belief_tag_in_canon_is_boundary_error(tmp_path: Path) -> None:
    seed = copy_seed(tmp_path)
    world_path = seed / "world.yaml"
    world = read_yaml(world_path)
    world["canon_facts"][0]["tags"].append("belief")
    write_yaml(world_path, world)

    report = validate_seed(seed)

    assert not report.success
    assert "belief_record_in_canon" in error_types(report)


def test_canon_fact_missing_core_reference_is_reported(tmp_path: Path) -> None:
    seed = copy_seed(tmp_path)
    world_path = seed / "world.yaml"
    world = read_yaml(world_path)
    world["canon_facts"][0]["subject_ids"] = ["missing_core_entity"]
    write_yaml(world_path, world)

    report = validate_seed(seed)

    assert not report.success
    assert "missing_referenced_core_entity" in error_types(report)


def test_missing_product_location_is_reported(tmp_path: Path) -> None:
    seed = copy_seed(tmp_path)
    world_path = seed / "world.yaml"
    world = read_yaml(world_path)
    world["locations"] = [
        location for location in world["locations"] if location["id"] != "council_square"
    ]
    write_yaml(world_path, world)

    report = validate_seed(seed)

    assert not report.success
    assert "missing_product_location" in error_types(report)


def test_guard_tower_must_not_be_location(tmp_path: Path) -> None:
    seed = copy_seed(tmp_path)
    world_path = seed / "world.yaml"
    world = read_yaml(world_path)
    world["locations"].append(
        {
            "id": "guard_tower",
            "name": "Guard Tower",
            "summary": "Invalid sixth location for Phase 2A.",
        }
    )
    write_yaml(world_path, world)

    report = validate_seed(seed)

    assert not report.success
    assert "guard_tower_must_be_facility" in error_types(report)


def test_player_must_not_be_seed_agent(tmp_path: Path) -> None:
    seed = copy_seed(tmp_path)
    agents_path = seed / "agents.yaml"
    agents = read_yaml(agents_path)
    player_agent = dict(agents["agents"][0])
    player_agent["id"] = "player"
    player_agent["name"] = "Player"
    agents["agents"].append(player_agent)
    write_yaml(agents_path, agents)

    report = validate_seed(seed)

    assert not report.success
    assert "player_must_not_be_seed_agent" in error_types(report)


def test_invalid_rumor_source_is_reported(tmp_path: Path) -> None:
    seed = copy_seed(tmp_path)
    metadata_path = seed / "metadata.yaml"
    metadata = read_yaml(metadata_path)
    metadata["rumors"][0]["source_agent_id"] = "missing_agent"
    write_yaml(metadata_path, metadata)

    report = validate_seed(seed)

    assert not report.success
    assert "missing_referenced_agent" in error_types(report)


def test_invalid_pressure_resource_is_reported(tmp_path: Path) -> None:
    seed = copy_seed(tmp_path)
    metadata_path = seed / "metadata.yaml"
    metadata = read_yaml(metadata_path)
    metadata["pressure_seeds"][0]["resource_id"] = "missing_resource"
    write_yaml(metadata_path, metadata)

    report = validate_seed(seed)

    assert not report.success
    assert "missing_referenced_resource" in error_types(report)
