from pathlib import Path

from aethelis.product.content_contracts import (
    product_content_hash,
    product_content_hash_candidates,
)
from aethelis.product.content_loader import ProductContentPackageLoader, _seed_hash
from aethelis.product.projection_contracts import VisibleResourceView
from aethelis.product.projections import (
    _current_objectives,
    _dialogue_interactions,
    _opportunity_views,
    _resource_custody_label,
    _scene_actions,
    _situation_view,
)
from aethelis.schemas.world import (
    DialogueActKind,
    PlayerDialogueTurn,
    PlayerInventoryItem,
    PlayerKnowledgeKind,
    PlayerKnowledgeRecord,
)
from aethelis.seeds.loader import SeedLoader

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path("content/mistgate/v1")


def test_mistgate_product_package_separates_public_and_private_character_state() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)

    assert package.blueprint.package_id == "mistgate_product_v1_10"
    assert package.blueprint.dialogue_expression_policy.enabled
    assert package.blueprint.default_locale == "zh-CN"
    assert len(package.initial_world_state.locations) == 5
    assert len(package.initial_world_state.entities) == 9
    assert len(package.initial_world_state.agent_profiles) == 6
    assert package.initial_world_state.agent_beliefs
    assert package.initial_world_state.agent_memories
    assert package.initial_world_state.agent_relationships
    assert len(package.blueprint.routes) == 4
    assert len(package.blueprint.dialogue_options) == 2
    assert len(package.blueprint.resource_exchange_options) == 1
    assert len(package.blueprint.repair_options) == 1
    assert len(package.blueprint.commitment_breach_options) == 1
    assert len(package.blueprint.resource_release_options) == 1
    assert len(package.blueprint.resource_validation_options) == 1
    assert len(package.blueprint.final_repair_options) == 1
    assert len(package.blueprint.world_response_options) == 2
    assert len(package.blueprint.asset_slots) == 12
    assert sum(slot.status == "approved" for slot in package.blueprint.asset_slots) == 7
    assert sum(slot.status == "planned" for slot in package.blueprint.asset_slots) == 5
    assert len(package.blueprint.opportunities) == 4
    assert len(package.blueprint.pacing_rules) == 2
    assert len(package.blueprint.recovery_paths) == 4
    assert len(package.blueprint.outcomes) == 2
    assert package.initial_world_state.player.current_location_id == "council_square"
    rowan = next(item for item in package.initial_world_state.entities if item.id == "rowan")
    assert rowan.name == "罗文·凯斯特"
    assert "Rowan knows the regulator is unstable" not in rowan.summary
    rowan_agent = next(
        item for item in package.initial_world_state.agent_profiles if item.id == "rowan"
    )
    assert rowan_agent.private_summary
    assert len(product_content_hash(package)) == 64


def test_product_package_pins_canonical_source_seed_hash() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)
    seed = SeedLoader().load(ROOT / "seeds/mistgate_v01")

    assert seed.bundle is not None
    assert _seed_hash(seed.bundle) == package.blueprint.source_seed_sha256


def test_published_v1_3_package_has_a_narrow_legacy_hash_candidate() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)
    legacy = package.model_copy(
        update={
            "blueprint": package.blueprint.model_copy(
                update={
                    "package_id": "mistgate_product_v1_3",
                    "content_version_id": "mistgate_product_v1_3_0",
                    "repair_options": (),
                    "commitment_breach_options": (),
                }
            )
        }
    )

    candidates = product_content_hash_candidates(legacy)
    assert candidates[0] == product_content_hash(legacy)
    assert len(candidates) == 2


def test_published_v1_4_package_has_a_narrow_legacy_hash_candidate() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)
    legacy = package.model_copy(
        update={
            "blueprint": package.blueprint.model_copy(
                update={
                    "package_id": "mistgate_product_v1_4",
                    "content_version_id": "mistgate_product_v1_4_0",
                    "repair_options": (),
                    "commitment_breach_options": (),
                }
            )
        }
    )

    candidates = product_content_hash_candidates(legacy)
    assert candidates[0] == product_content_hash(legacy)
    assert len(candidates) == 2


def test_published_v1_5_package_has_a_narrow_legacy_hash_candidate() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)
    legacy = package.model_copy(
        update={
            "blueprint": package.blueprint.model_copy(
                update={
                    "package_id": "mistgate_product_v1_5",
                    "content_version_id": "mistgate_product_v1_5_0",
                    "resource_release_options": (),
                    "resource_validation_options": (),
                    "final_repair_options": (),
                }
            )
        }
    )

    candidates = product_content_hash_candidates(legacy)
    assert candidates[0] == product_content_hash(legacy)
    assert len(candidates) == 2


def test_published_v1_6_package_has_a_narrow_legacy_hash_candidate() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)
    legacy = package.model_copy(
        update={
            "blueprint": package.blueprint.model_copy(
                update={
                    "package_id": "mistgate_product_v1_6",
                    "content_version_id": "mistgate_product_v1_6_0",
                    "world_response_options": (),
                }
            )
        }
    )

    candidates = product_content_hash_candidates(legacy)
    assert candidates[0] == product_content_hash(legacy)
    assert len(candidates) == 2


def test_published_v1_7_package_has_a_narrow_living_world_hash_candidate() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)
    old_actions = tuple(
        action
        for action in package.blueprint.actions
        if action.action_id not in {"advance_world", "ask_world"}
    )
    legacy = package.model_copy(
        update={
            "blueprint": package.blueprint.model_copy(
                update={
                    "package_id": "mistgate_product_v1_7",
                    "content_version_id": "mistgate_product_v1_7_0",
                    "actions": old_actions,
                }
            )
        }
    )

    candidates = product_content_hash_candidates(legacy)
    assert candidates[0] == product_content_hash(legacy)
    assert len(candidates) == 2


def test_published_v1_9_package_has_a_narrow_opportunity_hash_candidate() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)
    legacy = package.model_copy(
        update={
            "blueprint": package.blueprint.model_copy(
                update={
                    "package_id": "mistgate_product_v1_9",
                    "content_version_id": "mistgate_product_v1_9_0",
                }
            )
        }
    )

    candidates = product_content_hash_candidates(legacy)
    assert candidates[0] == product_content_hash(legacy)
    assert len(candidates) == 2


def test_scene_actions_offer_targeted_inspection_for_visible_resources() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)

    actions = _scene_actions(
        package=package,
        locale="zh-CN",
        world=package.initial_world_state,
        location_id="market_row",
        has_undiscovered=False,
        visible_resources=(
            VisibleResourceView(
                id="stabilizer_parts",
                name="稳定器零件",
                summary="由商会控制的维修材料。",
                quantity=3,
            ),
        ),
    )

    inspection = next(action for action in actions if action.action_id == "inspect_resource")
    assert inspection.target_id == "stabilizer_parts"
    assert inspection.label == "仔细检查：稳定器零件"
    dialogue_targets = {
        action.target_id for action in actions if action.action_id == "ask_character"
    }
    assert dialogue_targets == {"selka", "nara"}


def test_initial_situation_projects_parallel_leads_and_recovery() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)
    world = package.initial_world_state
    locations = {item.id: item.name for item in world.locations}

    opportunities = _opportunity_views(
        package,
        world,
        world.player.id,
        "zh-CN",
        locations,
    )
    objectives = _current_objectives(package, world, "zh-CN", opportunities)
    situation = _situation_view(package, world, "zh-CN")

    assert situation.phase == "unstable"
    assert situation.completed_steps == 0
    assert situation.recovery_guidance == (
        "若缺少稳定器零件，可在集市街调查供应与商会条件，而不是让世界陷入死锁。",
    )
    assert {item.id for item in opportunities if not item.is_optional} == {
        "opportunity_market_parts",
        "opportunity_workshop_key",
        "opportunity_aqueduct_signal",
    }
    assert {item.title for item in opportunities if not item.is_optional} == set(objectives)


def test_key_and_lens_can_progress_before_market_parts() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)
    player = package.initial_world_state.player.model_copy(
        update={
            "inventory": (
                PlayerInventoryItem(
                    id="inventory_calibration_key",
                    resource_id="calibration_key",
                    quantity=1,
                    acquired_event_id="event_key",
                ),
            ),
            "knowledge": (
                PlayerKnowledgeRecord(
                    id="knowledge_gate_lens_validated",
                    kind=PlayerKnowledgeKind.CONFIRMED_FACT,
                    statement="The gate lens was validated.",
                    source_entity_id="gate_lens",
                    confidence="high",
                    committed_event_id="event_lens",
                ),
            ),
        }
    )
    world = package.initial_world_state.model_copy(update={"player": player})
    opportunities = _opportunity_views(
        package,
        world,
        player.id,
        "zh-CN",
        {item.id: item.name for item in world.locations},
    )
    situation = _situation_view(package, world, "zh-CN")
    objectives = _current_objectives(package, world, "zh-CN", opportunities)

    statuses = {item.id: item.is_completed for item in opportunities}
    assert statuses["opportunity_workshop_key"]
    assert statuses["opportunity_aqueduct_signal"]
    assert not statuses["opportunity_market_parts"]
    assert situation.completed_steps == 2
    assert objectives == ("追踪稀缺零件",)


def test_dialogue_projection_groups_mixed_turns_into_one_history_version() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)
    turns = (
        PlayerDialogueTurn(
            id="turn_fixed",
            interaction_id="interaction_1",
            character_id="selka",
            dialogue_option_id="ask_selka_about_parts",
            utterance="零件确实紧张。",
            committed_event_id="event_fixed",
        ),
        PlayerDialogueTurn(
            id="turn_free",
            interaction_id="interaction_1",
            character_id="selka",
            dialogue_act=DialogueActKind.QUESTION,
            player_utterance="那你希望我先做什么？",
            utterance="先拿出可行的修理方案。",
            committed_event_id="event_free",
        ),
    )

    projected = _dialogue_interactions(package, turns, {"selka": "赛尔卡"}, "zh-CN")

    assert len(projected) == 1
    assert projected[0].contains_free_expression
    assert [exchange.input_kind for exchange in projected[0].exchanges] == ["preset", "free"]
    assert projected[0].exchanges[0].player_text == "向塞尔卡询问稳定器零件"
    assert projected[0].exchanges[1].player_text == "那你希望我先做什么？"


def test_resource_custody_uses_public_world_names_without_private_state() -> None:
    package = ProductContentPackageLoader(ROOT).load(PACKAGE)
    world = package.initial_world_state
    resource = next(item for item in world.resources if item.id == "calibration_key")

    label = _resource_custody_label(
        resource,
        location_names={item.id: item.name for item in world.locations},
        entity_names={item.id: item.name for item in world.entities},
    )

    assert label == "由工坊保险柜保管"
    assert "private" not in label.lower()
