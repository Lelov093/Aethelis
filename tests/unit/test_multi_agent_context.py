from __future__ import annotations

from pathlib import Path

from aethelis.agents.retrieval import build_multi_agent_step_context
from aethelis.schemas.common import RecordStatus
from aethelis.schemas.ledger import BeliefKind
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"


def test_active_set_context_keeps_per_agent_frames_isolated() -> None:
    context = build_multi_agent_step_context(
        _load_valid_bundle(),
        step_id="r5_b2_step",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo", "mira"),
    )

    ivo = context.frame_for("ivo")
    mira = context.frame_for("mira")

    assert context.no_shared_omniscient_context is True
    assert ivo.observation.location_id == "workshop_lane"
    assert mira.observation.location_id == "central_archive"
    assert "belief_ivo_key_in_safe" in ivo.retrieval.selected_belief_ids
    assert "belief_mira_key_in_archive" in mira.retrieval.selected_belief_ids
    assert "belief_mira_key_in_archive" not in ivo.retrieval.selected_belief_ids
    assert "mem_mira_archive_ledger" not in ivo.retrieval.selected_memory_ids
    assert mira.boundary_flags.private_cross_agent_beliefs_filtered > 0
    assert ivo.boundary_flags.cross_agent_memories_filtered > 0
    assert ivo.boundary_flags.hidden_canon_filtered > 0


def test_context_packing_records_selected_and_suppressed_sources() -> None:
    context = build_multi_agent_step_context(
        _load_valid_bundle(),
        step_id="r5_b2_budget",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo",),
        context_budget_per_agent=3,
    )
    frame = context.frame_for("ivo")
    records = frame.retrieval.source_records

    assert len(frame.packed_source_ids) == 3
    assert frame.suppressed_source_ids
    assert all(record.selected for record in records if record.source_id in frame.packed_source_ids)
    assert all(
        record.suppress_reason == "context_budget"
        for record in records
        if record.source_id in frame.suppressed_source_ids
    )
    assert [record.score for record in records] == sorted(
        (record.score for record in records),
        reverse=True,
    )
    assert {record.source_type for record in records} >= {"belief", "memory", "relationship"}


def test_rejected_inactive_or_outdated_beliefs_are_filtered_from_sources() -> None:
    bundle = _load_valid_bundle()
    rejected_id = "belief_mira_key_in_archive"
    inactive_id = "belief_rowan_no_key_location"
    outdated_id = "belief_rowan_repair_requires_clearance"
    beliefs = tuple(
        belief.model_copy(
            update={
                "kind": BeliefKind.REJECTED_CLAIM,
                "status": RecordStatus.REJECTED,
            }
        )
        if belief.id == rejected_id
        else belief.model_copy(update={"status": RecordStatus.INACTIVE})
        if belief.id == inactive_id
        else belief.model_copy(update={"status": RecordStatus.SUPERSEDED})
        if belief.id == outdated_id
        else belief
        for belief in bundle.beliefs.beliefs
    )
    bundle = bundle.model_copy(
        update={"beliefs": bundle.beliefs.model_copy(update={"beliefs": beliefs})}
    )

    context = build_multi_agent_step_context(
        bundle,
        step_id="r5_b2_filtered",
        scenario_id="mira_search_archive_wrong_key",
        active_agent_ids=("mira", "rowan"),
        context_budget_per_agent=20,
    )
    _assert_filtered_belief_context(context.frame_for("mira"), rejected_id)
    _assert_filtered_belief_context(context.frame_for("rowan"), inactive_id)
    _assert_filtered_belief_context(context.frame_for("rowan"), outdated_id)


def test_suppressed_memories_are_filtered_from_source_records() -> None:
    bundle = _load_valid_bundle()
    suppressed_ids = tuple(
        memory.id for memory in bundle.memories.memories if memory.owner_agent_id == "ivo"
    )
    memories = tuple(
        memory.model_copy(
            update={
                "salience": 1,
                "source_event_id": None,
                "related_entity_ids": (),
                "related_resource_ids": (),
            }
        )
        if memory.id in suppressed_ids
        else memory
        for memory in bundle.memories.memories
    )
    bundle = bundle.model_copy(
        update={"memories": bundle.memories.model_copy(update={"memories": memories})}
    )

    context = build_multi_agent_step_context(
        bundle,
        step_id="r5_b2_suppressed_memory",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo",),
        context_budget_per_agent=20,
    )
    frame = context.frame_for("ivo")

    assert suppressed_ids
    for memory_id in suppressed_ids:
        _assert_suppressed_memory_context(frame, memory_id)


def test_active_set_context_is_read_only_evidence_not_mutation_path() -> None:
    context = build_multi_agent_step_context(
        _load_valid_bundle(),
        step_id="r5_b2_read_only",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        active_agent_ids=("ivo", "taren"),
    )
    taren = context.frame_for("taren")

    assert context.can_modify_world_state is False
    assert context.can_mutate_canon is False
    assert taren.boundary_flags.can_modify_world_state is False
    assert taren.boundary_flags.can_mutate_canon is False
    assert taren.boundary_flags.faction_limited_filtered > 0
    assert "state_diff" not in str(context.safe_summary()).lower()


def _load_valid_bundle():
    load_result = SeedLoader().load(VALID_SEED)
    report = SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )
    assert report.success
    assert load_result.bundle is not None
    return load_result.bundle


def _assert_filtered_belief_context(frame, belief_id: str) -> None:
    selected_source_ids = {
        record.source_id for record in frame.retrieval.source_records if record.selected
    }

    assert belief_id not in frame.retrieval.selected_belief_ids
    assert belief_id not in frame.packed_source_ids
    assert belief_id not in selected_source_ids
    assert belief_id in frame.retrieval.filtered_belief_ids
    assert belief_id in frame.boundary_flags.rejected_or_outdated_filtered_ids


def _assert_suppressed_memory_context(frame, memory_id: str) -> None:
    selected_source_ids = {
        record.source_id for record in frame.retrieval.source_records if record.selected
    }

    assert memory_id not in frame.retrieval.selected_memory_ids
    assert memory_id not in frame.packed_source_ids
    assert memory_id not in selected_source_ids
    assert memory_id in frame.retrieval.suppressed_memory_ids
