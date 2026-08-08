from __future__ import annotations

from pathlib import Path

from aethelis.agents.retrieval import CognitionRetriever
from aethelis.evolution import evolution_state_safe_context
from aethelis.schemas.evolution import EvolutionRuntimeState
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"


def test_cognition_retriever_returns_own_context_and_safe_summary_only() -> None:
    bundle = _load_valid_bundle()
    retrieved = CognitionRetriever().retrieve(
        bundle,
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        pressure_context={"pressure_seed_count": 4},
        evolution_context=evolution_state_safe_context(EvolutionRuntimeState()),
    )

    assert retrieved.observation.location.id == "workshop_lane"
    assert "belief_ivo_key_in_safe" in {belief.id for belief in retrieved.cognition.owned_beliefs}
    assert "belief_mira_key_in_archive" not in {
        belief.id for belief in retrieved.cognition.owned_beliefs
    }
    assert retrieved.summary.own_belief_count == len(retrieved.cognition.owned_beliefs)
    assert retrieved.summary.own_memory_count == len(retrieved.cognition.owned_memories)
    assert retrieved.summary.pressure_context_available is True
    assert retrieved.summary.evolution_context_available is True
    safe_summary = str(retrieved.summary.safe_summary())
    assert "secret_" not in safe_summary
    assert "canon_key_in_workshop_safe" not in safe_summary
    assert "calibration key is in the workshop safe" not in safe_summary.lower()


def test_cognition_retriever_exposes_visible_relationship_count_without_private_dump() -> None:
    bundle = _load_valid_bundle()
    retrieved = CognitionRetriever().retrieve(
        bundle,
        agent_id="mira",
        scenario_id="mira_search_archive_wrong_key",
    )

    assert retrieved.summary.visible_relationship_count == len(retrieved.visible_relationships)
    assert retrieved.summary.hidden_context_used is False
    assert retrieved.summary.provider_called is False
    assert "private_summary" not in str(retrieved.summary.safe_summary())


def test_retrieval_suppression_removes_memory_from_selected_cognition() -> None:
    bundle = _load_valid_bundle()
    memories = tuple(
        memory.model_copy(update={"salience": 1, "source_event_id": None})
        if memory.owner_agent_id == "ivo"
        else memory
        for memory in bundle.memories.memories
    )
    bundle = bundle.model_copy(
        update={"memories": bundle.memories.model_copy(update={"memories": memories})}
    )

    retrieved = CognitionRetriever().retrieve(
        bundle,
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
    )

    selected = {memory.id for memory in retrieved.cognition.owned_memories}
    suppressed = set(retrieved.summary.suppressed_memory_ids)
    assert suppressed
    assert selected.isdisjoint(suppressed)
    assert set(retrieved.summary.selected_memory_ids) == selected


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
