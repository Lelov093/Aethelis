from __future__ import annotations

from pathlib import Path

from aethelis.runtime.single_step import run_single_step
from aethelis.schemas.trace import FormalTraceEnvelope, TraceType
from aethelis.trace.formal import build_formal_trace_preview

VALID_SEED = Path("seeds/mistgate_v01")


def test_formal_trace_contract_for_reject_path() -> None:
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="mira",
        scenario_id="mira_search_archive_wrong_key",
        settings=object(),  # deterministic branch does not read provider settings
    )

    trace = build_formal_trace_preview(result, seed_id="mistgate_v01")

    assert isinstance(trace, FormalTraceEnvelope)
    assert trace.trace_type == TraceType.FORMAL
    assert trace.formal_experiment_result is False
    assert trace.records[0].verification_decision.value == "reject"
    assert trace.records[0].committed_event_id is None
    assert trace.records[0].state_diff_id is None
    assert trace.records[0].state_diff_applied is False
    assert "llm_output_hash_only" in trace.records[0].safety_flags


def test_formal_trace_contract_for_player_claim() -> None:
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="player",
        scenario_id="player_claim_key_in_hand",
        settings=object(),
    )

    trace = build_formal_trace_preview(result, seed_id="mistgate_v01")
    player_claim = trace.records[0].player_claim

    assert trace.trace_type == TraceType.FORMAL
    assert trace.formal_experiment_result is False
    assert player_claim is not None
    assert player_claim.verification_decision.value == "reject"
    assert player_claim.canon_updated is False
    assert player_claim.state_diff_id is None
    assert player_claim.rejected_claim_ids == ("claim_player_key_in_hand",)
