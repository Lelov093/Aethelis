from __future__ import annotations

import pytest

from aethelis.events.fixtures import DeterministicActionProposalFactory
from aethelis.schemas.events import ActionProposal, ActionProposalSummary


def test_factory_supports_current_deterministic_scenarios() -> None:
    factory = DeterministicActionProposalFactory()

    cases = [
        ("ivo", "ivo_inspect_workshop_safe_fixture"),
        ("mira", "mira_search_archive_wrong_key"),
        ("selka", "selka_consume_stabilizer_part_fixture"),
        ("selka", "selka_restock_market_credit_fixture"),
        ("rowan", "unsafe_force_open_safe"),
        ("ivo", "malformed_or_incomplete_action"),
        ("elin", "elin_inspect_cargo_manifest_fixture"),
        ("sora", "sora_release_relief_crates_fixture"),
        ("niven", "niven_search_lantern_wrong_pass"),
        ("niven", "niven_force_quay_lock"),
    ]
    for agent_id, scenario_id in cases:
        proposal = factory.build(agent_id=agent_id, scenario_id=scenario_id)
        summary = ActionProposalSummary.from_proposal(proposal)

        assert isinstance(proposal, ActionProposal)
        assert proposal.proposer_agent_id == agent_id
        assert summary.contains_state_diff is False
        assert summary.contains_canon_mutation is False
        assert summary.generated_by == "deterministic_fixture"
        assert "StateDiff" not in proposal.model_dump_json()
        assert "CanonFact" not in proposal.model_dump_json()


def test_factory_rejects_inspect_workshop_safe_real_llm_path() -> None:
    with pytest.raises(ValueError, match="requires the explicit real LLM"):
        DeterministicActionProposalFactory().build(
            agent_id="ivo",
            scenario_id="inspect_workshop_safe",
        )


def test_factory_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="Unsupported deterministic ActionProposal scenario"):
        DeterministicActionProposalFactory().build(
            agent_id="mira",
            scenario_id="unknown_scenario",
        )


def test_malformed_or_incomplete_action_remains_underspecified_fixture() -> None:
    proposal = DeterministicActionProposalFactory().build(
        agent_id="ivo",
        scenario_id="malformed_or_incomplete_action",
    )

    assert proposal.target_location_id == "workshop_lane"
    assert proposal.target_entity_ids == ()
    assert "revision" in proposal.expected_outcome.lower()
