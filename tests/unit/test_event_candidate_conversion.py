from __future__ import annotations

from aethelis.events.conversion import action_proposal_to_event_candidate, event_candidate_summary
from aethelis.events.fixtures import DeterministicActionProposalFactory


def test_event_candidate_conversion_preserves_action_proposal_links() -> None:
    proposal = DeterministicActionProposalFactory().build(
        agent_id="mira",
        scenario_id="mira_search_archive_wrong_key",
    )

    candidate = action_proposal_to_event_candidate(
        proposal,
        scenario_id="mira_search_archive_wrong_key",
        candidate_kind="archive_search",
    )
    summary = event_candidate_summary(candidate, candidate_kind="archive_search")

    assert candidate.source_action_proposal_id == proposal.id
    assert candidate.actor_agent_id == proposal.proposer_agent_id
    assert candidate.involved_location_ids == ("central_archive",)
    assert candidate.involved_entity_ids == ("harmonic_tuner",)
    assert summary.event_candidate_id == candidate.id
    assert summary.source_action_proposal_id == proposal.id
    assert summary.actor_agent_id == "mira"
    assert summary.candidate_kind == "archive_search"
    assert summary.can_modify_world_state is False
    assert summary.predicted_state_diff_id is None


def test_event_candidate_summary_has_no_applicable_state_diff() -> None:
    proposal = DeterministicActionProposalFactory().build(
        agent_id="rowan",
        scenario_id="unsafe_force_open_safe",
    )
    candidate = action_proposal_to_event_candidate(
        proposal,
        scenario_id="unsafe_force_open_safe",
        candidate_kind="unsafe_access_attempt",
    )

    summary = event_candidate_summary(candidate, candidate_kind="unsafe_access_attempt")

    assert summary.can_modify_world_state is False
    assert summary.predicted_state_diff_id is None
    assert "StateDiff" not in summary.model_dump_json()
