from __future__ import annotations

from aethelis.runtime.scenario_matrix import get_proposal_fixture_contract
from aethelis.schemas.events import ActionProposal


class DeterministicActionProposalFactory:
    """Build schema-valid deterministic ActionProposal fixtures.

    These fixtures are not LLM output, not facts, and never contain world-state
    mutation payloads.
    """

    def build(self, *, agent_id: str, scenario_id: str) -> ActionProposal:
        contract = get_proposal_fixture_contract(scenario_id)
        return ActionProposal(
            id=contract.proposal_id,
            proposer_agent_id=agent_id,
            intent=contract.intent,
            rationale=contract.rationale,
            target_location_id=contract.target_location_id,
            target_entity_ids=contract.target_entity_ids,
            expected_outcome=contract.expected_outcome,
        )
