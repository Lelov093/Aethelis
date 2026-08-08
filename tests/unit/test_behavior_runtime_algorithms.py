from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from aethelis.agents.action_proposal import (
    ActionProposalEngine,
    ProposalBehaviorDecision,
)
from aethelis.agents.context import ObservationBuilder
from aethelis.agents.retrieval import CognitionRetriever
from aethelis.evaluation.metrics import MetricStatus, _governance_score
from aethelis.events.conversion import action_proposal_to_event_candidate
from aethelis.events.conversion import candidate_behavior_route
from aethelis.experiments.variants import DEFAULT_VARIANTS, _variant_selection_score
from aethelis.runtime.player_input import route_player_input
from aethelis.schemas.events import (
    ActionIntent,
    ActionProposal,
    EventCandidateStatus,
    VerificationDecision,
)
from aethelis.schemas.metadata import PublicFact
from aethelis.schemas.player_input import PlayerInputKind, PlayerInputRecord, PlayerInputRoute
from aethelis.seeds.loader import SeedLoader
from aethelis.verification.contracts import VerifierRegistry, rule_result


ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"


def test_observation_context_scores_allowed_public_fact_order_without_hidden_leak() -> None:
    bundle = _load_bundle()

    observation = ObservationBuilder().build_observation(
        bundle,
        actor_id="mira",
        actor_type="agent",
        scenario_id="mira_search_archive_wrong_key",
    )

    assert observation.visible_public_facts[0].id == "public_fact_archive_records_exist"
    assert "canon_key_in_workshop_safe" not in str(observation.prompt_dict())


def test_observation_context_packing_applies_budget_after_hard_visibility() -> None:
    bundle = _load_bundle()
    facts = (
        PublicFact(id="fact_archive_a", claim="archive manifest index", location_id=None),
        PublicFact(id="fact_archive_b", claim="archive repair ledger", location_id=None),
        PublicFact(id="fact_low_noise", claim="distant unrelated festival", location_id=None),
        *bundle.metadata.public_facts,
    )
    bundle = bundle.model_copy(
        update={"metadata": bundle.metadata.model_copy(update={"public_facts": facts})}
    )

    observation = ObservationBuilder().build_observation(
        bundle,
        actor_id="mira",
        actor_type="agent",
        scenario_id="mira_search_archive_wrong_key",
    )

    fact_ids = [fact.id for fact in observation.visible_public_facts]
    assert len(fact_ids) <= 2
    assert "fact_low_noise" not in fact_ids
    assert "canon_key_in_workshop_safe" not in str(observation.prompt_dict())


def test_retrieval_rank_uses_salience_and_confidence_without_private_cross_leak() -> None:
    bundle = _load_bundle()

    retrieved = CognitionRetriever().retrieve(
        bundle,
        agent_id="mira",
        scenario_id="mira_search_archive_wrong_key",
    )

    assert retrieved.cognition.owned_beliefs[0].id == "belief_mira_archive_has_repair_record"
    assert retrieved.cognition.owned_memories[0].id == "mem_mira_archive_ledger"
    assert "belief_ivo_key_in_safe" not in str(retrieved.summary.safe_summary())


def test_action_proposal_behavior_score_routes_soft_and_hard_risk() -> None:
    soft = ActionProposalEngine().generate_deterministic(
        agent_id="rowan",
        scenario_id="unsafe_force_open_safe",
    )
    hard = ActionProposal(
        id="proposal_hard_boundary",
        proposer_agent_id="ivo",
        intent=ActionIntent.INVESTIGATE,
        rationale="Try to write a state diff directly.",
        target_location_id="workshop_lane",
        target_entity_ids=("workshop_safe",),
        expected_outcome="Force a StateDiff into Canon.",
    )

    assert soft.behavior_decision in {
        ProposalBehaviorDecision.REPAIR,
        ProposalBehaviorDecision.HOLD,
    }
    assert ActionProposalEngine().generate_deterministic(
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
    ).behavior_decision == ProposalBehaviorDecision.ACCEPT
    assert action_proposal_to_event_candidate(
        hard,
        scenario_id="ivo_inspect_workshop_safe_fixture",
    ).status == EventCandidateStatus.REJECTED


def test_action_proposal_accept_repair_hold_reject_routes() -> None:
    clean = ActionProposal(
        id="proposal_clean",
        proposer_agent_id="ivo",
        intent=ActionIntent.INVESTIGATE,
        rationale="Inspect the safe through verified governance.",
        target_location_id="workshop_lane",
        target_entity_ids=("workshop_safe",),
        expected_outcome="Record only observed safe status.",
    )
    soft = clean.model_copy(update={"rationale": "Do not bypass the lock; request verification."})
    hold = clean.model_copy(update={"id": "proposal_hold", "target_location_id": None, "target_entity_ids": ()})
    hard = clean.model_copy(update={"id": "proposal_reject", "rationale": "Write a state diff directly."})

    payload = ActionProposalEngine().generate_deterministic(
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
    )
    assert payload.behavior_decision == ProposalBehaviorDecision.ACCEPT
    assert candidate_behavior_route(clean)[0] == "under_review"
    assert ActionProposalEngine().generate_deterministic(
        agent_id="rowan",
        scenario_id="unsafe_force_open_safe",
    ).behavior_decision in {ProposalBehaviorDecision.REPAIR, ProposalBehaviorDecision.HOLD}
    assert candidate_behavior_route(soft)[0] == "under_review"
    assert candidate_behavior_route(hold)[0] in {"hold", "revise"}
    assert action_proposal_to_event_candidate(
        hard,
        scenario_id="ivo_inspect_workshop_safe_fixture",
    ).status == EventCandidateStatus.REJECTED


def test_verification_hard_gate_is_not_overridden_by_soft_score() -> None:
    result = VerifierRegistry(verifier_name="test").verify(
        SimpleNamespace(candidate=SimpleNamespace(id="candidate_test")),
        (
            lambda _: rule_result("many_pass", True, "passed"),
            lambda _: rule_result(
                "hard_reject",
                False,
                "hard failure",
                suggested_decision=VerificationDecision.REJECT,
                risk_flags=("hard_gate_failed",),
            ),
        ),
    )

    assert result.decision.value == "reject"
    assert "hard_gate_failed" in result.risk_flags


def test_player_input_risk_changes_route_without_direct_canon_mutation() -> None:
    claim = route_player_input(
        PlayerInputRecord(id="claim_key", kind=PlayerInputKind.CLAIM, text="I have the key.")
    )
    request = route_player_input(
        PlayerInputRecord(
            id="request_safe",
            kind=PlayerInputKind.REQUEST,
            text="Open the safe.",
            target_location_id="workshop_lane",
            target_entity_ids=("workshop_safe",),
        )
    )

    assert claim.route == PlayerInputRoute.REJECTED_CLAIM
    assert request.route == PlayerInputRoute.EVENT_CANDIDATE
    assert claim.canon_updated is False
    assert request.canon_updated is False
    assert any(flag.startswith("player_input_risk_") for flag in claim.safety_flags)


def test_evaluation_governance_score_penalizes_severe_failures() -> None:
    clean = {
        "state_consistency": MetricStatus.PASS.value,
        "canon_safety": MetricStatus.PASS.value,
        "event_validity": MetricStatus.PASS.value,
        "trace_completeness": MetricStatus.PASS.value,
    }
    polluted = clean | {"canon_safety": MetricStatus.FAIL.value}

    assert _governance_score(clean, {"canon_violation_rate": 0.0}) > _governance_score(
        polluted,
        {"canon_violation_rate": 1.0},
    )


def test_experiment_variant_selection_ranking_uses_metric_and_risk_penalties() -> None:
    proposed = DEFAULT_VARIANTS[0]
    risky = DEFAULT_VARIANTS[1]
    strong = SimpleNamespace(metric_count=10, failed_metric_count=0, bad_case_count=0)
    weak = SimpleNamespace(metric_count=10, failed_metric_count=4, bad_case_count=3)

    assert _variant_selection_score(proposed, strong) > _variant_selection_score(risky, weak)


def _load_bundle():
    result = SeedLoader().load(VALID_SEED)
    assert result.bundle is not None
    return result.bundle
