from __future__ import annotations

from pathlib import Path

from aethelis.agents.context import build_agent_context
from aethelis.events.conversion import action_proposal_to_event_candidate
from aethelis.events.fixtures import DeterministicActionProposalFactory
from aethelis.runtime.single_step import run_single_step
from aethelis.schemas.events import VerificationDecision
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator
from aethelis.verification.contracts import VerifierContext, VerifierRegistry, rule_result
from aethelis.verification.deterministic import DeterministicVerifier

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"
HARBOR_SEED = ROOT / "seeds" / "harbor_lantern_v01"


def load_valid_bundle(seed_path: Path = VALID_SEED):
    load_result = SeedLoader().load(seed_path)
    report = SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )
    assert report.success
    assert load_result.bundle is not None
    return load_result.bundle


def test_verifier_registry_decision_precedence() -> None:
    bundle = load_valid_bundle()
    proposal = DeterministicActionProposalFactory().build(
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
    )
    candidate = action_proposal_to_event_candidate(
        proposal,
        scenario_id="ivo_inspect_workshop_safe_fixture",
    )
    context = VerifierContext(
        bundle=bundle,
        scenario_id="ivo_inspect_workshop_safe_fixture",
        proposal=proposal,
        candidate=candidate,
    )

    result = VerifierRegistry(verifier_name="test_registry").verify(
        context,
        (
            lambda _: rule_result("pass_rule", True, "pass"),
            lambda _: rule_result(
                "gate_rule",
                False,
                "gate",
                suggested_decision=VerificationDecision.PENDING_GATE,
                risk_flags=("test_gate",),
            ),
            lambda _: rule_result(
                "revise_rule",
                False,
                "revise",
                suggested_decision=VerificationDecision.REVISE,
            ),
        ),
    )

    assert result.decision == VerificationDecision.PENDING_GATE
    assert result.risk_flags == ("test_gate",)
    assert [check.name for check in result.checks] == ["pass_rule", "gate_rule", "revise_rule"]


def test_deterministic_commit_fixture_verifies_and_builds_state_diff() -> None:
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
        settings=object(),
        apply=False,
    )

    assert result.structured_output is None
    assert result.verification_result is not None
    assert result.verification_result.verifier == "deterministic_governance_v0"
    assert result.verification_result.decision == VerificationDecision.COMMIT
    check_names = {check.name for check in result.verification_result.checks}
    assert {
        "target_location_exists",
        "proposal_location_matches_observation",
        "resource_update_targets_known_resource",
    }.issubset(check_names)
    assert result.committed_event is not None
    assert result.committed_event.state_diff.source_action_proposal_id is None
    assert result.committed_event.state_diff.committed_event_id == result.committed_event.id
    assert result.state_diff_applied is False


def test_deterministic_verifier_unifies_non_commit_paths() -> None:
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="rowan",
        scenario_id="unsafe_force_open_safe",
        settings=object(),
        apply=True,
    )

    assert result.verification_result is not None
    assert result.verification_result.decision == VerificationDecision.PENDING_GATE
    assert "high_impact_event_requires_gate" in result.verification_result.risk_flags
    checks = {check.name: check.passed for check in result.verification_result.checks}
    assert checks["target_location_exists"] is True
    assert checks["proposal_location_matches_observation"] is False
    assert result.committed_event is None
    assert result.state_diff_applied is False


def test_resource_quantity_verifier_contract_checks_known_resource() -> None:
    result = run_single_step(
        seed_path=VALID_SEED,
        agent_id="selka",
        scenario_id="selka_consume_stabilizer_part_fixture",
        settings=object(),
        apply=False,
    )

    assert result.verification_result is not None
    assert result.verification_result.decision == VerificationDecision.COMMIT
    checks = {check.name: check.passed for check in result.verification_result.checks}
    assert checks["resource_update_targets_known_resource"] is True
    assert checks["target_location_exists"] is True
    assert checks["proposal_location_matches_observation"] is True


def test_deterministic_verifier_direct_context_path() -> None:
    bundle = load_valid_bundle()
    proposal = DeterministicActionProposalFactory().build(
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
    )
    candidate = action_proposal_to_event_candidate(
        proposal,
        scenario_id="ivo_inspect_workshop_safe_fixture",
    )
    observation, cognition = build_agent_context(
        bundle,
        agent_id="ivo",
        scenario_id="ivo_inspect_workshop_safe_fixture",
    )

    verification = DeterministicVerifier().verify(
        bundle=bundle,
        observation=observation,
        cognition=cognition,
        proposal=proposal,
        candidate=candidate,
        scenario_id="ivo_inspect_workshop_safe_fixture",
    )

    assert verification.decision == VerificationDecision.COMMIT
    assert all(check.passed for check in verification.checks)


def test_harbor_deterministic_verifier_decisions() -> None:
    expected = {
        ("elin", "elin_inspect_cargo_manifest_fixture"): VerificationDecision.COMMIT,
        ("sora", "sora_release_relief_crates_fixture"): VerificationDecision.COMMIT,
        ("niven", "niven_search_lantern_wrong_pass"): VerificationDecision.REJECT,
        ("niven", "niven_force_quay_lock"): VerificationDecision.PENDING_GATE,
    }

    for (agent_id, scenario_id), decision in expected.items():
        result = run_single_step(
            seed_path=HARBOR_SEED,
            agent_id=agent_id,
            scenario_id=scenario_id,
            settings=object(),
            apply=False,
        )

        assert result.verification_result is not None
        assert result.verification_result.decision == decision
        assert (result.committed_event is not None) == (decision == VerificationDecision.COMMIT)

    elin = run_single_step(
        seed_path=HARBOR_SEED,
        agent_id="elin",
        scenario_id="elin_inspect_cargo_manifest_fixture",
        settings=object(),
        apply=False,
    )
    sora = run_single_step(
        seed_path=HARBOR_SEED,
        agent_id="sora",
        scenario_id="sora_release_relief_crates_fixture",
        settings=object(),
        apply=False,
    )

    assert elin.committed_event is not None
    assert elin.committed_event.state_diff.patches[0].target_id == "harbor_pass"
    assert sora.committed_event is not None
    assert sora.committed_event.state_diff.patches[0].target_id == "relief_crates"
