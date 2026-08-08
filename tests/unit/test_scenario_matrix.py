from __future__ import annotations

from aethelis.evaluation.regression_cases import (
    default_regression_cases,
    scenario_matrix_regression_cases,
)
from aethelis.runtime.scenario_matrix import (
    RUNTIME_SCENARIO_MATRIX,
    deterministic_scenario_ids,
    get_player_input_fixture_contract,
    get_proposal_fixture_contract,
    get_state_diff_contract,
    get_verifier_rule_pack,
    real_llm_scenario_ids,
    scenario_matrix_summary,
)
from aethelis.schemas.events import VerificationDecision


def test_runtime_scenario_matrix_has_expected_cases() -> None:
    scenarios = {scenario.scenario_id: scenario for scenario in RUNTIME_SCENARIO_MATRIX}

    assert set(scenarios) == {
        "inspect_workshop_safe",
        "ivo_inspect_workshop_safe_fixture",
        "mira_search_archive_wrong_key",
        "selka_consume_stabilizer_part_fixture",
        "selka_restock_market_credit_fixture",
        "malformed_or_incomplete_action",
        "unsafe_force_open_safe",
        "player_claim_key_in_hand",
        "player_request_open_workshop_safe",
        "elin_inspect_cargo_manifest",
        "elin_inspect_cargo_manifest_fixture",
        "sora_release_relief_crates_fixture",
        "niven_search_lantern_wrong_pass",
        "niven_force_quay_lock",
        "player_claim_harbor_pass",
        "player_request_open_quay_gate",
    }
    assert scenarios["inspect_workshop_safe"].expected_decision == VerificationDecision.COMMIT
    assert (
        scenarios["ivo_inspect_workshop_safe_fixture"].expected_decision
        == VerificationDecision.COMMIT
    )
    assert scenarios["ivo_inspect_workshop_safe_fixture"].allows_real_llm is False
    assert scenarios["ivo_inspect_workshop_safe_fixture"].expects_committed_event is True
    assert scenarios["ivo_inspect_workshop_safe_fixture"].expects_state_diff is True
    assert scenarios["ivo_inspect_workshop_safe_fixture"].seed_family == "mistgate"
    assert (
        scenarios["ivo_inspect_workshop_safe_fixture"].fixture_contract_id
        == "fixture_ivo_inspect_workshop_safe"
    )
    assert (
        scenarios["ivo_inspect_workshop_safe_fixture"].verifier_rule_pack_id
        == "rule_pack_workshop_safe_inspection"
    )
    assert (
        scenarios["ivo_inspect_workshop_safe_fixture"].state_diff_contract_id
        == "state_diff_workshop_safe"
    )
    assert scenarios["mira_search_archive_wrong_key"].expected_decision == (
        VerificationDecision.REJECT
    )
    assert scenarios["selka_consume_stabilizer_part_fixture"].expected_decision == (
        VerificationDecision.COMMIT
    )
    assert scenarios["selka_consume_stabilizer_part_fixture"].allows_apply is True
    assert scenarios["selka_consume_stabilizer_part_fixture"].candidate_kind == (
        "resource_quantity_decrement"
    )
    assert scenarios["selka_restock_market_credit_fixture"].expected_decision == (
        VerificationDecision.COMMIT
    )
    assert scenarios["selka_restock_market_credit_fixture"].allows_apply is True
    assert scenarios["selka_restock_market_credit_fixture"].candidate_kind == (
        "resource_quantity_increment"
    )
    assert scenarios["malformed_or_incomplete_action"].expected_decision == (
        VerificationDecision.REVISE
    )
    assert scenarios["unsafe_force_open_safe"].expected_decision == (
        VerificationDecision.PENDING_GATE
    )
    assert scenarios["player_claim_key_in_hand"].actor_type == "player"
    assert scenarios["player_claim_key_in_hand"].is_player_claim is True
    assert scenarios["player_claim_key_in_hand"].is_player_input is True
    assert scenarios["player_request_open_workshop_safe"].actor_type == "player"
    assert scenarios["player_request_open_workshop_safe"].is_player_input is True
    assert scenarios["player_request_open_workshop_safe"].player_input_kind == "request"
    assert scenarios["player_request_open_workshop_safe"].expected_decision == (
        VerificationDecision.PENDING_GATE
    )
    assert scenarios["elin_inspect_cargo_manifest_fixture"].expected_decision == (
        VerificationDecision.COMMIT
    )
    assert scenarios["elin_inspect_cargo_manifest"].allows_real_llm is True
    assert scenarios["elin_inspect_cargo_manifest"].seed_family == "harbor_lantern"
    assert scenarios["elin_inspect_cargo_manifest"].fixture_contract_id is None
    assert (
        scenarios["elin_inspect_cargo_manifest"].verifier_rule_pack_id
        == "rule_pack_harbor_record_discovery"
    )
    assert (
        scenarios["elin_inspect_cargo_manifest"].state_diff_contract_id
        == "state_diff_harbor_manifest"
    )
    assert scenarios["elin_inspect_cargo_manifest_fixture"].seed_family == "harbor_lantern"
    assert scenarios["elin_inspect_cargo_manifest_fixture"].expects_state_diff is True
    assert scenarios["sora_release_relief_crates_fixture"].candidate_kind == (
        "resource_quantity_decrement"
    )
    assert scenarios["niven_search_lantern_wrong_pass"].expected_decision == (
        VerificationDecision.REJECT
    )
    assert scenarios["niven_force_quay_lock"].expected_decision == (
        VerificationDecision.PENDING_GATE
    )
    assert scenarios["player_claim_harbor_pass"].is_player_claim is True
    assert scenarios["player_request_open_quay_gate"].player_input_kind == "request"


def test_scenario_matrix_drives_real_and_deterministic_sets() -> None:
    assert real_llm_scenario_ids() == frozenset(
        {"inspect_workshop_safe", "elin_inspect_cargo_manifest"}
    )
    assert deterministic_scenario_ids() == frozenset(
        {
            "ivo_inspect_workshop_safe_fixture",
            "mira_search_archive_wrong_key",
            "selka_consume_stabilizer_part_fixture",
            "selka_restock_market_credit_fixture",
            "malformed_or_incomplete_action",
            "unsafe_force_open_safe",
            "player_claim_key_in_hand",
            "player_request_open_workshop_safe",
            "elin_inspect_cargo_manifest_fixture",
            "sora_release_relief_crates_fixture",
            "niven_search_lantern_wrong_pass",
            "niven_force_quay_lock",
            "player_claim_harbor_pass",
            "player_request_open_quay_gate",
        }
    )


def test_regression_cases_match_scenario_matrix() -> None:
    matrix_by_case = {scenario.regression_case_id: scenario for scenario in RUNTIME_SCENARIO_MATRIX}
    regression_cases = default_regression_cases()

    assert {case.id for case in regression_cases} == {
        "reg_commit_inspect_workshop_safe",
        "reg_commit_ivo_safe_fixture",
        "reg_reject_mira_wrong_key",
        "reg_commit_selka_consume_part",
        "reg_commit_selka_restock_credit",
        "reg_revise_incomplete_action",
        "reg_pending_gate_force_open_safe",
        "reg_player_claim_key_in_hand",
        "reg_player_request_open_workshop_safe",
    }
    for case in regression_cases:
        scenario = matrix_by_case[case.id]
        assert case.agent_id == scenario.actor_id
        assert case.scenario_id == scenario.scenario_id
        assert case.expected_decision == scenario.expected_decision
        assert case.expects_committed_event == scenario.expects_committed_event
        assert case.expects_state_diff == scenario.expects_state_diff
        assert case.expects_state_diff_applied is False
        if scenario.is_player_input:
            assert case.expects_canon_updated is False


def test_scenario_matrix_regression_cases_include_harbor_expectations() -> None:
    cases = scenario_matrix_regression_cases()

    assert {case.scenario_id for case in cases} == {
        scenario.scenario_id for scenario in RUNTIME_SCENARIO_MATRIX
    }
    assert "elin_inspect_cargo_manifest_fixture" in {case.scenario_id for case in cases}
    assert "player_request_open_quay_gate" in {case.scenario_id for case in cases}


def test_scenario_matrix_summary_is_cli_safe() -> None:
    summary = scenario_matrix_summary()

    assert len(summary) == 16
    assert all("regression_case_id" in row for row in summary)
    assert "sk-" not in str(summary).lower()
    assert "authorization" not in str(summary).lower()


def test_scenario_contracts_cover_fixtures_rule_packs_and_state_diffs() -> None:
    for scenario in RUNTIME_SCENARIO_MATRIX:
        assert get_verifier_rule_pack(scenario.scenario_id).rule_pack_id == (
            scenario.verifier_rule_pack_id
        )
        if scenario.expects_state_diff:
            assert get_state_diff_contract(scenario.scenario_id) is not None
        if scenario.allows_real_llm:
            assert scenario.fixture_contract_id is None
        elif scenario.is_player_input:
            assert get_player_input_fixture_contract(scenario.scenario_id).input_id.startswith(
                "player_"
            )
        else:
            fixture = get_proposal_fixture_contract(scenario.scenario_id)
            assert fixture.fixture_contract_id == scenario.fixture_contract_id
