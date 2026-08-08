from __future__ import annotations

from typing import Literal

from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.events import ActionIntent, PatchOperation, VerificationDecision
from aethelis.schemas.player_input import PlayerInputKind

ActorType = Literal["agent", "player"]
VerifierRuleKind = Literal[
    "target_match",
    "false_belief_reject",
    "resource_quantity_commit",
    "gated_access",
    "malformed_action",
]
StateDiffContractKind = Literal["safe_inspection", "record_discovery", "resource_quantity"]


class ScenarioDefinition(AethelisModel):
    """Minimal scenario contract for deterministic preview and matrix checks."""

    scenario_id: Identifier
    seed_family: Identifier
    actor_id: Identifier
    actor_type: ActorType
    expected_decision: VerificationDecision
    allows_real_llm: bool
    expects_committed_event: bool
    expects_state_diff: bool
    allows_apply: bool
    is_player_claim: bool
    is_player_input: bool = False
    player_input_kind: Identifier | None = None
    regression_case_id: Identifier
    proposal_fixture_id: Identifier | None = None
    fixture_contract_id: Identifier | None = None
    verifier_rule_pack_id: Identifier
    state_diff_contract_id: Identifier | None = None
    candidate_kind: Identifier | None = None
    description: str


class ProposalFixtureContract(AethelisModel):
    fixture_contract_id: Identifier
    proposal_id: Identifier
    intent: ActionIntent
    rationale: str
    target_location_id: Identifier | None = None
    target_entity_ids: tuple[Identifier, ...] = ()
    expected_outcome: str


class PlayerInputFixtureContract(AethelisModel):
    fixture_contract_id: Identifier
    input_id: Identifier
    kind: PlayerInputKind
    text: str
    target_location_id: Identifier | None = None
    target_entity_ids: tuple[Identifier, ...] = ()


class VerifierRulePack(AethelisModel):
    rule_pack_id: Identifier
    rule_kind: VerifierRuleKind
    rule_id: Identifier
    message: str
    intent: ActionIntent | None = None
    target_location_id: Identifier | None = None
    target_entity_ids: tuple[Identifier, ...] = ()
    canon_fact_id: Identifier | None = None
    canon_object_ids: tuple[Identifier, ...] = ()
    resource_id: Identifier | None = None
    expected_location_id: Identifier | None = None
    suggested_decision: VerificationDecision | None = None
    risk_flags: tuple[Identifier, ...] = ()
    reason: str | None = None


class StateDiffContract(AethelisModel):
    state_diff_contract_id: Identifier
    contract_kind: StateDiffContractKind
    operation: PatchOperation
    target_id: Identifier
    path: str
    before: object
    after: object
    reason: str
    summary: str


RUNTIME_SCENARIO_MATRIX: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(
        scenario_id="inspect_workshop_safe",
        seed_family="mistgate",
        actor_id="ivo",
        actor_type="agent",
        expected_decision=VerificationDecision.COMMIT,
        allows_real_llm=True,
        expects_committed_event=True,
        expects_state_diff=True,
        allows_apply=True,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_commit_inspect_workshop_safe",
        proposal_fixture_id=None,
        fixture_contract_id=None,
        verifier_rule_pack_id="rule_pack_workshop_safe_inspection",
        state_diff_contract_id="state_diff_workshop_safe",
        candidate_kind="safe_inspection",
        description="Ivo inspects the workshop safe and can discover the calibration key.",
    ),
    ScenarioDefinition(
        scenario_id="ivo_inspect_workshop_safe_fixture",
        seed_family="mistgate",
        actor_id="ivo",
        actor_type="agent",
        expected_decision=VerificationDecision.COMMIT,
        allows_real_llm=False,
        expects_committed_event=True,
        expects_state_diff=True,
        allows_apply=True,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_commit_ivo_safe_fixture",
        proposal_fixture_id="fixture_ivo_inspect_workshop_safe",
        fixture_contract_id="fixture_ivo_inspect_workshop_safe",
        verifier_rule_pack_id="rule_pack_workshop_safe_inspection",
        state_diff_contract_id="state_diff_workshop_safe",
        candidate_kind="safe_inspection_fixture",
        description=(
            "Fixture-safe deterministic Ivo safe inspection; does not replace the real LLM path."
        ),
    ),
    ScenarioDefinition(
        scenario_id="mira_search_archive_wrong_key",
        seed_family="mistgate",
        actor_id="mira",
        actor_type="agent",
        expected_decision=VerificationDecision.REJECT,
        allows_real_llm=False,
        expects_committed_event=False,
        expects_state_diff=False,
        allows_apply=False,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_reject_mira_wrong_key",
        proposal_fixture_id="fixture_mira_search_archive_wrong_key",
        fixture_contract_id="fixture_mira_search_archive_wrong_key",
        verifier_rule_pack_id="rule_pack_mistgate_false_belief_reject",
        state_diff_contract_id=None,
        candidate_kind="archive_search",
        description="Mira searches from a false belief; canon key location must not change.",
    ),
    ScenarioDefinition(
        scenario_id="selka_consume_stabilizer_part_fixture",
        seed_family="mistgate",
        actor_id="selka",
        actor_type="agent",
        expected_decision=VerificationDecision.COMMIT,
        allows_real_llm=False,
        expects_committed_event=True,
        expects_state_diff=True,
        allows_apply=True,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_commit_selka_consume_part",
        proposal_fixture_id="fixture_selka_consume_stabilizer_part",
        fixture_contract_id="fixture_selka_consume_stabilizer_part",
        verifier_rule_pack_id="rule_pack_stabilizer_parts_quantity",
        state_diff_contract_id="state_diff_stabilizer_parts_consume",
        candidate_kind="resource_quantity_decrement",
        description="Selka verifies a stabilizer part is spent, covering resource decrement.",
    ),
    ScenarioDefinition(
        scenario_id="selka_restock_market_credit_fixture",
        seed_family="mistgate",
        actor_id="selka",
        actor_type="agent",
        expected_decision=VerificationDecision.COMMIT,
        allows_real_llm=False,
        expects_committed_event=True,
        expects_state_diff=True,
        allows_apply=True,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_commit_selka_restock_credit",
        proposal_fixture_id="fixture_selka_restock_market_credit",
        fixture_contract_id="fixture_selka_restock_market_credit",
        verifier_rule_pack_id="rule_pack_market_credit_quantity",
        state_diff_contract_id="state_diff_market_credit_restock",
        candidate_kind="resource_quantity_increment",
        description="Selka verifies market credit is restored, covering resource increment.",
    ),
    ScenarioDefinition(
        scenario_id="malformed_or_incomplete_action",
        seed_family="mistgate",
        actor_id="ivo",
        actor_type="agent",
        expected_decision=VerificationDecision.REVISE,
        allows_real_llm=False,
        expects_committed_event=False,
        expects_state_diff=False,
        allows_apply=False,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_revise_incomplete_action",
        proposal_fixture_id="fixture_malformed_or_incomplete_action",
        fixture_contract_id="fixture_malformed_or_incomplete_action",
        verifier_rule_pack_id="rule_pack_malformed_action",
        state_diff_contract_id=None,
        candidate_kind="incomplete_inspection",
        description="Incomplete action proposal requires revision before commit.",
    ),
    ScenarioDefinition(
        scenario_id="unsafe_force_open_safe",
        seed_family="mistgate",
        actor_id="rowan",
        actor_type="agent",
        expected_decision=VerificationDecision.PENDING_GATE,
        allows_real_llm=False,
        expects_committed_event=False,
        expects_state_diff=False,
        allows_apply=False,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_pending_gate_force_open_safe",
        proposal_fixture_id="fixture_unsafe_force_open_safe",
        fixture_contract_id="fixture_unsafe_force_open_safe",
        verifier_rule_pack_id="rule_pack_workshop_safe_gate",
        state_diff_contract_id=None,
        candidate_kind="unsafe_access_attempt",
        description="High-impact safe access requires governance gate before commit.",
    ),
    ScenarioDefinition(
        scenario_id="player_claim_key_in_hand",
        seed_family="mistgate",
        actor_id="player",
        actor_type="player",
        expected_decision=VerificationDecision.REJECT,
        allows_real_llm=False,
        expects_committed_event=False,
        expects_state_diff=False,
        allows_apply=False,
        is_player_claim=True,
        is_player_input=True,
        player_input_kind="claim",
        regression_case_id="reg_player_claim_key_in_hand",
        proposal_fixture_id=None,
        fixture_contract_id="player_input_key_claim",
        verifier_rule_pack_id="rule_pack_player_claim_reject",
        state_diff_contract_id=None,
        candidate_kind="player_claim",
        description="Player claim is rejected and must not update canon.",
    ),
    ScenarioDefinition(
        scenario_id="player_request_open_workshop_safe",
        seed_family="mistgate",
        actor_id="player",
        actor_type="player",
        expected_decision=VerificationDecision.PENDING_GATE,
        allows_real_llm=False,
        expects_committed_event=False,
        expects_state_diff=False,
        allows_apply=False,
        is_player_claim=False,
        is_player_input=True,
        player_input_kind="request",
        regression_case_id="reg_player_request_open_workshop_safe",
        proposal_fixture_id=None,
        fixture_contract_id="player_input_workshop_safe_request",
        verifier_rule_pack_id="rule_pack_player_request_gate",
        state_diff_contract_id=None,
        candidate_kind="player_request",
        description=(
            "Player request to open the workshop safe becomes an EventCandidate "
            "and waits at a governance gate."
        ),
    ),
    ScenarioDefinition(
        scenario_id="elin_inspect_cargo_manifest",
        seed_family="harbor_lantern",
        actor_id="elin",
        actor_type="agent",
        expected_decision=VerificationDecision.COMMIT,
        allows_real_llm=True,
        expects_committed_event=True,
        expects_state_diff=True,
        allows_apply=True,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_commit_elin_manifest_real_provider",
        proposal_fixture_id=None,
        fixture_contract_id=None,
        verifier_rule_pack_id="rule_pack_harbor_record_discovery",
        state_diff_contract_id="state_diff_harbor_manifest",
        candidate_kind="record_discovery",
        description=("Elin inspects the cargo manifest through the provider-structured path."),
    ),
    ScenarioDefinition(
        scenario_id="elin_inspect_cargo_manifest_fixture",
        seed_family="harbor_lantern",
        actor_id="elin",
        actor_type="agent",
        expected_decision=VerificationDecision.COMMIT,
        allows_real_llm=False,
        expects_committed_event=True,
        expects_state_diff=True,
        allows_apply=True,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_commit_elin_manifest",
        proposal_fixture_id="fixture_elin_inspect_cargo_manifest",
        fixture_contract_id="fixture_elin_inspect_cargo_manifest",
        verifier_rule_pack_id="rule_pack_harbor_record_discovery",
        state_diff_contract_id="state_diff_harbor_manifest",
        candidate_kind="record_discovery_fixture",
        description="Elin inspects the cargo manifest and can verify harbor pass evidence.",
    ),
    ScenarioDefinition(
        scenario_id="sora_release_relief_crates_fixture",
        seed_family="harbor_lantern",
        actor_id="sora",
        actor_type="agent",
        expected_decision=VerificationDecision.COMMIT,
        allows_real_llm=False,
        expects_committed_event=True,
        expects_state_diff=True,
        allows_apply=True,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_commit_sora_relief_crates",
        proposal_fixture_id="fixture_sora_release_relief_crates",
        fixture_contract_id="fixture_sora_release_relief_crates",
        verifier_rule_pack_id="rule_pack_relief_crates_quantity",
        state_diff_contract_id="state_diff_relief_crates_release",
        candidate_kind="resource_quantity_decrement",
        description="Sora verifies one relief crate is released through governed accounting.",
    ),
    ScenarioDefinition(
        scenario_id="niven_search_lantern_wrong_pass",
        seed_family="harbor_lantern",
        actor_id="niven",
        actor_type="agent",
        expected_decision=VerificationDecision.REJECT,
        allows_real_llm=False,
        expects_committed_event=False,
        expects_state_diff=False,
        allows_apply=False,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_reject_niven_wrong_pass",
        proposal_fixture_id="fixture_niven_search_lantern_wrong_pass",
        fixture_contract_id="fixture_niven_search_lantern_wrong_pass",
        verifier_rule_pack_id="rule_pack_harbor_false_belief_reject",
        state_diff_contract_id=None,
        candidate_kind="false_belief_search",
        description="Niven searches from a false belief; harbor pass canon must not change.",
    ),
    ScenarioDefinition(
        scenario_id="niven_force_quay_lock",
        seed_family="harbor_lantern",
        actor_id="niven",
        actor_type="agent",
        expected_decision=VerificationDecision.PENDING_GATE,
        allows_real_llm=False,
        expects_committed_event=False,
        expects_state_diff=False,
        allows_apply=False,
        is_player_claim=False,
        is_player_input=False,
        regression_case_id="reg_pending_gate_niven_quay_lock",
        proposal_fixture_id="fixture_niven_force_quay_lock",
        fixture_contract_id="fixture_niven_force_quay_lock",
        verifier_rule_pack_id="rule_pack_quay_lock_gate",
        state_diff_contract_id=None,
        candidate_kind="unsafe_access_attempt",
        description="Niven tries to force quay access and must wait at a governance gate.",
    ),
    ScenarioDefinition(
        scenario_id="player_claim_harbor_pass",
        seed_family="harbor_lantern",
        actor_id="player",
        actor_type="player",
        expected_decision=VerificationDecision.REJECT,
        allows_real_llm=False,
        expects_committed_event=False,
        expects_state_diff=False,
        allows_apply=False,
        is_player_claim=True,
        is_player_input=True,
        player_input_kind="claim",
        regression_case_id="reg_player_claim_harbor_pass",
        proposal_fixture_id=None,
        fixture_contract_id="player_input_harbor_pass_claim",
        verifier_rule_pack_id="rule_pack_player_claim_reject",
        state_diff_contract_id=None,
        candidate_kind="player_claim",
        description="Player claim of harbor pass possession is rejected before canon mutation.",
    ),
    ScenarioDefinition(
        scenario_id="player_request_open_quay_gate",
        seed_family="harbor_lantern",
        actor_id="player",
        actor_type="player",
        expected_decision=VerificationDecision.PENDING_GATE,
        allows_real_llm=False,
        expects_committed_event=False,
        expects_state_diff=False,
        allows_apply=False,
        is_player_claim=False,
        is_player_input=True,
        player_input_kind="request",
        regression_case_id="reg_player_request_open_quay_gate",
        proposal_fixture_id=None,
        fixture_contract_id="player_input_quay_gate_request",
        verifier_rule_pack_id="rule_pack_player_request_gate",
        state_diff_contract_id=None,
        candidate_kind="player_request",
        description="Player request to open the quay gate becomes an EventCandidate gate.",
    ),
)


PROPOSAL_FIXTURE_CONTRACTS: tuple[ProposalFixtureContract, ...] = (
    ProposalFixtureContract(
        fixture_contract_id="fixture_ivo_inspect_workshop_safe",
        proposal_id="proposal_ivo_inspect_workshop_safe_fixture",
        intent=ActionIntent.INVESTIGATE,
        rationale="Fixture-safe inspection of Ivo's own workshop safe.",
        target_location_id="workshop_lane",
        target_entity_ids=("workshop_safe",),
        expected_outcome="Inspect the workshop safe for the calibration key.",
    ),
    ProposalFixtureContract(
        fixture_contract_id="fixture_mira_search_archive_wrong_key",
        proposal_id="proposal_mira_search_archive_wrong_key",
        intent=ActionIntent.INVESTIGATE,
        rationale="Mira searches the archive from a known false local belief.",
        target_location_id="central_archive",
        target_entity_ids=("harmonic_tuner",),
        expected_outcome="Search archive records without changing the key canon location.",
    ),
    ProposalFixtureContract(
        fixture_contract_id="fixture_selka_consume_stabilizer_part",
        proposal_id="proposal_selka_consume_stabilizer_part_fixture",
        intent=ActionIntent.TRADE,
        rationale="Verify one stabilizer part was spent through governed trade.",
        target_location_id="market_row",
        expected_outcome="Record one stabilizer part as consumed after verification.",
    ),
    ProposalFixtureContract(
        fixture_contract_id="fixture_selka_restock_market_credit",
        proposal_id="proposal_selka_restock_market_credit_fixture",
        intent=ActionIntent.TRADE,
        rationale="Verify market credit was restored through guild accounting.",
        target_location_id="market_row",
        expected_outcome="Record one market credit as restored after verification.",
    ),
    ProposalFixtureContract(
        fixture_contract_id="fixture_unsafe_force_open_safe",
        proposal_id="proposal_unsafe_force_open_safe",
        intent=ActionIntent.INVESTIGATE,
        rationale="Force open the workshop safe despite access and impact risks.",
        target_location_id="workshop_lane",
        target_entity_ids=("workshop_safe",),
        expected_outcome="Force access to a locked safe.",
    ),
    ProposalFixtureContract(
        fixture_contract_id="fixture_elin_inspect_cargo_manifest",
        proposal_id="proposal_elin_inspect_cargo_manifest_fixture",
        intent=ActionIntent.INVESTIGATE,
        rationale="Fixture-safe inspection of local harbor manifest records.",
        target_location_id="ledger_house",
        target_entity_ids=("cargo_manifest",),
        expected_outcome="Inspect the cargo manifest for harbor pass evidence.",
    ),
    ProposalFixtureContract(
        fixture_contract_id="fixture_sora_release_relief_crates",
        proposal_id="proposal_sora_release_relief_crates_fixture",
        intent=ActionIntent.TRADE,
        rationale="Verify one relief crate is released through governed accounting.",
        target_location_id="quay_gate",
        expected_outcome="Record one relief crate as released after verification.",
    ),
    ProposalFixtureContract(
        fixture_contract_id="fixture_niven_search_lantern_wrong_pass",
        proposal_id="proposal_niven_search_lantern_wrong_pass",
        intent=ActionIntent.INVESTIGATE,
        rationale="Niven searches the lantern room from a false harbor-pass belief.",
        target_location_id="lantern_room",
        target_entity_ids=("lantern_console",),
        expected_outcome="Search local signal equipment without changing pass canon.",
    ),
    ProposalFixtureContract(
        fixture_contract_id="fixture_niven_force_quay_lock",
        proposal_id="proposal_niven_force_quay_lock",
        intent=ActionIntent.INVESTIGATE,
        rationale="Force the quay lock despite clearance and access risks.",
        target_location_id="quay_gate",
        target_entity_ids=("quay_lock",),
        expected_outcome="Force access to the quay gate.",
    ),
    ProposalFixtureContract(
        fixture_contract_id="fixture_malformed_or_incomplete_action",
        proposal_id="proposal_malformed_or_incomplete_action",
        intent=ActionIntent.INVESTIGATE,
        rationale="The proposal remains underspecified for verification.",
        target_location_id="workshop_lane",
        expected_outcome="Needs revision before verification can commit it.",
    ),
)


PLAYER_INPUT_FIXTURE_CONTRACTS: tuple[PlayerInputFixtureContract, ...] = (
    PlayerInputFixtureContract(
        fixture_contract_id="player_input_key_claim",
        input_id="player_claim_key_in_hand",
        kind=PlayerInputKind.CLAIM,
        text="The calibration key is in my hand.",
        target_entity_ids=("calibration_key",),
    ),
    PlayerInputFixtureContract(
        fixture_contract_id="player_input_workshop_safe_request",
        input_id="player_request_open_workshop_safe",
        kind=PlayerInputKind.REQUEST,
        text="Please let me open the workshop safe.",
        target_location_id="workshop_lane",
        target_entity_ids=("workshop_safe",),
    ),
    PlayerInputFixtureContract(
        fixture_contract_id="player_input_harbor_pass_claim",
        input_id="player_claim_harbor_pass",
        kind=PlayerInputKind.CLAIM,
        text="I already have the harbor pass.",
        target_entity_ids=("harbor_pass",),
    ),
    PlayerInputFixtureContract(
        fixture_contract_id="player_input_quay_gate_request",
        input_id="player_request_open_quay_gate",
        kind=PlayerInputKind.REQUEST,
        text="Please open the quay gate for me.",
        target_location_id="quay_gate",
        target_entity_ids=("quay_lock",),
    ),
)


VERIFIER_RULE_PACKS: tuple[VerifierRulePack, ...] = (
    VerifierRulePack(
        rule_pack_id="rule_pack_workshop_safe_inspection",
        rule_kind="target_match",
        rule_id="safe_inspection_commit_eligibility",
        message="Safe workshop inspection must target workshop_safe at workshop_lane.",
        intent=ActionIntent.INVESTIGATE,
        target_location_id="workshop_lane",
        target_entity_ids=("workshop_safe",),
    ),
    VerifierRulePack(
        rule_pack_id="rule_pack_harbor_record_discovery",
        rule_kind="target_match",
        rule_id="record_discovery_commit_eligibility",
        message="Harbor record discovery must inspect cargo_manifest at ledger_house.",
        intent=ActionIntent.INVESTIGATE,
        target_location_id="ledger_house",
        target_entity_ids=("cargo_manifest",),
    ),
    VerifierRulePack(
        rule_pack_id="rule_pack_mistgate_false_belief_reject",
        rule_kind="false_belief_reject",
        rule_id="belief_not_canon",
        message="False belief cannot rewrite canon location.",
        canon_fact_id="canon_key_in_workshop_safe",
        canon_object_ids=("workshop_safe",),
        suggested_decision=VerificationDecision.REJECT,
        risk_flags=("false_belief_no_canon_mutation",),
        reason=(
            "Action can be observed as a search, but it cannot produce a canon-location StateDiff."
        ),
    ),
    VerifierRulePack(
        rule_pack_id="rule_pack_harbor_false_belief_reject",
        rule_kind="false_belief_reject",
        rule_id="belief_not_canon",
        message="False belief cannot rewrite canon location.",
        canon_fact_id="canon_harbor_pass_in_manifest",
        canon_object_ids=("cargo_manifest",),
        suggested_decision=VerificationDecision.REJECT,
        risk_flags=("false_belief_no_canon_mutation",),
        reason=(
            "Action can be observed as a search, but it cannot produce a canon-location StateDiff."
        ),
    ),
    VerifierRulePack(
        rule_pack_id="rule_pack_stabilizer_parts_quantity",
        rule_kind="resource_quantity_commit",
        rule_id="resource_quantity_commit_eligibility",
        message="Verified resource update must be a trade action at market_row.",
        intent=ActionIntent.TRADE,
        expected_location_id="market_row",
        resource_id="stabilizer_parts",
    ),
    VerifierRulePack(
        rule_pack_id="rule_pack_market_credit_quantity",
        rule_kind="resource_quantity_commit",
        rule_id="resource_quantity_commit_eligibility",
        message="Verified resource update must be a trade action at market_row.",
        intent=ActionIntent.TRADE,
        expected_location_id="market_row",
        resource_id="market_credit",
    ),
    VerifierRulePack(
        rule_pack_id="rule_pack_relief_crates_quantity",
        rule_kind="resource_quantity_commit",
        rule_id="resource_quantity_commit_eligibility",
        message="Verified resource update must be a trade action at quay_gate.",
        intent=ActionIntent.TRADE,
        expected_location_id="quay_gate",
        resource_id="relief_crates",
    ),
    VerifierRulePack(
        rule_pack_id="rule_pack_workshop_safe_gate",
        rule_kind="gated_access",
        rule_id="high_impact_event_requires_gate",
        message="Forcing a locked safe is high-impact and requires a gate.",
        suggested_decision=VerificationDecision.PENDING_GATE,
        risk_flags=("high_impact_event_requires_gate", "unsafe_access_attempt"),
        reason="High-impact access requires explicit governance gate before commit.",
    ),
    VerifierRulePack(
        rule_pack_id="rule_pack_quay_lock_gate",
        rule_kind="gated_access",
        rule_id="high_impact_event_requires_gate",
        message="Forcing quay access is high-impact and requires a gate.",
        suggested_decision=VerificationDecision.PENDING_GATE,
        risk_flags=("high_impact_event_requires_gate", "unsafe_access_attempt"),
        reason="Restricted quay access requires explicit governance gate before commit.",
    ),
    VerifierRulePack(
        rule_pack_id="rule_pack_malformed_action",
        rule_kind="malformed_action",
        rule_id="target_entity_present",
        message="Proposal must specify a concrete target_entity_id before commit.",
        suggested_decision=VerificationDecision.REVISE,
        risk_flags=("incomplete_action_proposal",),
        reason="Revision required: specify target entity and actionable intent.",
    ),
    VerifierRulePack(
        rule_pack_id="rule_pack_player_claim_reject",
        rule_kind="gated_access",
        rule_id="player_claim_rejected_before_canon",
        message="Player claim remains non-canon before verification.",
        suggested_decision=VerificationDecision.REJECT,
        risk_flags=("unverified_player_claim",),
    ),
    VerifierRulePack(
        rule_pack_id="rule_pack_player_request_gate",
        rule_kind="gated_access",
        rule_id="player_request_event_candidate_gate",
        message="Player request remains an EventCandidate before commit.",
        suggested_decision=VerificationDecision.PENDING_GATE,
        risk_flags=("player_input_requires_gate",),
    ),
)


STATE_DIFF_CONTRACTS: tuple[StateDiffContract, ...] = (
    StateDiffContract(
        state_diff_contract_id="state_diff_workshop_safe",
        contract_kind="safe_inspection",
        operation=PatchOperation.APPEND,
        target_id="calibration_key",
        path="/resource/calibration_key/discovery_state/discovered_by_agent_ids",
        before=[],
        after=["ivo"],
        reason=(
            "Dry-run diff: Ivo lawfully inspected workshop_safe and discovered "
            "the calibration key. The WorldState is not applied unless --apply is set."
        ),
        summary="Ivo inspects the workshop safe and discovers the calibration key.",
    ),
    StateDiffContract(
        state_diff_contract_id="state_diff_harbor_manifest",
        contract_kind="record_discovery",
        operation=PatchOperation.APPEND,
        target_id="harbor_pass",
        path="/resource/harbor_pass/discovery_state/discovered_by_agent_ids",
        before=[],
        after=["elin"],
        reason=(
            "Dry-run diff: Elin lawfully inspected cargo_manifest and discovered "
            "harbor pass evidence. The WorldState is not applied unless --apply is set."
        ),
        summary="Elin inspects the cargo manifest and discovers harbor pass evidence.",
    ),
    StateDiffContract(
        state_diff_contract_id="state_diff_stabilizer_parts_consume",
        contract_kind="resource_quantity",
        operation=PatchOperation.DECREMENT,
        target_id="stabilizer_parts",
        path="/resource/stabilizer_parts/quantity",
        before=3,
        after=2,
        reason="Verified governed trade consumes one stabilizer part.",
        summary="Selka verifies one stabilizer part has been consumed.",
    ),
    StateDiffContract(
        state_diff_contract_id="state_diff_market_credit_restock",
        contract_kind="resource_quantity",
        operation=PatchOperation.INCREMENT,
        target_id="market_credit",
        path="/resource/market_credit/quantity",
        before=5,
        after=6,
        reason="Verified guild accounting restores one market credit.",
        summary="Selka verifies one market credit has been restored.",
    ),
    StateDiffContract(
        state_diff_contract_id="state_diff_relief_crates_release",
        contract_kind="resource_quantity",
        operation=PatchOperation.DECREMENT,
        target_id="relief_crates",
        path="/resource/relief_crates/quantity",
        before=4,
        after=3,
        reason="Verified governed release moves one relief crate.",
        summary="Sora verifies one relief crate has been released.",
    ),
)

_SCENARIOS_BY_ID = {scenario.scenario_id: scenario for scenario in RUNTIME_SCENARIO_MATRIX}
_PROPOSAL_FIXTURES_BY_ID = {
    contract.fixture_contract_id: contract for contract in PROPOSAL_FIXTURE_CONTRACTS
}
_PLAYER_INPUT_FIXTURES_BY_ID = {
    contract.fixture_contract_id: contract for contract in PLAYER_INPUT_FIXTURE_CONTRACTS
}
_VERIFIER_RULE_PACKS_BY_ID = {pack.rule_pack_id: pack for pack in VERIFIER_RULE_PACKS}
_STATE_DIFF_CONTRACTS_BY_ID = {
    contract.state_diff_contract_id: contract for contract in STATE_DIFF_CONTRACTS
}


def get_scenario_definition(scenario_id: str) -> ScenarioDefinition:
    try:
        return _SCENARIOS_BY_ID[scenario_id]
    except KeyError:
        raise ValueError(f"Unknown scenario_id: {scenario_id}") from None


def get_proposal_fixture_contract(scenario_id: str) -> ProposalFixtureContract:
    try:
        scenario = get_scenario_definition(scenario_id)
    except ValueError:
        raise ValueError(
            f"Unsupported deterministic ActionProposal scenario: {scenario_id}"
        ) from None
    if scenario.allows_real_llm:
        raise ValueError(f"{scenario_id} requires the explicit real LLM run-step path")
    if scenario.fixture_contract_id is None:
        raise ValueError(f"Unsupported deterministic ActionProposal scenario: {scenario_id}")
    try:
        return _PROPOSAL_FIXTURES_BY_ID[scenario.fixture_contract_id]
    except KeyError:
        raise ValueError(
            f"Missing proposal fixture contract: {scenario.fixture_contract_id}"
        ) from None


def get_player_input_fixture_contract(scenario_id: str) -> PlayerInputFixtureContract:
    try:
        scenario = get_scenario_definition(scenario_id)
    except ValueError:
        raise ValueError(f"Unsupported player input scenario: {scenario_id}") from None
    if scenario.fixture_contract_id is None:
        raise ValueError(f"Unsupported player input scenario: {scenario_id}")
    try:
        return _PLAYER_INPUT_FIXTURES_BY_ID[scenario.fixture_contract_id]
    except KeyError:
        raise ValueError(f"Unsupported player input scenario: {scenario_id}") from None


def get_verifier_rule_pack(scenario_id: str) -> VerifierRulePack:
    scenario = get_scenario_definition(scenario_id)
    return get_verifier_rule_pack_by_id(scenario.verifier_rule_pack_id)


def get_verifier_rule_pack_by_id(rule_pack_id: str) -> VerifierRulePack:
    try:
        return _VERIFIER_RULE_PACKS_BY_ID[rule_pack_id]
    except KeyError:
        raise ValueError(f"Unknown verifier_rule_pack_id: {rule_pack_id}") from None


def get_state_diff_contract(scenario_id: str) -> StateDiffContract | None:
    scenario = get_scenario_definition(scenario_id)
    if scenario.state_diff_contract_id is None:
        return None
    try:
        return _STATE_DIFF_CONTRACTS_BY_ID[scenario.state_diff_contract_id]
    except KeyError:
        raise ValueError(
            f"Unknown state_diff_contract_id: {scenario.state_diff_contract_id}"
        ) from None


def real_llm_scenario_ids() -> frozenset[str]:
    return frozenset(
        scenario.scenario_id for scenario in RUNTIME_SCENARIO_MATRIX if scenario.allows_real_llm
    )


def deterministic_scenario_ids() -> frozenset[str]:
    return frozenset(
        scenario.scenario_id for scenario in RUNTIME_SCENARIO_MATRIX if not scenario.allows_real_llm
    )


def player_claim_scenario_ids() -> frozenset[str]:
    return frozenset(
        scenario.scenario_id for scenario in RUNTIME_SCENARIO_MATRIX if scenario.is_player_claim
    )


def player_request_scenario_ids() -> frozenset[str]:
    return frozenset(
        scenario.scenario_id
        for scenario in RUNTIME_SCENARIO_MATRIX
        if scenario.is_player_input and not scenario.is_player_claim
    )


def scenario_matrix_summary() -> list[dict[str, object]]:
    return [
        {
            "scenario_id": scenario.scenario_id,
            "seed_family": scenario.seed_family,
            "actor_id": scenario.actor_id,
            "actor_type": scenario.actor_type,
            "expected_decision": scenario.expected_decision.value,
            "allows_real_llm": scenario.allows_real_llm,
            "expects_committed_event": scenario.expects_committed_event,
            "expects_state_diff": scenario.expects_state_diff,
            "allows_apply": scenario.allows_apply,
            "is_player_claim": scenario.is_player_claim,
            "is_player_input": scenario.is_player_input,
            "player_input_kind": scenario.player_input_kind,
            "regression_case_id": scenario.regression_case_id,
            "proposal_fixture_id": scenario.proposal_fixture_id,
            "fixture_contract_id": scenario.fixture_contract_id,
            "verifier_rule_pack_id": scenario.verifier_rule_pack_id,
            "state_diff_contract_id": scenario.state_diff_contract_id,
            "candidate_kind": scenario.candidate_kind,
        }
        for scenario in RUNTIME_SCENARIO_MATRIX
    ]
