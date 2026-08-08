from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from aethelis.agents.action_proposal import (
    ActionProposalEngine,
    ProposalSourceMode,
    ProposalBehaviorDecision,
    ProviderProposalFailureCode,
    repair_action_proposal,
)
from aethelis.agents.context import CognitionContext, ObservationContext
from aethelis.agents.retrieval import CognitionRetriever
from aethelis.config.settings import Settings
from aethelis.events.commit import build_committed_event_from_verification
from aethelis.events.conversion import (
    CandidateBehaviorRoute,
    action_proposal_to_event_candidate,
    candidate_behavior_route,
    candidate_gate_verification_result,
)
from aethelis.llm.base import LLMProvider, StructuredLLMResult
from aethelis.llm.openai_compatible import OpenAICompatibleLLMProvider
from aethelis.providers import ProviderError
from aethelis.runtime.player_input import (
    PlayerClaimAssessment,
    assess_player_claim,
    route_player_input_scenario,
)
from aethelis.runtime.scenario_matrix import (
    deterministic_scenario_ids,
    get_player_input_fixture_contract,
    get_scenario_definition,
    get_verifier_rule_pack,
    player_claim_scenario_ids,
    player_request_scenario_ids,
    real_llm_scenario_ids,
)
from aethelis.runtime.state_apply import ControlledStateDiffApplier, StateApplyReport
from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.events import (
    ActionProposal,
    ActionProposalSummary,
    CommittedEvent,
    EventCandidate,
    EventCandidateStatus,
    EventCandidateSummary,
    VerificationDecision,
    VerificationResult,
)
from aethelis.schemas.evolution import EvolutionUpdateSummary
from aethelis.schemas.seed import SeedBundle
from aethelis.schemas.world import WorldState
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator
from aethelis.utils.redaction import redact_text
from aethelis.verification.deterministic import DeterministicVerifier

REAL_LLM_SCENARIOS = real_llm_scenario_ids()
DETERMINISTIC_SCENARIOS = deterministic_scenario_ids()
PLAYER_CLAIM_SCENARIOS = player_claim_scenario_ids()
PLAYER_REQUEST_SCENARIOS = player_request_scenario_ids()


class SingleStepResult(AethelisModel):
    scenario_id: Identifier
    agent_id: Identifier
    dry_run: bool = True
    state_diff_applied: bool = False
    action_proposal: ActionProposal | None = None
    event_candidate: EventCandidate | None = None
    verification_result: VerificationResult | None = None
    committed_event: CommittedEvent | None = None
    apply_report: StateApplyReport | None = None
    applied_world_state: WorldState | None = None
    player_claim: PlayerClaimAssessment | None = None
    player_input_summary: dict[str, object] | None = None
    observation_summary: dict[str, object] | None = None
    retrieval_summary: dict[str, object] | None = None
    proposal_source: Identifier | None = None
    provider_mode: Identifier | None = None
    fallback_used: bool = False
    fallback_reason: Identifier | None = None
    evidence_class: Identifier | None = None
    structured_output: StructuredLLMResult[ActionProposal] | None = None
    provider_called: bool = False
    failure_code: ProviderProposalFailureCode | None = None
    proposal_behavior_score: float | None = None
    proposal_behavior_decision: Identifier | None = None
    proposal_behavior_risk_flags: tuple[Identifier, ...] = ()
    candidate_behavior_route: Identifier | None = None
    candidate_behavior_score: float | None = None
    evolution_update: EvolutionUpdateSummary | None = None
    error: str | None = None

    def safe_summary(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "agent_id": self.agent_id,
            "dry_run": self.dry_run,
            "state_diff_applied": self.state_diff_applied,
            "model_name": (
                self.structured_output.model_name if self.structured_output is not None else None
            ),
            "provider_name": (
                self.structured_output.provider_name if self.structured_output is not None else None
            ),
            "latency_ms": (
                self.structured_output.latency_ms if self.structured_output is not None else None
            ),
            "usage": _usage_from_attempts(self.structured_output),
            "raw_text_sha256": (
                self.structured_output.raw_text_sha256
                if self.structured_output is not None
                else None
            ),
            "action_proposal_id": (
                self.action_proposal.id if self.action_proposal is not None else None
            ),
            "action_proposal_summary": _action_proposal_summary(
                self.action_proposal,
                generated_by=self.proposal_source or "deterministic_fixture",
            ),
            "event_candidate_id": (
                self.event_candidate.id if self.event_candidate is not None else None
            ),
            "event_candidate_summary": _event_candidate_summary(
                self.event_candidate,
                self.scenario_id,
            ),
            "verification": _verification_summary(self.verification_result),
            "committed_event_id": (
                self.committed_event.id if self.committed_event is not None else None
            ),
            "state_diff_id": (
                self.committed_event.state_diff.id if self.committed_event is not None else None
            ),
            "state_diff": _state_diff_summary(self.committed_event),
            "apply_report": (
                self.apply_report.safe_dict() if self.apply_report is not None else None
            ),
            "player_claim": _player_claim_summary(self.player_claim),
            "player_input_summary": self.player_input_summary,
            "observation_summary": self.observation_summary,
            "retrieval_summary": self.retrieval_summary,
            "proposal_source": self.proposal_source,
            "provider_mode": self.provider_mode,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "evidence_class": self.evidence_class,
            "provider_called": self.provider_called,
            "failure_code": (self.failure_code.value if self.failure_code is not None else None),
            "proposal_behavior_score": self.proposal_behavior_score,
            "proposal_behavior_decision": self.proposal_behavior_decision,
            "proposal_behavior_risk_flags": list(self.proposal_behavior_risk_flags),
            "candidate_behavior_route": self.candidate_behavior_route,
            "candidate_behavior_score": self.candidate_behavior_score,
            "evolution_update": (
                self.evolution_update.safe_dict() if self.evolution_update is not None else None
            ),
            "structured_error": _structured_error(self.structured_output),
            "error": redact_text(self.error) if self.error is not None else None,
        }


class StepContextSnapshot(AethelisModel):
    observation: ObservationContext
    cognition: CognitionContext
    retrieval_summary: dict[str, object] | None = None
    prompt: str = Field(min_length=1)


def run_single_step(
    *,
    seed_path: Path,
    agent_id: str,
    scenario_id: str,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    proposal_source: ProposalSourceMode | str = ProposalSourceMode.PROVIDER_STRUCTURED,
    provider_proposals_enabled: bool = True,
    allow_real_provider: bool = True,
    apply: bool = False,
    world_state_override: WorldState | None = None,
    pressure_context: dict[str, object] | None = None,
    evolution_context: dict[str, object] | None = None,
) -> SingleStepResult:
    proposal_mode = ProposalSourceMode(proposal_source)
    bundle = _load_valid_seed(seed_path)
    if world_state_override is not None:
        bundle = bundle.model_copy(update={"world": world_state_override})
    if scenario_id in DETERMINISTIC_SCENARIOS:
        return _run_deterministic_scenario(
            bundle=bundle,
            agent_id=agent_id,
            scenario_id=scenario_id,
            apply=apply,
            pressure_context=pressure_context,
            evolution_context=evolution_context,
        )
    if scenario_id not in REAL_LLM_SCENARIOS:
        raise ValueError(f"Unknown scenario_id: {scenario_id}")
    if (
        proposal_mode != ProposalSourceMode.PROVIDER_STRUCTURED
        or not provider_proposals_enabled
        or not allow_real_provider
    ):
        return _provider_failure_result(
            scenario_id=scenario_id,
            agent_id=agent_id,
            apply=apply,
            proposal_source=proposal_mode,
            failure_code=ProviderProposalFailureCode.DISABLED_BY_CONFIG,
            error="Provider proposal path disabled by config.",
        )

    context = build_step_context(
        bundle,
        agent_id=agent_id,
        scenario_id=scenario_id,
        pressure_context=pressure_context,
        evolution_context=evolution_context,
    )
    if provider is None and settings is None:
        return _provider_failure_result(
            scenario_id=scenario_id,
            agent_id=agent_id,
            apply=apply,
            proposal_source=proposal_mode,
            failure_code=ProviderProposalFailureCode.PROVIDER_UNAVAILABLE,
            error="Provider settings unavailable.",
            observation_summary=context.observation.compact_prompt_dict(),
            retrieval_summary=context.retrieval_summary,
        )
    owns_provider = provider is None
    llm_provider = provider or OpenAICompatibleLLMProvider(settings)
    try:
        generated = ActionProposalEngine().generate_structured(
            provider=llm_provider,
            prompt=context.prompt,
            max_tokens=260,
            temperature=0.0,
        )
    except ProviderError as exc:
        return _provider_failure_result(
            scenario_id=scenario_id,
            agent_id=agent_id,
            apply=apply,
            proposal_source=proposal_mode,
            failure_code=ProviderProposalFailureCode.PROVIDER_UNAVAILABLE,
            error=str(exc),
            observation_summary=context.observation.compact_prompt_dict(),
            retrieval_summary=context.retrieval_summary,
            provider_called=True,
        )
    finally:
        if owns_provider and hasattr(llm_provider, "close"):
            llm_provider.close()

    if generated.proposal is None:
        return SingleStepResult(
            scenario_id=scenario_id,
            agent_id=agent_id,
            dry_run=not apply,
            observation_summary=context.observation.compact_prompt_dict(),
            retrieval_summary=context.retrieval_summary,
            proposal_source=generated.source.value,
            provider_mode="real_provider",
            fallback_reason=generated.failure_code.value if generated.failure_code else None,
            evidence_class="internal_failure_path",
            structured_output=generated.structured_output,
            provider_called=generated.provider_called,
            failure_code=generated.failure_code,
            error=generated.error,
        )

    proposal = generated.proposal
    if generated.behavior_decision in {
        ProposalBehaviorDecision.HOLD,
        ProposalBehaviorDecision.REJECT,
    }:
        return SingleStepResult(
            scenario_id=scenario_id,
            agent_id=agent_id,
            dry_run=not apply,
            observation_summary=context.observation.compact_prompt_dict(),
            retrieval_summary=context.retrieval_summary,
            action_proposal=proposal,
            proposal_source=generated.source.value,
            provider_mode="real_provider",
            evidence_class="proposal_behavior_gate",
            structured_output=generated.structured_output,
            provider_called=generated.provider_called,
            failure_code=ProviderProposalFailureCode.UNSAFE_CONTENT_OR_RAW_TEXT_BLOCKED,
            error=f"ActionProposal behavior gate={generated.behavior_decision.value}.",
        )
    if generated.behavior_decision == ProposalBehaviorDecision.REPAIR:
        proposal = repair_action_proposal(proposal)
    candidate = action_proposal_to_event_candidate(
        proposal,
        scenario_id=scenario_id,
        candidate_kind=_candidate_kind(scenario_id),
    )
    candidate_route, candidate_quality = candidate_behavior_route(proposal)
    if candidate_route != CandidateBehaviorRoute.UNDER_REVIEW:
        verification = candidate_gate_verification_result(
            candidate,
            route=candidate_route,
            quality=candidate_quality,
        )
        return SingleStepResult(
            scenario_id=scenario_id,
            agent_id=agent_id,
            dry_run=not apply,
            action_proposal=proposal,
            event_candidate=candidate,
            verification_result=verification,
            observation_summary=context.observation.compact_prompt_dict(),
            retrieval_summary=context.retrieval_summary,
            proposal_behavior_score=generated.behavior_score,
            proposal_behavior_decision=(
                generated.behavior_decision.value if generated.behavior_decision is not None else None
            ),
            proposal_behavior_risk_flags=generated.behavior_risk_flags,
            candidate_behavior_route=candidate_route,
            candidate_behavior_score=candidate_quality,
            proposal_source=generated.source.value,
            provider_mode="real_provider",
            evidence_class="candidate_behavior_gate",
            structured_output=generated.structured_output,
            provider_called=generated.provider_called,
            failure_code=ProviderProposalFailureCode.SCHEMA_VALIDATION_FAILED,
            error=f"EventCandidate behavior quality gate route={candidate_route}.",
        )
    verification = DeterministicVerifier().verify(
        bundle=bundle,
        observation=context.observation,
        cognition=context.cognition,
        proposal=proposal,
        candidate=candidate,
        scenario_id=scenario_id,
    )
    committed_event = None
    apply_report = None
    applied_world_state = None
    if verification.decision == VerificationDecision.COMMIT:
        committed_event = build_committed_event(
            candidate=candidate,
            verification=verification,
            scenario_id=scenario_id,
            world_state=bundle.world,
        )
        if apply and committed_event is not None:
            applied_world_state, apply_report = ControlledStateDiffApplier().apply(
                world_state=bundle.world,
                committed_event=committed_event,
                verification_result=verification,
            )
    return SingleStepResult(
        scenario_id=scenario_id,
        agent_id=agent_id,
        dry_run=not apply,
        state_diff_applied=bool(apply_report and apply_report.applied),
        action_proposal=proposal,
        event_candidate=candidate,
        verification_result=verification,
        committed_event=committed_event,
        apply_report=apply_report,
        applied_world_state=applied_world_state,
        observation_summary=context.observation.compact_prompt_dict(),
        retrieval_summary=context.retrieval_summary,
        proposal_behavior_score=generated.behavior_score,
        proposal_behavior_decision=(
            generated.behavior_decision.value if generated.behavior_decision is not None else None
        ),
        proposal_behavior_risk_flags=generated.behavior_risk_flags,
        candidate_behavior_route=candidate_route,
        candidate_behavior_score=candidate_quality,
        proposal_source=generated.source.value,
        provider_mode="real_provider",
        evidence_class="real_provider",
        structured_output=generated.structured_output,
        provider_called=generated.provider_called,
    )


def _provider_failure_result(
    *,
    scenario_id: str,
    agent_id: str,
    apply: bool,
    proposal_source: ProposalSourceMode,
    failure_code: ProviderProposalFailureCode,
    error: str,
    observation_summary: dict[str, object] | None = None,
    retrieval_summary: dict[str, object] | None = None,
    provider_called: bool = False,
) -> SingleStepResult:
    return SingleStepResult(
        scenario_id=scenario_id,
        agent_id=agent_id,
        dry_run=not apply,
        observation_summary=observation_summary,
        retrieval_summary=retrieval_summary,
        proposal_source=proposal_source.value,
        provider_mode="fallback",
        fallback_used=True,
        fallback_reason=failure_code.value,
        evidence_class="fallback",
        provider_called=provider_called,
        failure_code=failure_code,
        error=error,
    )


def build_step_context(
    bundle: SeedBundle,
    *,
    agent_id: str,
    scenario_id: str,
    pressure_context: dict[str, object] | None = None,
    evolution_context: dict[str, object] | None = None,
) -> StepContextSnapshot:
    retrieved = CognitionRetriever().retrieve(
        bundle,
        agent_id=agent_id,
        scenario_id=scenario_id,
        pressure_context=pressure_context,
        evolution_context=evolution_context,
    )
    return StepContextSnapshot(
        observation=retrieved.observation,
        cognition=retrieved.cognition,
        retrieval_summary=retrieved.summary.safe_summary(),
        prompt=_build_action_prompt(
            retrieved.observation,
            retrieved.cognition,
            scenario_id=scenario_id,
        ),
    )


def build_committed_event(
    *,
    candidate: EventCandidate,
    verification: VerificationResult,
    scenario_id: str,
    world_state: WorldState | None = None,
) -> CommittedEvent | None:
    return build_committed_event_from_verification(
        candidate=candidate,
        verification=verification,
        scenario_id=scenario_id,
        world_state=world_state,
    )


def _run_deterministic_scenario(
    *,
    bundle: SeedBundle,
    agent_id: str,
    scenario_id: str,
    apply: bool,
    pressure_context: dict[str, object] | None = None,
    evolution_context: dict[str, object] | None = None,
) -> SingleStepResult:
    if scenario_id in PLAYER_CLAIM_SCENARIOS:
        return _player_claim_result(agent_id=agent_id, scenario_id=scenario_id, apply=apply)
    if scenario_id in PLAYER_REQUEST_SCENARIOS:
        return _player_input_result(
            agent_id=agent_id,
            scenario_id=scenario_id,
            apply=apply,
        )

    context = build_step_context(
        bundle,
        agent_id=agent_id,
        scenario_id=scenario_id,
        pressure_context=pressure_context,
        evolution_context=evolution_context,
    )
    generated = ActionProposalEngine().generate_deterministic(
        agent_id=agent_id,
        scenario_id=scenario_id,
    )
    if generated.proposal is None:
        raise ValueError(f"Deterministic proposal generation failed: {scenario_id}")
    proposal = generated.proposal
    if generated.behavior_decision in {
        ProposalBehaviorDecision.HOLD,
        ProposalBehaviorDecision.REJECT,
    }:
        return SingleStepResult(
            scenario_id=scenario_id,
            agent_id=agent_id,
            dry_run=not apply,
            action_proposal=proposal,
            observation_summary=context.observation.compact_prompt_dict(),
            retrieval_summary=context.retrieval_summary,
            proposal_source=generated.source.value,
            provider_mode="deterministic_baseline",
            evidence_class="proposal_behavior_gate",
            failure_code=ProviderProposalFailureCode.UNSAFE_CONTENT_OR_RAW_TEXT_BLOCKED,
            error=f"ActionProposal behavior gate={generated.behavior_decision.value}.",
        )
    if generated.behavior_decision == ProposalBehaviorDecision.REPAIR:
        proposal = repair_action_proposal(proposal)
    candidate = action_proposal_to_event_candidate(
        proposal,
        scenario_id=scenario_id,
        candidate_kind=_candidate_kind(scenario_id),
    )
    candidate_route, candidate_quality = candidate_behavior_route(proposal)
    if candidate_route != CandidateBehaviorRoute.UNDER_REVIEW:
        verification = candidate_gate_verification_result(
            candidate,
            route=candidate_route,
            quality=candidate_quality,
        )
        return SingleStepResult(
            scenario_id=scenario_id,
            agent_id=agent_id,
            dry_run=not apply,
            action_proposal=proposal,
            event_candidate=candidate,
            verification_result=verification,
            observation_summary=context.observation.compact_prompt_dict(),
            retrieval_summary=context.retrieval_summary,
            proposal_behavior_score=generated.behavior_score,
            proposal_behavior_decision=(
                generated.behavior_decision.value if generated.behavior_decision is not None else None
            ),
            proposal_behavior_risk_flags=generated.behavior_risk_flags,
            candidate_behavior_route=candidate_route,
            candidate_behavior_score=candidate_quality,
            proposal_source=generated.source.value,
            provider_mode="deterministic_baseline",
            evidence_class="candidate_behavior_gate",
            failure_code=ProviderProposalFailureCode.SCHEMA_VALIDATION_FAILED,
            error=f"EventCandidate behavior quality gate route={candidate_route}.",
        )
    verification = DeterministicVerifier().verify(
        bundle=bundle,
        observation=context.observation,
        cognition=context.cognition,
        proposal=proposal,
        candidate=candidate,
        scenario_id=scenario_id,
    )
    committed_event = None
    apply_report = None
    applied_world_state = None
    if verification.decision == VerificationDecision.COMMIT:
        committed_event = build_committed_event(
            candidate=candidate,
            verification=verification,
            scenario_id=scenario_id,
            world_state=bundle.world,
        )
        if apply and committed_event is not None:
            applied_world_state, apply_report = ControlledStateDiffApplier().apply(
                world_state=bundle.world,
                committed_event=committed_event,
                verification_result=verification,
            )
    return SingleStepResult(
        scenario_id=scenario_id,
        agent_id=agent_id,
        dry_run=not apply,
        state_diff_applied=bool(apply_report and apply_report.applied),
        action_proposal=proposal,
        event_candidate=candidate,
        verification_result=verification,
        committed_event=committed_event,
        apply_report=apply_report,
        applied_world_state=applied_world_state,
        observation_summary=context.observation.compact_prompt_dict(),
        retrieval_summary=context.retrieval_summary,
        proposal_behavior_score=generated.behavior_score,
        proposal_behavior_decision=(
            generated.behavior_decision.value if generated.behavior_decision is not None else None
        ),
        proposal_behavior_risk_flags=generated.behavior_risk_flags,
        candidate_behavior_route=candidate_route,
        candidate_behavior_score=candidate_quality,
        proposal_source=generated.source.value,
        provider_mode="deterministic_baseline",
        evidence_class="deterministic_baseline",
        error=None,
    )


def _player_claim_result(*, agent_id: str, scenario_id: str, apply: bool) -> SingleStepResult:
    if agent_id != "player":
        raise ValueError(f"{scenario_id} requires --agent player")
    contract = get_player_input_fixture_contract(scenario_id)
    claim_id = contract.input_id.replace("player_claim_", "claim_player_", 1)
    assessment = assess_player_claim(
        claim_id=claim_id,
        player_id="player",
        claim=contract.text,
    )
    candidate = EventCandidate(
        id=assessment.claim_id,
        source_action_proposal_id="player_input_claim",
        actor_agent_id="player",
        summary=f"Player claims: {_summarize_text(contract.text)}",
        status=EventCandidateStatus.REJECTED,
    )
    return SingleStepResult(
        scenario_id=scenario_id,
        agent_id=agent_id,
        dry_run=not apply,
        state_diff_applied=False,
        event_candidate=candidate,
        verification_result=assessment.verification_result,
        player_claim=assessment,
        provider_mode="deterministic_baseline",
        evidence_class="deterministic_baseline",
        player_input_summary=(
            assessment.routed_input.safe_summary() if assessment.routed_input is not None else None
        ),
    )


def _player_input_result(*, agent_id: str, scenario_id: str, apply: bool) -> SingleStepResult:
    if agent_id != "player":
        raise ValueError(f"{scenario_id} requires --agent player")
    routed = route_player_input_scenario(scenario_id=scenario_id, player_id=agent_id)
    return SingleStepResult(
        scenario_id=scenario_id,
        agent_id=agent_id,
        dry_run=not apply,
        state_diff_applied=False,
        event_candidate=routed.event_candidate,
        verification_result=routed.verification_result,
        provider_mode="deterministic_baseline",
        evidence_class="deterministic_baseline",
        player_input_summary=routed.safe_summary(),
    )


def _load_valid_seed(seed_path: Path) -> SeedBundle:
    load_result = SeedLoader().load(seed_path)
    report = SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )
    if not report.success or load_result.bundle is None:
        raise ValueError(f"Seed validation failed: {report.safe_dict()}")
    return load_result.bundle


def _build_action_prompt(
    observation: ObservationContext,
    cognition: CognitionContext,
    *,
    scenario_id: str,
) -> str:
    rule_pack = get_verifier_rule_pack(scenario_id)
    required_target_location_id = rule_pack.target_location_id or observation.location.id
    required_target_entity_ids = list(rule_pack.target_entity_ids)
    payload = {
        "task": "Generate one ActionProposal JSON for this agent.",
        "scenario": scenario_id,
        "required": {
            "id": f"proposal_{scenario_id}_{cognition.agent.id}",
            "proposer_agent_id": cognition.agent.id,
            "intent": rule_pack.intent.value if rule_pack.intent is not None else "investigate",
            "target_location_id": required_target_location_id,
            "target_entity_ids": required_target_entity_ids,
        },
        "context": {
            "observation": observation.compact_prompt_dict(),
            "cognition": cognition.compact_prompt_dict(),
        },
        "guard": [
            "Use exactly required id/proposer/target values.",
            "ActionProposal only; no EventCandidate, VerificationResult, StateDiff, CanonFact.",
            "Do not claim world or canon changed.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _verification_summary(result: VerificationResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "id": result.id,
        "decision": result.decision.value,
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "message": check.message,
            }
            for check in result.checks
        ],
        "reasons": list(result.reasons),
        "risk_flags": list(result.risk_flags),
        "rejected_claim_ids": list(result.rejected_claim_ids),
    }


def _state_diff_summary(event: CommittedEvent | None) -> dict[str, object] | None:
    if event is None:
        return None
    return {
        "id": event.state_diff.id,
        "source_event_candidate_id": event.state_diff.source_event_candidate_id,
        "committed_event_id": event.state_diff.committed_event_id,
        "patches": [
            {
                "operation": patch.operation.value,
                "target_type": patch.target_type.value,
                "target_id": patch.target_id,
                "path": patch.path,
                "before": patch.before,
                "after": patch.after,
                "reason": patch.reason,
            }
            for patch in event.state_diff.patches
        ],
    }


def _player_claim_summary(assessment: PlayerClaimAssessment | None) -> dict[str, object] | None:
    if assessment is None:
        return None
    return {
        "claim_id": assessment.claim_id,
        "player_id": assessment.player_id,
        "claim_summary": _summarize_text(assessment.claim, limit=120),
        "canon_updated": assessment.canon_updated,
        "state_diff_id": assessment.state_diff_id,
        "rejected_claim_ids": list(assessment.verification_result.rejected_claim_ids),
    }


def _structured_error(
    output: StructuredLLMResult[ActionProposal] | None,
) -> dict[str, str | None] | None:
    if output is None or output.success:
        return None
    return {
        "json_parse_error": output.json_parse_error,
        "validation_error": output.validation_error,
    }


def _action_proposal_summary(
    proposal: ActionProposal | None,
    *,
    generated_by: str = "deterministic_fixture",
) -> dict[str, object] | None:
    if proposal is None:
        return None
    return ActionProposalSummary.from_proposal(
        proposal,
        generated_by=generated_by,
    ).model_dump(mode="json")


def _event_candidate_summary(
    candidate: EventCandidate | None,
    scenario_id: str,
) -> dict[str, object] | None:
    if candidate is None:
        return None
    return EventCandidateSummary.from_candidate(
        candidate,
        candidate_kind=_candidate_kind(scenario_id),
    ).model_dump(mode="json")


def _candidate_kind(scenario_id: str) -> str | None:
    try:
        return get_scenario_definition(scenario_id).candidate_kind
    except ValueError:
        return None


def _summarize_text(value: str, limit: int = 160) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _usage_from_attempts(output: StructuredLLMResult[ActionProposal] | None) -> dict[str, int]:
    if output is None:
        return {}
    return output.usage
