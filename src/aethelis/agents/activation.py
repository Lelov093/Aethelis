from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aethelis.algorithms.runtime_features import clamp01, weighted_scheduler_score
from aethelis.agents.context import ObservationBuilder, ObservationContext
from aethelis.schemas.activation import (
    ActivationCandidate,
    ActivationReason,
    ActivationResult,
    ActivationStatus,
    AgentActivationConfig,
)
from aethelis.schemas.agents import AgentProfile
from aethelis.schemas.seed import SeedBundle

if TYPE_CHECKING:
    from aethelis.runtime.scenario_matrix import ScenarioDefinition
    from aethelis.schemas.run import RunStepPlanItem


@dataclass(frozen=True)
class ScenarioActivationHints:
    target_location_id: str | None
    action_type: str | None
    role_keywords: tuple[str, ...]
    goal_keywords: tuple[str, ...]
    pressure_types: tuple[str, ...]


SCENARIO_ACTIVATION_HINTS: dict[str, ScenarioActivationHints] = {
    "ivo_inspect_workshop_safe_fixture": ScenarioActivationHints(
        target_location_id="workshop_lane",
        action_type="inspect",
        role_keywords=("workshop", "technician"),
        goal_keywords=("workshop", "safe", "calibration", "key"),
        pressure_types=(),
    ),
    "mira_search_archive_wrong_key": ScenarioActivationHints(
        target_location_id="central_archive",
        action_type="search",
        role_keywords=("archive", "archivist"),
        goal_keywords=("archive", "record", "calibration", "key"),
        pressure_types=(),
    ),
    "malformed_or_incomplete_action": ScenarioActivationHints(
        target_location_id="workshop_lane",
        action_type="inspect",
        role_keywords=("workshop", "technician"),
        goal_keywords=("workshop", "safe", "calibration", "key"),
        pressure_types=(),
    ),
    "unsafe_force_open_safe": ScenarioActivationHints(
        target_location_id="workshop_lane",
        action_type="request_access",
        role_keywords=("guard", "captain"),
        goal_keywords=("safe", "access", "repair"),
        pressure_types=("civic_trust",),
    ),
    "player_claim_key_in_hand": ScenarioActivationHints(
        target_location_id="council_square",
        action_type="report_rumor",
        role_keywords=("player",),
        goal_keywords=("claim", "key"),
        pressure_types=("civic_trust", "rumor_spread"),
    ),
    "player_request_open_workshop_safe": ScenarioActivationHints(
        target_location_id="workshop_lane",
        action_type="request_access",
        role_keywords=("player",),
        goal_keywords=("open", "safe", "workshop"),
        pressure_types=("resource_scarcity", "civic_trust"),
    ),
}


class AgentActivationBuilder:
    """Build layered deterministic scheduler-v0 explanations without executing them."""

    def __init__(self, observation_builder: ObservationBuilder | None = None) -> None:
        self._observation_builder = observation_builder or ObservationBuilder()

    def build_for_step(
        self,
        *,
        bundle: SeedBundle,
        run_id: str,
        step: RunStepPlanItem,
        config: AgentActivationConfig,
        evolution_context: dict[str, object] | None = None,
    ) -> ActivationResult:
        from aethelis.runtime.scenario_matrix import get_scenario_definition

        scenario = get_scenario_definition(step.scenario_id)
        evaluated_candidates = tuple(
            _ranked_candidate(
                bundle=bundle,
                run_id=run_id,
                step=step,
                scenario=scenario,
                config=config,
                observation_builder=self._observation_builder,
                actor_id=actor_id,
                actor_type=actor_type,
                evolution_context=evolution_context,
            )
            for actor_id, actor_type in _candidate_actor_specs(bundle, step.actor_type)
        )
        ranked_candidates = tuple(
            candidate.model_copy(
                update={
                    "top_k_rank": index + 1,
                    "tie_break_key": _tie_break_key(candidate, step),
                }
            )
            for index, candidate in enumerate(
                sorted(
                    evaluated_candidates,
                    key=lambda candidate: (
                        -candidate.score_total,
                        candidate.agent_id != step.agent_id,
                        candidate.agent_id,
                    ),
                )
            )
        )
        selected_candidates = tuple(
            candidate.model_copy(update={"status": ActivationStatus.SELECTED_STATIC_PLAN})
            if index < config.top_k and candidate.threshold_passed
            else candidate.model_copy(
                update={
                    "status": (
                        ActivationStatus.BACKGROUND
                        if candidate.threshold_passed
                        else ActivationStatus.SKIPPED
                    )
                }
            )
            for index, candidate in enumerate(ranked_candidates)
        )
        selected_candidate = next(
            (
                candidate
                for candidate in selected_candidates
                if candidate.status == ActivationStatus.SELECTED_STATIC_PLAN
            ),
            selected_candidates[0],
        )
        projected_candidates = (
            selected_candidates
            if config.include_non_selected_candidates
            else tuple(selected_candidates[: config.top_k])
        )
        return ActivationResult(
            activation_result_id=f"activation_result_{step.step_id}",
            run_id=run_id,
            step_id=step.step_id,
            scenario_id=step.scenario_id,
            mode=config.mode,
            scoring_version=config.scoring_version,
            selected_candidate=selected_candidate,
            candidate_count=len(selected_candidates),
            candidates=projected_candidates,
        )


def build_public_observation_for_activation(
    bundle: SeedBundle,
    *,
    actor_id: str,
    actor_type: str,
    scenario_id: str,
    observation_builder: ObservationBuilder | None = None,
) -> ObservationContext:
    builder = observation_builder or ObservationBuilder()
    return builder.build_observation(
        bundle,
        actor_id=actor_id,
        actor_type=actor_type,
        scenario_id=scenario_id,
    )


def _candidate_actor_specs(bundle: SeedBundle, actor_type: str) -> tuple[tuple[str, str], ...]:
    if actor_type == "player":
        return (("player", "player"),)
    if actor_type == "agent":
        return tuple((agent.id, "agent") for agent in bundle.agents.agents)
    raise ValueError(f"Unsupported actor_type for activation: {actor_type}")


def _ranked_candidate(
    *,
    bundle: SeedBundle,
    run_id: str,
    step: RunStepPlanItem,
    scenario: ScenarioDefinition,
    config: AgentActivationConfig,
    observation_builder: ObservationBuilder,
    actor_id: str,
    actor_type: str,
    evolution_context: dict[str, object] | None,
) -> ActivationCandidate:
    observation = build_public_observation_for_activation(
        bundle,
        actor_id=actor_id,
        actor_type=actor_type,
        scenario_id=step.scenario_id,
        observation_builder=observation_builder,
    )
    agent = _find_agent(bundle, actor_id) if actor_type == "agent" else None
    reasons = _activation_reasons(
        bundle=bundle,
        step=step,
        scenario=scenario,
        observation=observation,
        agent=agent,
        actor_id=actor_id,
        actor_type=actor_type,
        config=config,
        evolution_context=evolution_context,
    )
    behavior_score = _behavior_scheduler_score(reasons)
    score_total = round(behavior_score * 100)
    threshold_passed = score_total >= config.selection_threshold
    return ActivationCandidate(
        candidate_id=f"activation_candidate_{step.step_id}_{actor_id}",
        run_id=run_id,
        step_id=step.step_id,
        agent_id=actor_id,
        actor_type=actor_type,
        scenario_id=step.scenario_id,
        status=ActivationStatus.CANDIDATE,
        score_total=score_total,
        reasons=reasons,
        threshold_passed=threshold_passed,
        top_k_rank=1,
        selected_by="weighted_scheduler_v1_top_k_threshold",
        tie_break_key=_tie_break_key_stub(actor_id, step),
    )


def _behavior_scheduler_score(reasons: tuple[ActivationReason, ...]) -> float:
    by_type = {reason.reason_type: reason.score / 3 for reason in reasons}
    base = weighted_scheduler_score(
        goal=max(
            by_type.get("goal_relevance_from_static_profile", 0.0),
            by_type.get("static_plan_alignment", 0.0),
        ),
        pressure=by_type.get("pressure_relevance", 0.0),
        relationship=by_type.get("relationship_relevance", 0.0),
        memory=by_type.get("recent_committed_event_relevance", 0.0),
        causal=by_type.get("causal_open_thread_relevance", 0.0),
        risk=0.0 if by_type.get("static_plan_alignment", 0.0) else 0.15,
    )
    return clamp01(base + 0.40 * by_type.get("static_plan_alignment", 0.0))


def _activation_reasons(
    *,
    bundle: SeedBundle,
    step: RunStepPlanItem,
    scenario: ScenarioDefinition,
    observation: ObservationContext,
    agent: AgentProfile | None,
    actor_id: str,
    actor_type: str,
    config: AgentActivationConfig,
    evolution_context: dict[str, object] | None,
) -> tuple[ActivationReason, ...]:
    hints = SCENARIO_ACTIVATION_HINTS.get(step.scenario_id)
    return (
        _static_plan_alignment_reason(step, actor_id, actor_type),
        _location_reason(step, observation, hints),
        _scenario_reason(scenario),
        _pressure_reason(bundle, observation, hints, config, evolution_context),
        _action_metadata_reason(bundle, actor_type, hints, config),
        _role_reason(actor_type, agent, hints),
        _goal_reason(agent, hints),
        _relationship_reason(bundle, actor_id, actor_type, config),
        _player_input_reason(scenario, actor_type),
        _recent_event_reason(evolution_context),
        _causal_open_thread_reason(evolution_context),
    )


def _static_plan_alignment_reason(
    step: RunStepPlanItem,
    actor_id: str,
    actor_type: str,
) -> ActivationReason:
    aligned = actor_id == step.agent_id and actor_type == step.actor_type
    return ActivationReason(
        reason_type="static_plan_alignment",
        score=3 if aligned else 0,
        evidence_ids=(step.step_id, actor_id),
        message=(
            "Candidate matches the deterministic static step plan."
            if aligned
            else "Candidate is scored for comparison but does not replace the static plan actor."
        ),
        visibility_scope="scenario_metadata",
    )


def _location_reason(
    step: RunStepPlanItem,
    observation: ObservationContext,
    hints: ScenarioActivationHints | None,
) -> ActivationReason:
    target_location_id = hints.target_location_id if hints is not None else None
    score = 3 if target_location_id == observation.location.id else 1
    return ActivationReason(
        reason_type="location_relevance",
        score=score,
        evidence_ids=(observation.location.id,),
        message=f"Static step {step.step_id} is evaluated from public observation location.",
        visibility_scope="location_visible",
    )


def _scenario_reason(scenario: ScenarioDefinition) -> ActivationReason:
    return ActivationReason(
        reason_type="scenario_relevance",
        score=3,
        evidence_ids=(scenario.scenario_id, scenario.regression_case_id),
        message="Scenario matrix selected this deterministic static step.",
        visibility_scope="scenario_metadata",
    )


def _pressure_reason(
    bundle: SeedBundle,
    observation: ObservationContext,
    hints: ScenarioActivationHints | None,
    config: AgentActivationConfig,
    evolution_context: dict[str, object] | None,
) -> ActivationReason:
    if not config.use_pressure_seeds or bundle.metadata is None:
        return ActivationReason(
            reason_type="pressure_relevance",
            score=0,
            message="Pressure seed scoring disabled or unavailable.",
            visibility_scope="pressure_seed",
        )
    hint_pressure_types = set(hints.pressure_types if hints is not None else ())
    pressure_matches = tuple(
        pressure
        for pressure in bundle.metadata.pressure_seeds
        if pressure.location_id == observation.location.id
        or pressure.pressure_type in hint_pressure_types
    )
    if not pressure_matches:
        return ActivationReason(
            reason_type="pressure_relevance",
            score=0,
            message="No public pressure seed matched this static activation step.",
            visibility_scope="pressure_seed",
        )
    runtime_levels = _runtime_pressure_levels(evolution_context)
    level_candidates = (
        *(pressure.level for pressure in pressure_matches),
        *(level for pressure_type, level in runtime_levels if pressure_type in hint_pressure_types),
    )
    max_level = max(level_candidates)
    score = 3 if max_level >= 7 else 2 if max_level >= 4 else 1
    return ActivationReason(
        reason_type="pressure_relevance",
        score=score,
        evidence_ids=(
            *(pressure.id for pressure in pressure_matches),
            *(
                f"runtime_pressure:{pressure_type}"
                for pressure_type, _ in runtime_levels
                if pressure_type in hint_pressure_types
            ),
        ),
        message="Public pressure seeds explain why this actor/scenario is relevant.",
        visibility_scope="pressure_seed",
    )


def _action_metadata_reason(
    bundle: SeedBundle,
    actor_type: str,
    hints: ScenarioActivationHints | None,
    config: AgentActivationConfig,
) -> ActivationReason:
    if not config.use_action_metadata or bundle.metadata is None or hints is None:
        return ActivationReason(
            reason_type="action_metadata_relevance",
            score=0,
            message="Action metadata scoring disabled or unavailable.",
            visibility_scope="action_metadata",
        )
    matches = tuple(
        action
        for action in bundle.metadata.action_metadata
        if action.action_type == hints.action_type and actor_type in action.allowed_actor_types
    )
    return ActivationReason(
        reason_type="action_metadata_relevance",
        score=3 if matches else 0,
        evidence_ids=tuple(action.id for action in matches),
        message="Action metadata is compatible with the static actor/scenario.",
        visibility_scope="action_metadata",
    )


def _role_reason(
    actor_type: str,
    agent: AgentProfile | None,
    hints: ScenarioActivationHints | None,
) -> ActivationReason:
    if actor_type == "player":
        return ActivationReason(
            reason_type="actor_role_relevance",
            score=1,
            evidence_ids=("player",),
            message="Player actor is a special governed input actor, not a seed Agent.",
            visibility_scope="scenario_metadata",
        )
    if agent is None or hints is None:
        return ActivationReason(
            reason_type="actor_role_relevance",
            score=0,
            message="No static role metadata was available for scoring.",
            visibility_scope="scenario_metadata",
        )
    role_text = f"{agent.role} {agent.public_summary}".lower()
    matched = any(keyword in role_text for keyword in hints.role_keywords)
    return ActivationReason(
        reason_type="actor_role_relevance",
        score=3 if matched else 1,
        evidence_ids=(agent.id, agent.faction_id) if agent.faction_id else (agent.id,),
        message="AgentProfile static role metadata explains this static step.",
        visibility_scope="scenario_metadata",
    )


def _goal_reason(
    agent: AgentProfile | None,
    hints: ScenarioActivationHints | None,
) -> ActivationReason:
    if agent is None or hints is None:
        return ActivationReason(
            reason_type="goal_relevance_from_static_profile",
            score=0,
            message="No AgentProfile static goals were available for this actor.",
            visibility_scope="scenario_metadata",
        )
    goal_matches = tuple(
        goal
        for goal in agent.cognitive_state.goals
        if any(keyword in goal.description.lower() for keyword in hints.goal_keywords)
    )
    if not goal_matches:
        return ActivationReason(
            reason_type="goal_relevance_from_static_profile",
            score=0,
            message="Static AgentProfile goals did not match this scenario.",
            visibility_scope="scenario_metadata",
        )
    max_priority = max(goal.priority for goal in goal_matches)
    score = 3 if max_priority >= 4 else 2 if max_priority >= 2 else 1
    return ActivationReason(
        reason_type="goal_relevance_from_static_profile",
        score=score,
        evidence_ids=tuple(goal.id for goal in goal_matches),
        message="AgentProfile static goals explain this activation candidate.",
        visibility_scope="scenario_metadata",
    )


def _relationship_reason(
    bundle: SeedBundle,
    actor_id: str,
    actor_type: str,
    config: AgentActivationConfig,
) -> ActivationReason:
    if not config.use_relationship_placeholder:
        return ActivationReason(
            reason_type="relationship_relevance",
            score=0,
            message="Relationship relevance disabled.",
            visibility_scope="relationship_safe_ids",
        )
    if actor_type == "player":
        return ActivationReason(
            reason_type="relationship_relevance",
            score=0,
            message="Player activation does not use seed relationship scoring.",
            visibility_scope="relationship_safe_ids",
        )
    matches = tuple(
        relationship
        for relationship in bundle.agents.relationships
        if actor_id in {relationship.source_agent_id, relationship.target_agent_id}
    )
    return ActivationReason(
        reason_type="relationship_relevance",
        score=1 if matches else 0,
        evidence_ids=tuple(relationship.id for relationship in matches),
        message=(
            "Visible relationship ids contribute a bounded scheduler-v0 signal."
            if matches
            else "No visible relationship id contributed to this activation candidate."
        ),
        visibility_scope="relationship_safe_ids",
    )


def _player_input_reason(
    scenario: ScenarioDefinition,
    actor_type: str,
) -> ActivationReason:
    return ActivationReason(
        reason_type="player_input_relevance",
        score=2 if actor_type == "player" or scenario.is_player_input else 0,
        evidence_ids=(scenario.scenario_id,) if scenario.is_player_input else (),
        message=(
            "Governed player-input scenario contributes to scheduler-v0 relevance."
            if scenario.is_player_input
            else "Scenario is not a player-input route."
        ),
        visibility_scope="scenario_metadata",
    )


def _recent_event_reason(evolution_context: dict[str, object] | None) -> ActivationReason:
    latest_ids = _latest_committed_event_ids(evolution_context)
    return ActivationReason(
        reason_type="recent_committed_event_relevance",
        score=1 if latest_ids else 0,
        evidence_ids=latest_ids[-3:],
        message=(
            "Recent committed event ids contribute a bounded scheduler-v0 signal."
            if latest_ids
            else "No recent committed event is available to scheduler-v0."
        ),
        visibility_scope="evolution_safe_summary",
    )


def _causal_open_thread_reason(evolution_context: dict[str, object] | None) -> ActivationReason:
    causal_count = _int_context_value(evolution_context, "causal_node_count")
    latest_ids = _latest_committed_event_ids(evolution_context)
    return ActivationReason(
        reason_type="causal_open_thread_relevance",
        score=1 if causal_count > 0 else 0,
        evidence_ids=latest_ids[-3:],
        message=(
            "Safe causal runtime summary contributes a bounded scheduler-v0 signal."
            if causal_count > 0
            else "No causal runtime summary is available to scheduler-v0."
        ),
        visibility_scope="evolution_safe_summary",
    )


def _latest_committed_event_ids(
    evolution_context: dict[str, object] | None,
) -> tuple[str, ...]:
    if not isinstance(evolution_context, dict):
        return ()
    causal = evolution_context.get("causal_runtime_summary")
    if not isinstance(causal, dict):
        return ()
    ids = causal.get("latest_committed_event_ids")
    if not isinstance(ids, list):
        return ()
    return tuple(str(item) for item in ids)


def _int_context_value(evolution_context: dict[str, object] | None, key: str) -> int:
    if not isinstance(evolution_context, dict):
        return 0
    value = evolution_context.get(key)
    return value if isinstance(value, int) else 0


def _runtime_pressure_levels(
    evolution_context: dict[str, object] | None,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(evolution_context, dict):
        return ()
    levels = evolution_context.get("latest_pressure_levels")
    if not isinstance(levels, list):
        return ()
    safe_levels: list[tuple[str, int]] = []
    for item in levels:
        if not isinstance(item, dict):
            continue
        pressure_type = item.get("pressure_type")
        after_level = item.get("after_level")
        if isinstance(pressure_type, str) and isinstance(after_level, int):
            safe_levels.append((pressure_type, after_level))
    return tuple(safe_levels)


def _tie_break_key(candidate: ActivationCandidate, step: RunStepPlanItem) -> tuple[str, ...]:
    return (
        str(candidate.agent_id != step.agent_id).lower(),
        candidate.agent_id,
    )


def _tie_break_key_stub(actor_id: str, step: RunStepPlanItem) -> tuple[str, ...]:
    return (
        str(actor_id != step.agent_id).lower(),
        actor_id,
    )


def _find_agent(bundle: SeedBundle, agent_id: str) -> AgentProfile:
    for agent in bundle.agents.agents:
        if agent.id == agent_id:
            return agent
    raise ValueError(f"Unknown agent id for activation: {agent_id}")
