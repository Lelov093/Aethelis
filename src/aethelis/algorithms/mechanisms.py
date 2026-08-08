from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from aethelis.schemas.common import AethelisModel, Identifier
from aethelis.schemas.seed import SeedBundle
from aethelis.seeds.loader import SeedLoader


class MechanismKind(StrEnum):
    AGENT_ACTIVATION = "agent_activation"
    OBSERVATION_SCOPE = "observation_scope"
    COGNITION_RETRIEVAL = "cognition_retrieval"
    ACTION_PROPOSAL = "action_proposal"
    EVENT_CANDIDATE = "event_candidate"
    EVENT_VERIFICATION = "event_verification"
    BELIEF_CONFIDENCE = "belief_confidence"
    MEMORY_DECAY = "memory_decay"
    RELATIONSHIP_UPDATE = "relationship_update"
    WORLD_PRESSURE = "world_pressure"
    EVOLUTION_OPPORTUNITY = "evolution_opportunity"
    CAUSAL_GRAPH = "causal_graph"
    PLAYER_INPUT = "player_input"
    EVALUATION_SCORING = "evaluation_scoring"
    MODEL_COMBINATION = "model_combination"


class AlgorithmMechanismConfig(AethelisModel):
    experiment_id: Identifier = "v11_algorithm_mechanism_completion"
    temperature: float = Field(default=0.7, gt=0)
    decay_lambda: float = Field(default=0.18, ge=0)
    evidence_weight: float = Field(default=0.65, ge=0, le=1)
    risk_weight: float = Field(default=0.35, ge=0, le=1)
    pressure_smoothing: float = Field(default=0.45, ge=0, le=1)
    relationship_learning_rate: float = Field(default=0.22, ge=0, le=1)
    mechanism_ids: tuple[MechanismKind, ...] = tuple(MechanismKind)


class AlgorithmMechanismSummary(AethelisModel):
    mechanism_id: MechanismKind
    model_families: tuple[str, ...]
    formula: str
    baseline_score: float = Field(ge=0, le=1)
    complex_score: float = Field(ge=0, le=1)
    improvement: float
    selected_output: str
    trace: dict[str, float | int | str]


class AlgorithmMechanismReport(AethelisModel):
    experiment_id: Identifier
    seed_path: str
    mechanism_count: int
    model_family_count: int
    average_complex_score: float = Field(ge=0, le=1)
    coverage_passed: bool
    provider_called: bool = False
    raw_text_saved: bool = False
    summaries: tuple[AlgorithmMechanismSummary, ...]

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class AlgorithmMechanismMatrixReport(AethelisModel):
    experiment_id: Identifier
    seed_count: int
    mechanism_count: int
    model_family_count: int
    average_complex_score: float = Field(ge=0, le=1)
    min_seed_score: float = Field(ge=0, le=1)
    coverage_passed: bool
    provider_called: bool = False
    raw_text_saved: bool = False
    mechanism_averages: dict[MechanismKind, float]
    seed_reports: tuple[AlgorithmMechanismReport, ...]

    def safe_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class AlgorithmMechanismConfigurationError(ValueError):
    """Safe algorithm mechanism config error."""


def load_algorithm_mechanism_config(path: Path | None) -> AlgorithmMechanismConfig:
    if path is None:
        return AlgorithmMechanismConfig()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AlgorithmMechanismConfigurationError(
            f"{exc.__class__.__name__}: algorithm config could not be read"
        ) from None
    if payload is None:
        return AlgorithmMechanismConfig()
    try:
        return AlgorithmMechanismConfig.model_validate(payload)
    except ValidationError as exc:
        raise AlgorithmMechanismConfigurationError(
            "ValidationError: algorithm mechanism config schema invalid"
        ) from exc


def run_algorithm_mechanism_experiment(
    *,
    seed_path: Path,
    config: AlgorithmMechanismConfig,
) -> AlgorithmMechanismReport:
    loaded = SeedLoader().load(seed_path)
    if loaded.bundle is None:
        raise AlgorithmMechanismConfigurationError("Seed could not be loaded for algorithm run.")
    summaries = tuple(_run_mechanism(kind, loaded.bundle, config) for kind in config.mechanism_ids)
    average = sum(summary.complex_score for summary in summaries) / max(len(summaries), 1)
    families = {family for summary in summaries for family in summary.model_families}
    return AlgorithmMechanismReport(
        experiment_id=config.experiment_id,
        seed_path=str(seed_path),
        mechanism_count=len(summaries),
        model_family_count=len(families),
        average_complex_score=_clamp01(average),
        coverage_passed=len(summaries) == len(MechanismKind) and average >= 0.55,
        summaries=summaries,
    )


def run_algorithm_mechanism_matrix(
    *,
    seed_paths: tuple[Path, ...],
    config: AlgorithmMechanismConfig,
) -> AlgorithmMechanismMatrixReport:
    if len(seed_paths) < 2:
        raise AlgorithmMechanismConfigurationError(
            "Algorithm mechanism matrix requires at least two seed paths."
        )
    reports = tuple(
        run_algorithm_mechanism_experiment(seed_path=seed_path, config=config)
        for seed_path in seed_paths
    )
    seed_scores = tuple(report.average_complex_score for report in reports)
    mechanism_averages = {
        kind: _clamp01(sum(_score_for_mechanism(report, kind) for report in reports) / len(reports))
        for kind in config.mechanism_ids
    }
    families = {
        family
        for report in reports
        for summary in report.summaries
        for family in summary.model_families
    }
    mechanism_count = len(config.mechanism_ids)
    average_score = sum(seed_scores) / len(seed_scores)
    min_seed_score = min(seed_scores)
    return AlgorithmMechanismMatrixReport(
        experiment_id=config.experiment_id,
        seed_count=len(seed_paths),
        mechanism_count=mechanism_count,
        model_family_count=len(families),
        average_complex_score=_clamp01(average_score),
        min_seed_score=_clamp01(min_seed_score),
        coverage_passed=(
            all(report.coverage_passed for report in reports)
            and mechanism_count == len(MechanismKind)
            and min_seed_score >= 0.55
        ),
        mechanism_averages=mechanism_averages,
        seed_reports=reports,
    )


def _score_for_mechanism(report: AlgorithmMechanismReport, kind: MechanismKind) -> float:
    for summary in report.summaries:
        if summary.mechanism_id == kind:
            return summary.complex_score
    raise AlgorithmMechanismConfigurationError(f"Missing mechanism summary: {kind}")


def _run_mechanism(
    kind: MechanismKind,
    bundle: SeedBundle,
    config: AlgorithmMechanismConfig,
) -> AlgorithmMechanismSummary:
    facts = _seed_facts(bundle)
    table = {
        MechanismKind.AGENT_ACTIVATION: _agent_activation,
        MechanismKind.OBSERVATION_SCOPE: _observation_scope,
        MechanismKind.COGNITION_RETRIEVAL: _cognition_retrieval,
        MechanismKind.ACTION_PROPOSAL: _action_proposal,
        MechanismKind.EVENT_CANDIDATE: _event_candidate,
        MechanismKind.EVENT_VERIFICATION: _event_verification,
        MechanismKind.BELIEF_CONFIDENCE: _belief_confidence,
        MechanismKind.MEMORY_DECAY: _memory_decay,
        MechanismKind.RELATIONSHIP_UPDATE: _relationship_update,
        MechanismKind.WORLD_PRESSURE: _world_pressure,
        MechanismKind.EVOLUTION_OPPORTUNITY: _evolution_opportunity,
        MechanismKind.CAUSAL_GRAPH: _causal_graph,
        MechanismKind.PLAYER_INPUT: _player_input,
        MechanismKind.EVALUATION_SCORING: _evaluation_scoring,
        MechanismKind.MODEL_COMBINATION: _model_combination,
    }
    return table[kind](facts, config)


def _seed_facts(bundle: SeedBundle) -> dict[str, float]:
    agent_count = len(bundle.agents.agents)
    location_count = len(bundle.world.locations)
    canon_count = len(bundle.world.canon_facts)
    belief_count = len(bundle.beliefs.beliefs)
    memory_count = len(bundle.memories.memories)
    relationship_count = len(bundle.agents.relationships)
    pressure_count = len(bundle.metadata.pressure_seeds) if bundle.metadata else 0
    action_count = len(bundle.metadata.action_metadata) if bundle.metadata else 0
    resource_count = len(bundle.world.resources)
    hidden_canon_count = sum(
        1 for fact in bundle.world.canon_facts if fact.visibility == "hidden_canon"
    )
    false_belief_count = sum(
        1 for belief in bundle.beliefs.beliefs if belief.truth_status == "false"
    )
    avg_goal_priority = sum(
        goal.priority for agent in bundle.agents.agents for goal in agent.cognitive_state.goals
    ) / max(sum(len(agent.cognitive_state.goals) for agent in bundle.agents.agents), 1)
    avg_pressure = (
        sum(seed.level for seed in bundle.metadata.pressure_seeds) / max(pressure_count, 1)
        if bundle.metadata
        else 0
    )
    avg_trust = sum(record.trust for record in bundle.agents.relationships) / max(
        relationship_count, 1
    )
    return {
        "agent_count": float(agent_count),
        "location_count": float(location_count),
        "canon_count": float(canon_count),
        "belief_count": float(belief_count),
        "memory_count": float(memory_count),
        "relationship_count": float(relationship_count),
        "pressure_count": float(pressure_count),
        "action_count": float(action_count),
        "resource_count": float(resource_count),
        "hidden_canon_ratio": hidden_canon_count / max(canon_count, 1),
        "false_belief_ratio": false_belief_count / max(belief_count, 1),
        "avg_goal_priority": avg_goal_priority / 5,
        "avg_pressure": avg_pressure / 10,
        "avg_trust": (avg_trust + 5) / 10,
    }


def _summary(
    kind: MechanismKind,
    families: tuple[str, ...],
    formula: str,
    baseline: float,
    complex_score: float,
    output: str,
    trace: dict[str, float | int | str],
) -> AlgorithmMechanismSummary:
    baseline = _clamp01(baseline)
    complex_score = _clamp01(complex_score)
    return AlgorithmMechanismSummary(
        mechanism_id=kind,
        model_families=families,
        formula=formula,
        baseline_score=baseline,
        complex_score=complex_score,
        improvement=round(complex_score - baseline, 4),
        selected_output=output,
        trace={
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in trace.items()
        },
    )


def _agent_activation(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    utility = (
        0.42 * facts["avg_goal_priority"]
        + 0.28 * facts["avg_pressure"]
        + 0.2 * facts["avg_trust"]
        + 0.1
    )
    probability = _sigmoid((utility - 0.5) / config.temperature)
    entropy = _entropy((probability, 1 - probability))
    return _summary(
        MechanismKind.AGENT_ACTIVATION,
        ("utility_risk", "softmax", "entropy_regularized_scheduler"),
        "p_i = softmax((0.42g + 0.28p + 0.20r + 0.10) / T)",
        facts["avg_goal_priority"],
        probability * (1 - 0.15 * entropy),
        "rank agents by temperature-scaled utility with entropy penalty",
        {"utility": utility, "probability": probability, "entropy": entropy},
    )


def _observation_scope(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    exposure = 1 - facts["hidden_canon_ratio"]
    scope = _clamp01(
        0.55 * exposure + 0.25 * facts["location_count"] / 5 + 0.2 * facts["resource_count"] / 5
    )
    return _summary(
        MechanismKind.OBSERVATION_SCOPE,
        ("visibility_filter", "information_gain", "access_control_gate"),
        "scope = 0.55 public_visibility + 0.25 location_coverage + 0.20 resource_coverage",
        exposure,
        scope,
        "select observable facts by visibility/access and expected information gain",
        {"public_visibility": exposure, "scope": scope, "temperature": config.temperature},
    )


def _cognition_retrieval(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    density = facts["memory_count"] / max(facts["agent_count"], 1)
    semantic = _clamp01(math.log1p(density) / 2)
    salience_decay = math.exp(-config.decay_lambda * max(density - 1, 0))
    score = _clamp01(0.6 * semantic + 0.4 * salience_decay)
    return _summary(
        MechanismKind.COGNITION_RETRIEVAL,
        ("bm25_like_sparse_retrieval", "salience_decay", "semantic_similarity"),
        "retrieval = 0.60 log(1 + memories/agent)/2 + 0.40 exp(-lambda age)",
        min(density / 5, 1),
        score,
        "rank memories and beliefs by semantic match with salience decay",
        {"memory_density": density, "semantic": semantic, "salience_decay": salience_decay},
    )


def _action_proposal(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    reward = (
        0.5 * facts["avg_goal_priority"] + 0.3 * facts["avg_pressure"] + 0.2 * facts["avg_trust"]
    )
    risk = 0.55 * facts["hidden_canon_ratio"] + 0.45 * facts["false_belief_ratio"]
    utility = config.evidence_weight * reward - config.risk_weight * risk
    score = _sigmoid(2.5 * utility)
    return _summary(
        MechanismKind.ACTION_PROPOSAL,
        ("expected_utility", "risk_gate", "structured_llm_policy"),
        "proposal_score = sigmoid(2.5 * (evidence_weight * reward - risk_weight * risk))",
        reward,
        score,
        "generate proposal only when expected utility clears risk-adjusted gate",
        {"reward": reward, "risk": risk, "utility": utility},
    )


def _event_candidate(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    completeness = _geometric_mean(
        facts["agent_count"] / 6,
        facts["location_count"] / 5,
        facts["resource_count"] / 4,
        facts["action_count"] / 10,
    )
    score = _clamp01(0.75 * completeness + 0.25 * config.evidence_weight)
    return _summary(
        MechanismKind.EVENT_CANDIDATE,
        ("schema_completeness", "constraint_projection", "typed_event_builder"),
        (
            "candidate_quality = 0.75 geometric_mean(actor, location, target, action) "
            "+ 0.25 evidence_weight"
        ),
        completeness,
        score,
        "project proposals into typed candidates with actor/location/resource completeness",
        {"completeness": completeness, "action_count": facts["action_count"]},
    )


def _event_verification(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    prior = 1 - facts["false_belief_ratio"]
    gate_strength = 0.45 + 0.35 * (1 - facts["hidden_canon_ratio"]) + 0.2 * facts["avg_trust"]
    posterior = _bayes(prior, gate_strength, 1 - gate_strength)
    return _summary(
        MechanismKind.EVENT_VERIFICATION,
        ("hard_rule_gate", "bayesian_verifier_ensemble", "risk_threshold"),
        "Bayesian P(commit|evidence) = prior*gate / (prior*gate + (1-prior)*(1-gate))",
        prior,
        posterior,
        "aggregate hard gates and Bayesian evidence into commit/reject/revise/pending",
        {"prior": prior, "gate_strength": gate_strength, "posterior": posterior},
    )


def _belief_confidence(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    prior = 1 - facts["false_belief_ratio"]
    source_reliability = 0.55 + 0.3 * facts["avg_trust"] + 0.15 * (1 - facts["hidden_canon_ratio"])
    posterior = _bayes(prior, source_reliability, 1 - source_reliability)
    return _summary(
        MechanismKind.BELIEF_CONFIDENCE,
        ("bayesian_update", "source_trust", "evidence_reliability"),
        "belief' = Bayes(prior, source_reliability, contradiction_rate)",
        prior,
        posterior,
        "update belief confidence from source trust and canon-safe evidence reliability",
        {"prior": prior, "source_reliability": source_reliability, "posterior": posterior},
    )


def _memory_decay(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    reinforcement = _clamp01(facts["avg_pressure"] + facts["avg_goal_priority"]) / 2
    retained = math.exp(-config.decay_lambda * facts["memory_count"] / max(facts["agent_count"], 1))
    score = _clamp01(retained + (1 - retained) * reinforcement)
    return _summary(
        MechanismKind.MEMORY_DECAY,
        ("exponential_decay", "reinforcement_boost", "salience_retention"),
        "memory_strength = exp(-lambda * density) + (1-exp(-lambda*density))*reinforcement",
        retained,
        score,
        "decay stale memories while boosting goal/pressure-relevant memories",
        {"retained": retained, "reinforcement": reinforcement},
    )


def _relationship_update(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    surprise = abs(facts["avg_trust"] - 0.5)
    delta = config.relationship_learning_rate * math.tanh(2 * (facts["avg_pressure"] - surprise))
    score = _clamp01(facts["avg_trust"] + delta)
    return _summary(
        MechanismKind.RELATIONSHIP_UPDATE,
        ("bounded_delta_rule", "trust_learning_rate", "tanh_saturation"),
        "trust' = clamp(trust + lr * tanh(2 * (pressure - surprise)))",
        facts["avg_trust"],
        score,
        "update trust with bounded pressure/surprise response",
        {"surprise": surprise, "delta": delta, "learning_rate": config.relationship_learning_rate},
    )


def _world_pressure(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    event_impulse = _clamp01(
        0.45 * facts["resource_count"] / 5 + 0.35 * facts["false_belief_ratio"] + 0.2
    )
    smoothed = (
        config.pressure_smoothing * event_impulse
        + (1 - config.pressure_smoothing) * facts["avg_pressure"]
    )
    return _summary(
        MechanismKind.WORLD_PRESSURE,
        ("ewma_pressure_field", "event_impulse", "damped_system"),
        "pressure' = alpha * event_impulse + (1-alpha) * pressure",
        facts["avg_pressure"],
        smoothed,
        "update pressure as an EWMA dynamic system with event impulses",
        {"event_impulse": event_impulse, "smoothed_pressure": smoothed},
    )


def _evolution_opportunity(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    opportunity = _sigmoid(3 * (facts["avg_pressure"] + facts["false_belief_ratio"] - 0.7))
    safety = 1 - facts["hidden_canon_ratio"]
    score = _clamp01(0.65 * opportunity + 0.35 * safety)
    return _summary(
        MechanismKind.EVOLUTION_OPPORTUNITY,
        ("opportunity_selection", "pressure_trigger", "safety_gate"),
        "opportunity = 0.65 sigmoid(3*(pressure+conflict-0.7)) + 0.35 safety",
        opportunity,
        score,
        "select evolution opportunities when pressure/conflict clears a safety gate",
        {"opportunity": opportunity, "safety": safety},
    )


def _causal_graph(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    node_support = _clamp01((facts["canon_count"] + facts["resource_count"]) / 14)
    edge_probability = _sigmoid(2.2 * (node_support + facts["avg_pressure"] - 0.8))
    return _summary(
        MechanismKind.CAUSAL_GRAPH,
        ("causal_edge_probability", "typed_graph_projection", "counterfactual_signal"),
        "P(edge) = sigmoid(2.2 * (node_support + pressure - 0.8))",
        node_support,
        edge_probability,
        "score typed causal edges from event/support/pressure evidence",
        {"node_support": node_support, "edge_probability": edge_probability},
    )


def _player_input(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    claim_risk = _clamp01(facts["hidden_canon_ratio"] + facts["false_belief_ratio"])
    assimilation = _clamp01(config.evidence_weight * (1 - claim_risk))
    return _summary(
        MechanismKind.PLAYER_INPUT,
        ("claim_risk_model", "belief_candidate_gate", "canon_contamination_guard"),
        "assimilation = evidence_weight * (1 - hidden_or_false_claim_risk)",
        1 - claim_risk,
        assimilation,
        "route player input to governed candidate/belief path, never direct Canon",
        {"claim_risk": claim_risk, "assimilation": assimilation},
    )


def _evaluation_scoring(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    governance = 1 - facts["false_belief_ratio"]
    traceability = _clamp01(
        (facts["canon_count"] + facts["belief_count"] + facts["memory_count"]) / 30
    )
    reproducibility = 1.0
    harmonic = _harmonic_mean(governance, traceability, reproducibility)
    return _summary(
        MechanismKind.EVALUATION_SCORING,
        ("weighted_harmonic_mean", "governance_metrics", "traceability_score"),
        "score = harmonic_mean(governance, traceability, reproducibility)",
        (governance + traceability) / 2,
        harmonic,
        "score runs with governance-safe harmonic aggregation",
        {
            "governance": governance,
            "traceability": traceability,
            "reproducibility": reproducibility,
        },
    )


def _model_combination(
    facts: dict[str, float], config: AlgorithmMechanismConfig
) -> AlgorithmMechanismSummary:
    complexity_penalty = math.log1p(facts["agent_count"] + facts["resource_count"]) / 10
    evidence_fit = _clamp01(
        0.5 * facts["avg_goal_priority"] + 0.5 * (1 - facts["false_belief_ratio"])
    )
    pareto_score = _clamp01(evidence_fit - 0.25 * complexity_penalty + 0.15)
    return _summary(
        MechanismKind.MODEL_COMBINATION,
        ("pareto_selection", "complexity_penalty", "experiment_arm_ranking"),
        "model_score = evidence_fit - 0.25*log(1+complexity)/10 + 0.15",
        evidence_fit,
        pareto_score,
        "rank model-family experiment arms by evidence fit and complexity penalty",
        {"evidence_fit": evidence_fit, "complexity_penalty": complexity_penalty},
    )


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _bayes(prior: float, true_likelihood: float, false_likelihood: float) -> float:
    prior = _clamp01(prior)
    numerator = prior * _clamp01(true_likelihood)
    denominator = numerator + (1 - prior) * _clamp01(false_likelihood)
    return _clamp01(numerator / denominator) if denominator else 0


def _entropy(values: tuple[float, ...]) -> float:
    return -sum(value * math.log(max(value, 1e-9), 2) for value in values)


def _geometric_mean(*values: float) -> float:
    product = 1.0
    for value in values:
        product *= max(_clamp01(value), 1e-9)
    return product ** (1 / len(values))


def _harmonic_mean(*values: float) -> float:
    clean = tuple(max(_clamp01(value), 1e-9) for value in values)
    return len(clean) / sum(1 / value for value in clean)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
