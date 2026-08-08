from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy.exc import SQLAlchemyError

from aethelis import __version__
from aethelis.agents.action_proposal import ProposalSourceMode
from aethelis.algorithms import (
    AlgorithmMechanismConfigurationError,
    load_algorithm_mechanism_config,
    run_algorithm_mechanism_experiment,
    run_algorithm_mechanism_matrix,
)
from aethelis.config.errors import ConfigurationError
from aethelis.config.settings import DEFAULT_ENV_FILE, load_settings
from aethelis.connectivity import check_embedding, check_llm
from aethelis.db import (
    DatabaseConfigurationError,
    RuntimeDBRepository,
    check_database_health,
    create_db_engine,
    load_database_settings,
    upgrade_database,
)
from aethelis.evaluation import (
    FormalEvaluationError,
    evaluate_formal_run,
    evaluate_formal_trace_preview,
)
from aethelis.experiments import (
    FormalExperimentConfigurationError,
    RunMatrixConfigurationError,
    RuntimeRegressionConfigurationError,
    inspect_matrix_summary,
    load_aggregate_evaluation_config,
    load_formal_experiment_config,
    load_run_matrix_config,
    load_runtime_regression_config,
    render_matrix_review_markdown,
    run_experiment_comparison,
    run_formal_experiment,
    run_matrix,
    run_runtime_regression,
)
from aethelis.providers import ProviderError
from aethelis.runtime.db_real_provider import (
    run_mistgate_long_horizon_db,
    run_real_provider_db_step,
)
from aethelis.runtime.multi_agent_db_validation import run_multi_agent_real_provider_db_validation
from aethelis.runtime.scenario_matrix import real_llm_scenario_ids
from aethelis.runtime.single_step import run_single_step
from aethelis.runtime.world_run import WorldRunConfigurationError, load_run_config, run_world
from aethelis.seeds.loader import SeedLoader
from aethelis.seeds.validator import SeedValidator
from aethelis.trace.debug_trace import write_debug_trace
from aethelis.trace.formal import (
    inspect_formal_trace_file,
    load_formal_trace,
    validate_formal_trace_file,
    write_formal_trace_preview,
    write_world_run_trace_preview,
)
from aethelis.utils.redaction import redact_text
from aethelis.utils.ssl_diagnostics import ssl_diagnostics

app = typer.Typer(
    name="aethelis",
    help="World-state governed multi-agent runtime for AI-native virtual worlds.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"Aethelis {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the installed Aethelis version.",
    ),
) -> None:
    """Aethelis command-line interface."""


@app.command("config-check")
def config_check(
    env_file: Annotated[
        Path,
        typer.Option(
            "--env-file",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Environment file to validate.",
        ),
    ] = DEFAULT_ENV_FILE,
    show_summary: Annotated[
        bool,
        typer.Option(
            "--show-summary",
            help="Print a safe configuration summary with credentials redacted.",
        ),
    ] = False,
) -> None:
    """Validate required provider configuration without making API calls."""

    try:
        settings = load_settings(env_file)
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None

    typer.echo("Configuration valid. No external provider call was made.")
    if show_summary:
        typer.echo(json.dumps(settings.safe_summary(), ensure_ascii=False, indent=2))


@app.command("provider-check")
def provider_check(
    llm: Annotated[
        bool,
        typer.Option("--llm", help="Check the configured LLM provider."),
    ] = False,
    embedding: Annotated[
        bool,
        typer.Option("--embedding", help="Check the configured embedding provider."),
    ] = False,
    all_providers: Annotated[
        bool,
        typer.Option("--all", help="Check both configured providers."),
    ] = False,
    env_file: Annotated[
        Path,
        typer.Option(
            "--env-file",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Environment file to validate and use.",
        ),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """Perform low-cost real provider connectivity checks."""

    if not any((llm, embedding, all_providers)):
        typer.echo("Select --llm, --embedding, or --all.", err=True)
        raise typer.Exit(code=2)
    if all_providers and (llm or embedding):
        typer.echo("--all cannot be combined with --llm or --embedding.", err=True)
        raise typer.Exit(code=2)

    try:
        settings = load_settings(env_file)
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None

    reports = []
    if llm or all_providers:
        reports.append(check_llm(settings))
    if embedding or all_providers:
        reports.append(check_embedding(settings))

    output = {
        "ssl": ssl_diagnostics(),
        "checks": [report.safe_dict() for report in reports],
    }
    typer.echo(json.dumps(output, ensure_ascii=False, indent=2))
    if not all(report.success for report in reports):
        raise typer.Exit(code=1)


@app.command("db-health")
def db_health(
    env_file: Annotated[
        Path,
        typer.Option(
            "--env-file",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Environment file containing DATABASE_URL.",
        ),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """Check PostgreSQL connectivity and pgvector availability without printing secrets."""

    try:
        db_settings = load_database_settings(env_file)
        engine = create_db_engine(db_settings)
        health = check_database_health(engine)
    except DatabaseConfigurationError as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(health, ensure_ascii=False, indent=2))


@app.command("db-upgrade")
def db_upgrade() -> None:
    """Run Alembic migrations against DATABASE_URL."""

    try:
        upgrade_database()
    except Exception as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None
    typer.echo(json.dumps({"migration": "upgraded", "revision": "head"}, indent=2))


@app.command("seed-validate")
def seed_validate(
    seed_path: Annotated[
        Path,
        typer.Argument(
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Path to a seed directory.",
        ),
    ],
) -> None:
    """Load and validate a structured world seed without running simulation."""

    load_result = SeedLoader().load(seed_path)
    report = SeedValidator().validate(
        load_result.seed_path,
        load_result.bundle,
        load_errors=load_result.errors,
        loaded_files=load_result.loaded_files,
    )
    typer.echo(json.dumps(report.safe_dict(), ensure_ascii=False, indent=2))
    if not report.success:
        raise typer.Exit(code=1)


@app.command("algorithm-mechanism-run")
def algorithm_mechanism_run(
    seed_path: Annotated[
        Path,
        typer.Option(
            "--seed",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Path to a seed directory.",
        ),
    ],
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Optional algorithm mechanism config.",
        ),
    ] = None,
) -> None:
    """Run Product 05 algorithm mechanism coverage without provider calls."""

    try:
        config = load_algorithm_mechanism_config(config_path)
        report = run_algorithm_mechanism_experiment(seed_path=seed_path, config=config)
    except AlgorithmMechanismConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(report.safe_dict(), ensure_ascii=False, indent=2))
    if not report.coverage_passed:
        raise typer.Exit(code=1)


@app.command("algorithm-mechanism-matrix")
def algorithm_mechanism_matrix(
    seed_paths: Annotated[
        list[Path] | None,
        typer.Option(
            "--seed",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Seed directory to include. Repeat for multi-seed comparison.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Optional algorithm mechanism config.",
        ),
    ] = None,
) -> None:
    """Run Product 05 mechanism comparison across multiple seeds."""

    if not seed_paths:
        typer.echo("Provide at least two --seed values.", err=True)
        raise typer.Exit(code=2)
    try:
        config = load_algorithm_mechanism_config(config_path)
        report = run_algorithm_mechanism_matrix(
            seed_paths=tuple(seed_paths),
            config=config,
        )
    except AlgorithmMechanismConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(report.safe_dict(), ensure_ascii=False, indent=2))
    if not report.coverage_passed:
        raise typer.Exit(code=1)


@app.command("runtime-real-provider-db-run")
def runtime_real_provider_db_run(
    seed_path: Annotated[
        Path,
        typer.Option(
            "--seed",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Path to a seed directory.",
        ),
    ],
    agent_id: Annotated[str, typer.Option("--agent", help="Exact agent id from the seed.")],
    scenario_id: Annotated[str, typer.Option("--scenario", help="Real-provider scenario id.")],
    env_file: Annotated[
        Path,
        typer.Option(
            "--env-file",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Environment file for provider and DATABASE_URL settings.",
        ),
    ] = DEFAULT_ENV_FILE,
    algorithm_config_path: Annotated[
        Path | None,
        typer.Option(
            "--algorithm-config",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Optional Product 05 algorithm mechanism config.",
        ),
    ] = None,
    upgrade_db: Annotated[
        bool,
        typer.Option("--upgrade-db/--no-upgrade-db", help="Run Alembic upgrade before execution."),
    ] = True,
) -> None:
    """Run a real-provider governed step, persist it to PostgreSQL, then read it back."""

    try:
        if upgrade_db:
            upgrade_database()
        settings = load_settings(env_file)
        db_settings = load_database_settings(env_file)
        engine = create_db_engine(db_settings)
        result = run_real_provider_db_step(
            engine=engine,
            seed_path=seed_path,
            agent_id=agent_id,
            scenario_id=scenario_id,
            settings=settings,
            algorithm_config_path=algorithm_config_path,
        )
        readback = RuntimeDBRepository(engine).readback_summary(run_id=result.run_id)
    except (
        ConfigurationError,
        DatabaseConfigurationError,
        ProviderError,
        RuntimeError,
        SQLAlchemyError,
        ValueError,
    ) as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    payload = result.safe_dict() | {"readback": readback}
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    if not (
        result.provider_called
        and not result.fallback_used
        and result.structured_validation_passed
        and result.db_persisted
        and result.db_readback_passed
        and result.mechanism_coverage_passed
        and result.embedding_provider_called
        and not result.embedding_fallback_used
        and result.embedding_db_readback_passed
    ):
        raise typer.Exit(code=1)


@app.command("r2-mistgate-long-horizon-db-run")
def r2_mistgate_long_horizon_db_run(
    seed_path: Annotated[
        Path,
        typer.Option(
            "--seed",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Mistgate seed directory.",
        ),
    ] = Path("seeds/mistgate_v01"),
    agent_id: Annotated[str, typer.Option("--agent", help="Mistgate agent id.")] = "ivo",
    scenario_id: Annotated[
        str,
        typer.Option("--scenario", help="Real-provider Mistgate scenario id."),
    ] = "inspect_workshop_safe",
    step_count: Annotated[
        int,
        typer.Option("--step-count", min=20, max=50, help="R2 long-horizon step count."),
    ] = 20,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Optional explicit DB run id."),
    ] = None,
    env_file: Annotated[
        Path,
        typer.Option(
            "--env-file",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Environment file for provider and DATABASE_URL settings.",
        ),
    ] = DEFAULT_ENV_FILE,
    algorithm_config_path: Annotated[
        Path | None,
        typer.Option(
            "--algorithm-config",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Optional Product 05 algorithm mechanism config.",
        ),
    ] = None,
    upgrade_db: Annotated[
        bool,
        typer.Option("--upgrade-db/--no-upgrade-db", help="Run Alembic upgrade before execution."),
    ] = True,
) -> None:
    """Run R2-B2 Mistgate long-horizon real-provider DB evidence path."""

    try:
        if upgrade_db:
            upgrade_database()
        settings = load_settings(env_file)
        db_settings = load_database_settings(env_file)
        engine = create_db_engine(db_settings)
        result = run_mistgate_long_horizon_db(
            engine=engine,
            seed_path=seed_path,
            agent_id=agent_id,
            scenario_id=scenario_id,
            settings=settings,
            algorithm_config_path=algorithm_config_path,
            step_count=step_count,
            run_id=run_id,
        )
    except (
        ConfigurationError,
        DatabaseConfigurationError,
        ProviderError,
        RuntimeError,
        SQLAlchemyError,
        ValueError,
    ) as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(result.safe_dict(), ensure_ascii=False, indent=2))
    if not (
        result.status == "completed"
        and result.completed_step_count == result.requested_step_count
        and result.db_readback_passed
        and result.mechanism_coverage_passed
        and result.long_horizon_db_readback_passed
    ):
        raise typer.Exit(code=1)


@app.command("r5-multi-agent-db-run")
def r5_multi_agent_db_run(
    seed_path: Annotated[
        Path,
        typer.Option(
            "--seed",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Seed directory for the R5-B5 multi-agent validation run.",
        ),
    ] = Path("seeds/harbor_lantern_v01"),
    scenario_id: Annotated[
        str,
        typer.Option("--scenario", help="Real-provider scenario id."),
    ] = "elin_inspect_cargo_manifest",
    active_agent_ids: Annotated[
        list[str] | None,
        typer.Option("--agent", help="Active agent id. Repeat for same-step multi-agent run."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Optional explicit DB run id."),
    ] = None,
    apply_state_diff: Annotated[
        bool,
        typer.Option("--apply/--no-apply", help="Apply committed StateDiffs to a WorldState copy."),
    ] = True,
    env_file: Annotated[
        Path,
        typer.Option(
            "--env-file",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Environment file for provider and DATABASE_URL settings.",
        ),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """Run R5-B5 real-provider same-step multi-agent validation and DB readback."""

    agents = tuple(active_agent_ids or ("elin", "bram"))
    try:
        settings = load_settings(env_file)
        db_settings = load_database_settings(env_file)
        engine = create_db_engine(db_settings)
        result = run_multi_agent_real_provider_db_validation(
            engine=engine,
            seed_path=seed_path,
            scenario_id=scenario_id,
            active_agent_ids=agents,
            settings=settings,
            apply=apply_state_diff,
            run_id=run_id,
        )
    except (
        ConfigurationError,
        DatabaseConfigurationError,
        ProviderError,
        RuntimeError,
        SQLAlchemyError,
        ValueError,
    ) as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(result.safe_dict(), ensure_ascii=False, indent=2))
    if not (
        result.provider_called
        and not result.fallback_used
        and result.structured_validation_passed
        and result.db_written
        and result.db_readback_passed
        and result.evidence_comparison_passed
    ):
        raise typer.Exit(code=1)


@app.command("run-step")
def run_step(
    seed_path: Annotated[
        Path,
        typer.Option(
            "--seed",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Path to a seed directory.",
        ),
    ],
    agent_id: Annotated[
        str,
        typer.Option(
            "--agent",
            help="Exact agent id from the seed.",
        ),
    ],
    scenario_id: Annotated[
        str,
        typer.Option(
            "--scenario",
            help="Single-step scenario id.",
        ),
    ],
    env_file: Annotated[
        Path,
        typer.Option(
            "--env-file",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Environment file to validate and use.",
        ),
    ] = DEFAULT_ENV_FILE,
    apply_state_diff: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Apply the committed StateDiff to a WorldState copy after verification.",
        ),
    ] = False,
    debug_trace_path: Annotated[
        Path | None,
        typer.Option(
            "--write-debug-trace",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Optional path for a non-formal debug trace JSON file.",
        ),
    ] = None,
    formal_trace_preview_path: Annotated[
        Path | None,
        typer.Option(
            "--write-formal-trace-preview",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Optional path for a non-experiment formal trace preview JSON file.",
        ),
    ] = None,
    proposal_source: Annotated[
        str,
        typer.Option(
            "--proposal-source",
            help="Proposal source: provider_structured or deterministic.",
        ),
    ] = ProposalSourceMode.PROVIDER_STRUCTURED.value,
    provider_proposals_enabled: Annotated[
        bool,
        typer.Option(
            "--provider-proposals-enabled",
            help="Enable provider proposal path.",
        ),
    ] = True,
    allow_real_provider: Annotated[
        bool,
        typer.Option(
            "--allow-real-provider",
            help="Allow real provider construction for provider_structured.",
        ),
    ] = True,
) -> None:
    """Run one dry-run vertical slice; use deterministic flags for baseline/regression."""

    try:
        mode = ProposalSourceMode(proposal_source)
        settings = (
            load_settings(env_file)
            if mode == ProposalSourceMode.PROVIDER_STRUCTURED
            and provider_proposals_enabled
            and allow_real_provider
            and scenario_id in real_llm_scenario_ids()
            else None
        )
        result = run_single_step(
            seed_path=seed_path,
            agent_id=agent_id,
            scenario_id=scenario_id,
            settings=settings,
            proposal_source=mode,
            provider_proposals_enabled=provider_proposals_enabled,
            allow_real_provider=allow_real_provider,
            apply=apply_state_diff,
        )
        if debug_trace_path is not None:
            write_debug_trace(result, debug_trace_path)
        if formal_trace_preview_path is not None:
            write_formal_trace_preview(
                result,
                formal_trace_preview_path,
                seed_id=seed_path.resolve().name,
            )
    except (ConfigurationError, ValueError) as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(result.safe_summary(), ensure_ascii=False, indent=2))
    if result.error is not None:
        raise typer.Exit(code=1)


@app.command("run-world")
def run_world_command(
    seed_path: Annotated[
        Path,
        typer.Option(
            "--seed",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Path to a seed directory.",
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Path to a deterministic preview run config.",
        ),
    ],
    apply_state_diff: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Allow committed StateDiffs to apply to a WorldState copy.",
        ),
    ] = False,
    env_file: Annotated[
        Path,
        typer.Option(
            "--env-file",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Environment file to validate and use for real-provider preview configs.",
        ),
    ] = DEFAULT_ENV_FILE,
    formal_trace_preview_path: Annotated[
        Path | None,
        typer.Option(
            "--write-formal-trace-preview",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Optional path for a non-experiment run-level trace preview JSON file.",
        ),
    ] = None,
) -> None:
    """Run a multi-step preview; deterministic configs do not call providers."""

    try:
        config = load_run_config(config_path)
        settings = load_settings(env_file) if config.allow_real_llm else None
        result = run_world(
            seed_path=seed_path,
            config=config,
            apply=apply_state_diff,
            settings=settings,
        )
        if formal_trace_preview_path is not None:
            write_world_run_trace_preview(
                result,
                formal_trace_preview_path,
                seed_id=seed_path.resolve().name,
            )
    except (ConfigurationError, WorldRunConfigurationError, ValueError) as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(result.safe_summary(), ensure_ascii=False, indent=2))
    if result.errors:
        raise typer.Exit(code=1)


@app.command("experiment-run")
def experiment_run(
    seed_path: Annotated[
        Path,
        typer.Option(
            "--seed",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Path to a seed directory.",
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Path to an explicit formal experiment config.",
        ),
    ],
    env_file: Annotated[
        Path,
        typer.Option(
            "--env-file",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Environment file to validate and use for real-provider experiments.",
        ),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """Run an explicit formal experiment and write runs/<run_id>/ artifacts."""

    try:
        config = load_formal_experiment_config(config_path)
        settings = (
            load_settings(env_file) if config.provider_mode.value == "real_provider" else None
        )
        result = run_formal_experiment(
            seed_path=seed_path,
            config=config,
            settings=settings,
        )
    except (
        ConfigurationError,
        FormalExperimentConfigurationError,
        WorldRunConfigurationError,
        ValueError,
    ) as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(result.safe_summary(), ensure_ascii=False, indent=2))


@app.command("experiment-evaluate")
def experiment_evaluate(
    run_dir: Annotated[
        Path,
        typer.Option(
            "--run",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Path to an explicit formal experiment run directory.",
        ),
    ],
) -> None:
    """Evaluate explicit formal experiment artifacts and write metrics/bad cases."""

    try:
        summary = evaluate_formal_run(run_dir)
    except FormalEvaluationError as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(summary.safe_dict(), ensure_ascii=False, indent=2))


@app.command("experiment-compare")
def experiment_compare(
    seed_path: Annotated[
        Path,
        typer.Option(
            "--seed",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Path to a seed directory.",
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Path to an explicit formal experiment config.",
        ),
    ],
) -> None:
    """Run controlled baseline/ablation variants and write comparison artifacts."""

    try:
        config = load_formal_experiment_config(config_path)
        result = run_experiment_comparison(seed_path=seed_path, config=config)
    except (FormalExperimentConfigurationError, WorldRunConfigurationError, ValueError) as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(result.safe_summary(), ensure_ascii=False, indent=2))


@app.command("matrix-run")
def matrix_run(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Path to a V0.4 run matrix config.",
        ),
    ],
    evaluation_config_path: Annotated[
        Path,
        typer.Option(
            "--evaluation-config",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Path to an aggregate evaluation config.",
        ),
    ],
) -> None:
    """Run a deterministic formal experiment matrix and write aggregate summaries."""

    try:
        config = load_run_matrix_config(config_path)
        evaluation_config = load_aggregate_evaluation_config(evaluation_config_path)
        result = run_matrix(config=config, evaluation_config=evaluation_config)
    except (RunMatrixConfigurationError, FormalExperimentConfigurationError, ValueError) as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(result.safe_dict(), ensure_ascii=False, indent=2))
    if not result.overall.passed:
        raise typer.Exit(code=1)


@app.command("matrix-inspect")
def matrix_inspect(
    summary_path: Annotated[
        Path,
        typer.Argument(
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Path to a generated matrix_summary.json file.",
        ),
    ],
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: json or markdown.",
        ),
    ] = "json",
) -> None:
    """Inspect a matrix summary without reading raw trace bodies."""

    try:
        review = inspect_matrix_summary(summary_path)
    except (OSError, ValueError, RunMatrixConfigurationError) as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    if output_format == "json":
        typer.echo(json.dumps(review, ensure_ascii=False, indent=2))
    elif output_format == "markdown":
        typer.echo(render_matrix_review_markdown(review))
    else:
        typer.echo("--format must be json or markdown.", err=True)
        raise typer.Exit(code=2)

    if not review["passed"]:
        raise typer.Exit(code=1)


@app.command("regression-run")
def regression_run(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Path to a deterministic runtime regression config.",
        ),
    ],
) -> None:
    """Run deterministic runtime regression targets and write a generated summary."""

    try:
        config = load_runtime_regression_config(config_path)
        result = run_runtime_regression(config=config)
    except RuntimeRegressionConfigurationError as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=2) from None

    typer.echo(json.dumps(result.safe_dict(), ensure_ascii=False, indent=2))
    if result.failed_count:
        raise typer.Exit(code=1)


@app.command("trace-validate")
def trace_validate(
    trace_path: Annotated[
        Path,
        typer.Argument(
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Path to a formal trace preview JSON file.",
        ),
    ],
) -> None:
    """Validate a formal trace preview without printing the full trace."""

    report = validate_formal_trace_file(trace_path)
    typer.echo(json.dumps(report.safe_dict(), ensure_ascii=False, indent=2))
    if not report.success:
        raise typer.Exit(code=1)


@app.command("trace-inspect")
def trace_inspect(
    trace_path: Annotated[
        Path,
        typer.Argument(
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Path to a formal trace preview JSON file.",
        ),
    ],
) -> None:
    """Print a safe short trace summary. This is not report export."""

    summary = inspect_formal_trace_file(trace_path)
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary.get("success"):
        raise typer.Exit(code=1)


@app.command("evaluation-check")
def evaluation_check(
    trace_path: Annotated[
        Path,
        typer.Option(
            "--trace",
            exists=False,
            dir_okay=False,
            resolve_path=True,
            help="Path to an existing formal trace preview JSON file.",
        ),
    ],
) -> None:
    """Evaluate an existing formal trace preview against the regression case pack."""

    report = validate_formal_trace_file(trace_path)
    if not report.success:
        typer.echo(json.dumps(report.safe_dict(), ensure_ascii=False, indent=2))
        raise typer.Exit(code=1)

    try:
        trace = load_formal_trace(trace_path)
        summary = evaluate_formal_trace_preview(trace)
    except ValueError as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(code=1) from None

    typer.echo(json.dumps(summary.safe_dict(), ensure_ascii=False, indent=2))
    if summary.failed_count:
        raise typer.Exit(code=1)
