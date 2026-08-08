from pathlib import Path

from typer.testing import CliRunner

from aethelis.algorithms import (
    load_algorithm_mechanism_config,
    run_algorithm_mechanism_experiment,
    run_algorithm_mechanism_matrix,
)
from aethelis.algorithms.mechanisms import MechanismKind
from aethelis.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "seeds" / "mistgate_v01"
SEEDS = (
    ROOT / "seeds" / "mistgate_v01",
    ROOT / "seeds" / "harbor_lantern_v01",
    ROOT / "seeds" / "mistgate_v01_civic_pressure_variant",
)
CONFIG = ROOT / "configs" / "v11_algorithm_mechanism_completion.yaml"


def test_algorithm_mechanism_experiment_covers_product_05_mechanisms() -> None:
    config = load_algorithm_mechanism_config(CONFIG)
    report = run_algorithm_mechanism_experiment(seed_path=SEED, config=config)

    assert report.coverage_passed is True
    assert report.mechanism_count == 15
    assert {summary.mechanism_id for summary in report.summaries} == set(MechanismKind)
    assert report.model_family_count >= 30
    assert report.average_complex_score >= 0.55
    assert report.provider_called is False
    assert report.raw_text_saved is False

    by_id = {summary.mechanism_id: summary for summary in report.summaries}
    assert "bayesian" in by_id[MechanismKind.EVENT_VERIFICATION].formula.lower()
    assert "exp(" in by_id[MechanismKind.MEMORY_DECAY].formula
    assert "tanh" in by_id[MechanismKind.RELATIONSHIP_UPDATE].formula
    assert "ewma" in by_id[MechanismKind.WORLD_PRESSURE].model_families[0]
    assert "harmonic" in by_id[MechanismKind.EVALUATION_SCORING].formula.lower()


def test_algorithm_mechanism_cli_is_safe_and_provider_free() -> None:
    result = CliRunner().invoke(
        app,
        [
            "algorithm-mechanism-run",
            "--seed",
            str(SEED),
            "--config",
            str(CONFIG),
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"mechanism_count": 15' in result.output
    assert '"coverage_passed": true' in result.output
    assert '"provider_called": false' in result.output
    assert '"raw_text_saved": false' in result.output
    assert "raw prompt" not in result.output.lower()
    assert "authorization:" not in result.output.lower()


def test_algorithm_mechanism_matrix_compares_multiple_seed_families() -> None:
    config = load_algorithm_mechanism_config(CONFIG)
    report = run_algorithm_mechanism_matrix(seed_paths=SEEDS, config=config)

    assert report.coverage_passed is True
    assert report.seed_count == 3
    assert report.mechanism_count == 15
    assert report.model_family_count >= 30
    assert report.min_seed_score >= 0.55
    assert set(report.mechanism_averages) == set(MechanismKind)
    assert report.provider_called is False
    assert report.raw_text_saved is False


def test_algorithm_mechanism_matrix_cli_is_provider_free() -> None:
    args = ["algorithm-mechanism-matrix", "--config", str(CONFIG)]
    for seed in SEEDS:
        args.extend(["--seed", str(seed)])

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    assert '"seed_count": 3' in result.output
    assert '"mechanism_count": 15' in result.output
    assert '"coverage_passed": true' in result.output
    assert '"provider_called": false' in result.output
    assert '"raw_text_saved": false' in result.output
