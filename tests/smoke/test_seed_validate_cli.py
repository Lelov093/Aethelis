from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aethelis.cli.app import app

ROOT = Path(__file__).resolve().parents[2]
VALID_SEED = ROOT / "seeds" / "mistgate_v01"


def test_seed_validate_cli_smoke() -> None:
    result = CliRunner().invoke(app, ["seed-validate", str(VALID_SEED)])

    assert result.exit_code == 0
    assert '"success": true' in result.stdout
    assert '"locations": 5' in result.stdout
    assert '"agents": 6' in result.stdout
    assert '"canon_facts": 8' in result.stdout
    assert '"public_facts": 4' in result.stdout
