from __future__ import annotations

from pathlib import Path

from alembic.config import Config

from alembic import command

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def upgrade_database(revision: str = "head") -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    command.upgrade(config, revision)
