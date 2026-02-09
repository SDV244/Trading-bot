"""Database migration runner utilities."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

import alembic.command as alembic_command
from alembic.config import Config


def run_migrations() -> None:
    """Run Alembic upgrades to latest head."""
    root = Path(__file__).resolve().parents[3]
    alembic_ini = root / "alembic.ini"
    if not alembic_ini.exists():
        logger.warning("alembic.ini not found; skipping migrations")
        return

    config = Config(str(alembic_ini))
    config.set_main_option("script_location", str(root / "alembic"))
    alembic_command.upgrade(config, "head")
