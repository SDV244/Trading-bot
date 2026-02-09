"""Tests for Alembic migration runner."""


from sqlalchemy import create_engine, inspect

from packages.core.config import reload_settings
from packages.core.database.migrations import run_migrations


def test_run_migrations_creates_tables(monkeypatch, tmp_path) -> None:
    """Migration runner upgrades schema to head."""
    import packages.core.config as config_module

    db_path = tmp_path / "migration_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    config_module._settings = None
    reload_settings()

    run_migrations()

    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert "alembic_version" in tables
    assert "orders" in tables
    assert "approvals" in tables
    config_module._settings = None
