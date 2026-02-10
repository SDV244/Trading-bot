"""scope position uniqueness by paper/live mode

Revision ID: 0003_positions_symbol_scope
Revises: 0002_order_attempts_and_performance_indexes
Create Date: 2026-02-10 00:00:01.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_positions_symbol_scope"
down_revision: str | None = "0002_order_attempts_and_performance_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table_exists(bind: sa.engine.Connection, table_name: str) -> bool:
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    if not _table_exists(bind, "positions"):
        return

    dialect = bind.dialect.name
    if dialect == "sqlite":
        op.execute(
            sa.text(
                """
                CREATE TABLE positions_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NULL,
                    quantity NUMERIC(20, 8) NOT NULL,
                    avg_entry_price NUMERIC(20, 8) NOT NULL,
                    unrealized_pnl NUMERIC(20, 8) NOT NULL,
                    realized_pnl NUMERIC(20, 8) NOT NULL,
                    total_fees NUMERIC(20, 8) NOT NULL,
                    is_paper BOOLEAN NOT NULL,
                    updated_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                    CONSTRAINT uq_positions_symbol_is_paper UNIQUE (symbol, is_paper)
                )
                """
            )
        )
        op.execute(
            sa.text(
                """
                INSERT INTO positions_new (
                    id, symbol, side, quantity, avg_entry_price, unrealized_pnl,
                    realized_pnl, total_fees, is_paper, updated_at
                )
                SELECT
                    id, symbol, side, quantity, avg_entry_price, unrealized_pnl,
                    realized_pnl, total_fees, is_paper, updated_at
                FROM positions
                """
            )
        )
        op.execute(sa.text("DROP TABLE positions"))
        op.execute(sa.text("ALTER TABLE positions_new RENAME TO positions"))
        op.create_index("idx_positions_paper_symbol", "positions", ["is_paper", "symbol"], unique=False)
        return

    with op.batch_alter_table("positions") as batch_op:
        batch_op.create_unique_constraint("uq_positions_symbol_is_paper", ["symbol", "is_paper"])


def downgrade() -> None:
    bind = op.get_bind()
    if not _table_exists(bind, "positions"):
        return

    dialect = bind.dialect.name
    if dialect == "sqlite":
        op.execute(
            sa.text(
                """
                CREATE TABLE positions_old (
                    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    symbol VARCHAR(20) NOT NULL UNIQUE,
                    side VARCHAR(10) NULL,
                    quantity NUMERIC(20, 8) NOT NULL,
                    avg_entry_price NUMERIC(20, 8) NOT NULL,
                    unrealized_pnl NUMERIC(20, 8) NOT NULL,
                    realized_pnl NUMERIC(20, 8) NOT NULL,
                    total_fees NUMERIC(20, 8) NOT NULL,
                    is_paper BOOLEAN NOT NULL,
                    updated_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)
                )
                """
            )
        )
        op.execute(
            sa.text(
                """
                INSERT INTO positions_old (
                    id, symbol, side, quantity, avg_entry_price, unrealized_pnl,
                    realized_pnl, total_fees, is_paper, updated_at
                )
                SELECT
                    id, symbol, side, quantity, avg_entry_price, unrealized_pnl,
                    realized_pnl, total_fees, is_paper, updated_at
                FROM positions
                """
            )
        )
        op.execute(sa.text("DROP TABLE positions"))
        op.execute(sa.text("ALTER TABLE positions_old RENAME TO positions"))
        op.create_index("idx_positions_paper_symbol", "positions", ["is_paper", "symbol"], unique=False)
        return

    with op.batch_alter_table("positions") as batch_op:
        batch_op.drop_constraint("uq_positions_symbol_is_paper", type_="unique")
