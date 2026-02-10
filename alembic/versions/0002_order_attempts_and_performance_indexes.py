"""order attempts and performance indexes

Revision ID: 0002_order_attempts_and_performance_indexes
Revises: 0001_initial_schema
Create Date: 2026-02-10 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_order_attempts_and_performance_indexes"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _index_exists(bind: sa.engine.Connection, table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(bind)
    return any(index.get("name") == index_name for index in inspector.get_indexes(table_name))


def _table_exists(bind: sa.engine.Connection, table_name: str) -> bool:
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    bind = op.get_bind()

    if not _table_exists(bind, "order_attempts"):
        op.create_table(
            "order_attempts",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
            sa.Column("idempotency_key", sa.String(length=64), nullable=False),
            sa.Column("symbol", sa.String(length=20), nullable=False),
            sa.Column("side", sa.String(length=10), nullable=False),
            sa.Column("order_type", sa.String(length=20), nullable=False),
            sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="PENDING"),
            sa.Column("client_order_id", sa.String(length=50), nullable=False),
            sa.Column("exchange_order_id", sa.String(length=50), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                onupdate=sa.func.now(),
                nullable=False,
            ),
            sa.UniqueConstraint("idempotency_key", name="uq_order_attempts_idempotency_key"),
        )

    indexes_to_create = [
        ("order_attempts", "idx_order_attempts_status_created", ["status", "created_at"]),
        ("order_attempts", "idx_order_attempts_symbol_created", ["symbol", "created_at"]),
        ("positions", "idx_positions_paper_symbol", ["is_paper", "symbol"]),
        ("orders", "idx_orders_paper_symbol_created", ["is_paper", "symbol", "created_at"]),
        ("fills", "idx_fills_paper_time", ["is_paper", "filled_at"]),
    ]

    for table_name, index_name, columns in indexes_to_create:
        if not _index_exists(bind, table_name, index_name):
            op.create_index(index_name, table_name, columns, unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    indexes_to_drop = [
        ("fills", "idx_fills_paper_time"),
        ("orders", "idx_orders_paper_symbol_created"),
        ("positions", "idx_positions_paper_symbol"),
        ("order_attempts", "idx_order_attempts_symbol_created"),
        ("order_attempts", "idx_order_attempts_status_created"),
    ]
    for table_name, index_name in indexes_to_drop:
        if _table_exists(bind, table_name) and _index_exists(bind, table_name, index_name):
            op.drop_index(index_name, table_name=table_name)

    if _table_exists(bind, "order_attempts"):
        op.drop_table("order_attempts")
