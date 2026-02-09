"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-02-08 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op
from packages.core.database.models import Base

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
