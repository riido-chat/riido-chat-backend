"""Split exact-question and semantic cache serving switches."""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260918_15"
down_revision: Optional[str] = "20260915_14"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


def upgrade() -> None:
    op.add_column(
        "chat_profile_revisions",
        sa.Column(
            "exact_cache_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.execute(
        sa.text(
            "UPDATE chat_profile_revisions "
            "SET exact_cache_enabled = semantic_cache_enabled"
        )
    )


def downgrade() -> None:
    op.drop_column("chat_profile_revisions", "exact_cache_enabled")
