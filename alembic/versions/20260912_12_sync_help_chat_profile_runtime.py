"""Synchronize legacy HELP_CHATBOT revisions with the deployed runtime contract.

Only the old, known profile contract is changed.  A revision with any other
model or prompt combination is intentionally left alone so an explicit profile
configuration cannot be overwritten by a deployment migration.
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260912_12"
down_revision: Optional[str] = "20260911_11"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


def upgrade() -> None:
    connection = op.get_bind()
    connection.execute(
        sa.text(
            "UPDATE chat_profile_revisions AS revision "
            "SET generation_prompt_version = 'v38', "
            "    query_rewrite_prompt_version = 'v8' "
            "FROM chat_profiles AS profile "
            "WHERE revision.profile_id = profile.id "
            "  AND profile.profile_key = 'HELP_CHATBOT' "
            "  AND revision.status IN ('PUBLISHED', 'TESTING') "
            "  AND revision.generation_model_name = 'gpt-5.6-terra' "
            "  AND revision.generation_prompt_version = 'v24' "
            "  AND revision.query_rewrite_model_name = 'gpt-5.4-mini' "
            "  AND revision.query_rewrite_prompt_version = 'v7'"
        )
    )


def downgrade() -> None:
    # The upgrade deliberately does not record which rows it changed.  A
    # reverse update could therefore overwrite a v38/v8 configuration that was
    # created intentionally after the migration.  Leave the data untouched.
    pass
