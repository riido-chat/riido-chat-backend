"""문서 그룹을 조회한다."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import DocumentGroup
from app.document.document_key import DEFAULT_DOCUMENT_GROUP_KEY


async def get_document_group(session: AsyncSession) -> DocumentGroup:
    """1차 문서 그룹을 조회한다. migration 20260904_06이 seed한다."""

    group = await session.scalar(
        select(DocumentGroup).where(
            DocumentGroup.group_key == DEFAULT_DOCUMENT_GROUP_KEY
        )
    )
    if group is None:
        raise ValueError(
            f"문서 그룹을 찾을 수 없습니다: {DEFAULT_DOCUMENT_GROUP_KEY}"
        )
    return group
