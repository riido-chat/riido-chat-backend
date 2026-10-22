"""문서 그룹을 조회한다."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import DocumentGroup
from app.document.document_key import DEFAULT_DOCUMENT_GROUP_KEY


async def get_document_group(
    session: AsyncSession,
    group_id: int,
) -> DocumentGroup:
    """ID로 문서 그룹을 조회한다. 없으면 None을 돌려준다.

    요청이 그룹을 지정하는 경로는 모두 이 함수를 쓴다. 상수로 찾으면
    API가 받은 groupId를 무시하게 되고, 그룹이 늘어도 코드가 보지 못한다.
    """

    return await session.get(DocumentGroup, group_id)


async def get_default_document_group(session: AsyncSession) -> DocumentGroup:
    """요청 맥락이 없는 경로가 쓰는 기본 그룹을 조회한다.

    CLI 시드처럼 그룹을 지정할 방법이 없는 곳만 이 함수를 쓴다.
    migration 20260904_06이 seed한다.
    """

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
