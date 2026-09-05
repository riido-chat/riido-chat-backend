"""문서 그룹의 수집 원천을 조회하고 만든다.

원천은 문서를 끌어오는 외부 시스템 하나를 가리킨다. 콘솔 업로드처럼
밀어 넣는 문서에는 원천이 없다.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import DocumentGroupSource, SourceProvider
from app.document.document_key import normalize_gitbook_root_url


async def find_gitbook_source(
    session: AsyncSession,
    group_id: int,
    root_url: str,
) -> Optional[DocumentGroupSource]:
    """그룹에서 이 루트를 가리키는 원천을 찾는다."""

    return await session.scalar(
        select(DocumentGroupSource).where(
            DocumentGroupSource.document_group_id == group_id,
            DocumentGroupSource.provider == SourceProvider.GITBOOK,
            DocumentGroupSource.root_url == normalize_gitbook_root_url(root_url),
        )
    )


async def get_or_create_gitbook_source(
    session: AsyncSession,
    group_id: int,
    root_url: str,
) -> DocumentGroupSource:
    """그룹에 이 루트의 원천이 없으면 만든다.

    한 그룹이 여러 GitBook 을 가질 수 있다. 문서 키는 원천 안에서만
    유일하므로 서로 다른 GitBook 이 같은 경로를 가져도 겹치지 않는다.
    """

    root = normalize_gitbook_root_url(root_url)
    source = await find_gitbook_source(session, group_id, root)
    if source is not None:
        if not source.enabled:
            source.enabled = True
            source.updated_at = datetime.now(timezone.utc)
            await session.flush()
        return source

    now = datetime.now(timezone.utc)
    source = DocumentGroupSource(
        document_group_id=group_id,
        provider=SourceProvider.GITBOOK,
        root_url=root,
        enabled=True,
        created_at=now,
        updated_at=now,
    )
    session.add(source)
    await session.flush()
    return source
