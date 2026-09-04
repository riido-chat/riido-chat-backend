"""GitBook 재탐색 배치를 접수하고 실행한다.

페이지마다 수집 실행을 만들고 batch_id 로 묶는다. 처리 규칙은 콘솔 업로드와
같으므로 run_admin_ingestion 을 그대로 쓴다.

배치 자체의 행은 두지 않는다. 계획대로 ingestion_runs.batch_id 가 집계와
진행 조회의 단위다. 그래서 페이지 목록 조회는 접수 요청 안에서 끝내고,
목록 조회 실패는 배치가 아니라 요청의 실패로 돌려준다.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.hashing import sha256_hex
from app.database.models import (
    DocumentSource,
    ExecutionStatus,
    IngestionResultCode,
    IngestionRun,
    IngestionStage,
)
from app.database.session import get_session_factory
from app.document.document_group import get_document_group
from app.document.document_key import (
    SOURCE_TYPE_GITBOOK,
    build_gitbook_document_key,
    normalize_gitbook_root_url,
)
from app.document.document_store import PARSER_NAME, PARSER_VERSION
from app.document.gitbook.client import GitBookPage, fetch_page, list_pages
from app.document.ingestion_service import run_admin_ingestion
from app.retrieval.embedding import OpenAIEmbedder


logger = logging.getLogger(__name__)

RECOLLECT_TRIGGER_TYPE = "RECOLLECT"
REMOVED_ACTION = "REMOVED"
UPSTREAM_ERROR = "UPSTREAM_ERROR"


@dataclass(frozen=True)
class AcceptedRecollect:
    """재탐색 접수 결과."""

    batch_id: uuid.UUID
    group_id: int
    page_count: int


async def accept_recollect_batch(
    session: AsyncSession,
    group_id: int,
    root_url: str,
    pages: List[GitBookPage],
) -> AcceptedRecollect:
    """읽어 온 목록으로 페이지별 실행과 제거 표시를 만든다.

    호출자가 그룹 잠금을 잡은 뒤 호출한다.
    """

    batch_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    root = normalize_gitbook_root_url(root_url)
    group = await get_document_group(session)

    existing = await _load_gitbook_sources(session, group.id, root)
    seen_keys = set()

    for page in pages:
        document_key = build_gitbook_document_key(page.url, root)
        seen_keys.add(document_key)
        source = existing.get(document_key)
        if source is None:
            source = DocumentSource(
                document_group_id=group.id,
                document_key=document_key,
                source_type=SOURCE_TYPE_GITBOOK,
                canonical_uri=page.url,
                title=page.title,
                metadata_={
                    "document_id": sha256_hex(page.url)[:12],
                    "category": page.category,
                },
                enabled=True,
                created_at=now,
                updated_at=now,
            )
            session.add(source)
            await session.flush()
        else:
            source.canonical_uri = page.url
            source.title = page.title
            source.metadata_ = {
                "document_id": (source.metadata_ or {}).get("document_id")
                or sha256_hex(page.url)[:12],
                "category": page.category,
            }
            # 사라졌다가 다시 나타난 페이지는 되살린다
            source.enabled = True
            source.updated_at = now

        session.add(
            IngestionRun(
                document_source_id=source.id,
                trigger_type=RECOLLECT_TRIGGER_TYPE,
                parser_name=PARSER_NAME,
                parser_version=PARSER_VERSION,
                status=ExecutionStatus.PROCESSING,
                stage=IngestionStage.RECEIVING,
                batch_id=batch_id,
                summary={"stage": "RECEIVING", "source_url": page.url},
                started_at=now,
            )
        )

    # 사라진 페이지 판정은 이번에 수집한 루트 아래 문서로만 한정한다.
    # 다른 GitBook 이나 콘솔 문서를 건드리면 안 된다.
    for document_key, source in existing.items():
        if document_key in seen_keys or not source.enabled:
            continue
        # GitBook 에서 사라진 페이지는 행과 판을 보존하고 사용만 멈춘다
        source.enabled = False
        source.updated_at = now
        session.add(
            IngestionRun(
                document_source_id=source.id,
                trigger_type=RECOLLECT_TRIGGER_TYPE,
                parser_name=PARSER_NAME,
                parser_version=PARSER_VERSION,
                status=ExecutionStatus.SUCCESS,
                stage=IngestionStage.PERSISTING,
                result_code=IngestionResultCode.NO_CHANGE,
                batch_id=batch_id,
                summary={
                    "stage": "COMPLETED",
                    "recollect_action": REMOVED_ACTION,
                },
                started_at=now,
                finished_at=now,
            )
        )

    await session.flush()
    accepted = AcceptedRecollect(
        batch_id=batch_id,
        group_id=group.id,
        page_count=len(pages),
    )
    await session.commit()
    return accepted


async def run_recollect_batch(
    batch_id: uuid.UUID,
    embedder_factory: Callable[[], OpenAIEmbedder] = OpenAIEmbedder,
) -> None:
    """배치의 페이지별 실행을 하나씩 처리한다.

    한 페이지가 실패해도 나머지를 계속 처리한다. 실패는 그 실행에만 남는다.
    """

    async with get_session_factory()() as session:
        targets = await _load_pending_runs(session, batch_id)

    for ingestion_run_id, source_url in targets:
        try:
            raw_content = await asyncio.to_thread(fetch_page, source_url)
        except Exception as error:
            logger.warning(
                "재탐색 페이지를 읽지 못했습니다: url=%s, error=%s",
                source_url,
                error,
            )
            await _fail_run(ingestion_run_id, error)
            continue

        await run_admin_ingestion(
            ingestion_run_id,
            raw_content,
            embedder_factory,
        )


async def _load_gitbook_sources(
    session: AsyncSession,
    group_id: int,
    root_url: str,
) -> Dict[str, DocumentSource]:
    """이번 루트 아래에 있는 GitBook 문서만 모은다."""

    rows = (
        await session.execute(
            select(DocumentSource).where(
                DocumentSource.document_group_id == group_id,
                DocumentSource.source_type == SOURCE_TYPE_GITBOOK,
                DocumentSource.canonical_uri.startswith(f"{root_url}/"),
            )
        )
    ).scalars()
    return {source.document_key: source for source in rows}


async def _load_pending_runs(
    session: AsyncSession,
    batch_id: uuid.UUID,
) -> List[tuple]:
    rows = (
        await session.execute(
            select(IngestionRun.id, DocumentSource.canonical_uri)
            .join(
                DocumentSource,
                DocumentSource.id == IngestionRun.document_source_id,
            )
            .where(
                IngestionRun.batch_id == batch_id,
                IngestionRun.status == ExecutionStatus.PROCESSING,
            )
            .order_by(IngestionRun.id)
        )
    ).all()
    return [(run_id, canonical_uri) for run_id, canonical_uri in rows]


async def _fail_run(ingestion_run_id: int, error: Exception) -> None:
    """페이지 원문을 읽지 못한 실행을 접수 단계 실패로 마감한다."""

    from app.document.document_store import DocumentStore

    async with get_session_factory()() as session:
        store = DocumentStore(session)
        try:
            await store.fail_ingestion(
                ingestion_run_id,
                error,
                failed_stage=IngestionStage.RECEIVING.value,
                error_code=UPSTREAM_ERROR,
            )
            await session.commit()
        except Exception:
            logger.exception(
                "재탐색 실패 로그를 마감하지 못했습니다: ingestion_run_id=%s",
                ingestion_run_id,
            )
            await session.rollback()
