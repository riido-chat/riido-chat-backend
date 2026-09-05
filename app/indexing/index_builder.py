"""DB에 적재된 문서 판으로 검색 버전을 만들고 적용한다.

임베딩은 접수 시점에 이미 만들어지므로 이 단계는 DB 작업만 한다.
외부 모델 호출이 없어 운영자 콘솔에서 바로 실행할 수 있다.

명세는 indexer-worker 가 chat-api 의 /internal/corpus/reload 를 호출하는
구조를 전제하지만 이 저장소는 단일 프로세스이므로 corpus 를 직접 교체한다.
전환 순서와 실패 처리 규칙은 그대로 따른다.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ContentNode,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    ExecutionStatus,
    IndexDocument,
    IndexRun,
    IndexRunStage,
    IndexVersion,
    IndexVersionStatus,
)
from app.core.error_message import sanitize_error_message
from app.core.hashing import sha256_hex
from app.document.chunking_config import get_or_create_chunking_config
from app.document.document_group import get_document_group
from app.indexing.index_writer import IndexWriter
from app.retrieval.corpus_state import CorpusState
from app.retrieval.embedding import OpenAIEmbedder, build_embedding_text
from app.retrieval.models import RetrievalChunk
from app.retrieval.search_reader import SearchReader


logger = logging.getLogger(__name__)

MANUAL_TRIGGER_TYPE = "MANUAL"
RETRY_TRIGGER_TYPE = "RETRY"

VALIDATION_FAILED = "VALIDATION_FAILED"
CORPUS_RELOAD_FAILED = "CORPUS_RELOAD_FAILED"
CORPUS_OUT_OF_SYNC = "CORPUS_OUT_OF_SYNC"
INTERNAL_ERROR = "INTERNAL_ERROR"


class MissingEmbeddingError(RuntimeError):
    """임베딩이 없는 청크가 남아 색인을 만들 수 없을 때 발생한다."""


class CorpusReloadFailedError(RuntimeError):
    """corpus 교체에 실패해 전환을 되돌렸을 때 발생한다."""


class CorpusOutOfSyncError(RuntimeError):
    """전환을 되돌린 뒤 corpus 복구까지 실패했을 때 발생한다."""


NEW_CHANGE = "NEW"
UPDATED_CHANGE = "UPDATED"
REMOVED_CHANGE = "REMOVED"


@dataclass(frozen=True)
class PendingDocument:
    """반영 대기 문서 한 건."""

    document_source_id: int
    title: str
    change_type: str


@dataclass(frozen=True)
class PendingDocuments:
    """ACTIVE 조합과 최신 READY 조합의 차이."""

    new: int
    updated: int
    removed: int

    @property
    def total(self) -> int:
        return self.new + self.updated + self.removed


async def load_latest_ready_versions(
    session: AsyncSession,
) -> Dict[int, Tuple[int, int]]:
    """사용 중인 문서 원본마다 가장 최근 READY 판을 찾는다.

    반환은 document_source_id 를 키로, (document_version_id, version_no) 다.
    """

    group = await get_document_group(session)
    statement = (
        select(
            DocumentVersion.document_source_id,
            DocumentVersion.id,
            DocumentVersion.version_no,
        )
        .distinct(DocumentVersion.document_source_id)
        .join(
            DocumentSource,
            DocumentSource.id == DocumentVersion.document_source_id,
        )
        .where(
            DocumentSource.document_group_id == group.id,
            DocumentSource.enabled.is_(True),
            DocumentVersion.status == DocumentVersionStatus.READY,
        )
        .order_by(
            DocumentVersion.document_source_id,
            DocumentVersion.version_no.desc(),
        )
    )
    rows = (await session.execute(statement)).all()
    return {
        source_id: (version_id, version_no)
        for source_id, version_id, version_no in rows
    }


async def load_active_document_versions(session: AsyncSession) -> Set[int]:
    """그룹의 ACTIVE 색인에 연결된 문서 판 집합을 읽는다."""

    group = await get_document_group(session)
    statement = (
        select(IndexDocument.document_version_id)
        .join(
            IndexVersion,
            IndexVersion.id == IndexDocument.index_version_id,
        )
        .where(
            IndexVersion.document_group_id == group.id,
            IndexVersion.status == IndexVersionStatus.ACTIVE,
        )
    )
    return set((await session.execute(statement)).scalars())


async def load_pending_documents(
    session: AsyncSession,
) -> List["PendingDocument"]:
    """반영 대기 문서를 종류와 함께 모은다.

    ACTIVE 조합에 없는 원본은 NEW, 판이 더 새로우면 UPDATED,
    ACTIVE 조합에만 남아 있으면 REMOVED 다.
    """

    latest = await load_latest_ready_versions(session)
    active_version_ids = await load_active_document_versions(session)
    active_source_ids = await _source_ids_of(session, active_version_ids)

    pending = []
    for source_id, (version_id, _) in latest.items():
        if source_id not in active_source_ids:
            pending.append((source_id, NEW_CHANGE))
        elif version_id not in active_version_ids:
            pending.append((source_id, UPDATED_CHANGE))
    for source_id in sorted(active_source_ids - set(latest)):
        pending.append((source_id, REMOVED_CHANGE))

    if not pending:
        return []

    titles = dict(
        (
            await session.execute(
                select(DocumentSource.id, DocumentSource.title).where(
                    DocumentSource.id.in_([source_id for source_id, _ in pending])
                )
            )
        ).all()
    )
    return [
        PendingDocument(
            document_source_id=source_id,
            title=titles.get(source_id) or "",
            change_type=change_type,
        )
        for source_id, change_type in pending
    ]


async def compute_pending_documents(session: AsyncSession) -> PendingDocuments:
    """반영 대기 문서 수를 센다."""

    pending = await load_pending_documents(session)
    return PendingDocuments(
        new=sum(1 for item in pending if item.change_type == NEW_CHANGE),
        updated=sum(1 for item in pending if item.change_type == UPDATED_CHANGE),
        removed=sum(1 for item in pending if item.change_type == REMOVED_CHANGE),
    )


async def _source_ids_of(
    session: AsyncSession,
    document_version_ids: Set[int],
) -> Set[int]:
    if not document_version_ids:
        return set()
    statement = select(DocumentVersion.document_source_id).where(
        DocumentVersion.id.in_(document_version_ids)
    )
    return set((await session.execute(statement)).scalars())


async def load_indexable_chunks(
    session: AsyncSession,
) -> List[RetrievalChunk]:
    """색인 대상 문서 판의 Chunk를 BM25 corpus 순서로 모은다."""

    latest = await load_latest_ready_versions(session)
    version_ids = [version_id for version_id, _ in latest.values()]
    if not version_ids:
        return []

    chunking_config = await get_or_create_chunking_config(
        session,
        datetime.now(timezone.utc),
    )
    statement = (
        select(DocumentChunk, ContentNode, DocumentVersion, DocumentSource)
        .join(ContentNode, ContentNode.id == DocumentChunk.id)
        .join(
            DocumentVersion,
            DocumentVersion.id == ContentNode.document_version_id,
        )
        .join(
            DocumentSource,
            DocumentSource.id == DocumentVersion.document_source_id,
        )
        .where(
            ContentNode.document_version_id.in_(version_ids),
            DocumentChunk.chunking_config_id == chunking_config.id,
        )
        .order_by(DocumentVersion.id, ContentNode.node_order)
    )
    rows = (await session.execute(statement)).all()
    return [
        SearchReader._to_retrieval_chunk(
            document_chunk,
            content_node,
            document_version,
            document_source,
            None,
        )
        for document_chunk, content_node, document_version, document_source in rows
    ]


async def start_reindex_run(
    session: AsyncSession,
    *,
    actor_id: Optional[str] = None,
) -> IndexRun:
    """후보 색인과 실행 행을 만들고 BUILDING 상태로 돌려준다.

    실제 적재와 적용은 background task 가 이어서 수행한다.
    """

    chunks = await load_indexable_chunks(session)
    writer = IndexWriter(session)
    return await writer.start_index(
        chunks,
        trigger_type=MANUAL_TRIGGER_TYPE,
        actor_id=actor_id,
    )


async def start_retry_apply_run(
    session: AsyncSession,
    failed_index_run: IndexRun,
    *,
    actor_id: Optional[str] = None,
) -> IndexRun:
    """실패한 실행이 참조한 READY 후보에 적용 전용 실행을 만든다."""

    writer = IndexWriter(session)
    return await writer.start_apply_run(
        failed_index_run.index_version_id,
        trigger_type=RETRY_TRIGGER_TYPE,
        actor_id=actor_id,
    )


async def run_index_job(
    session: AsyncSession,
    corpus_state: CorpusState,
    index_run_id: int,
    embedder_factory: Callable[[], OpenAIEmbedder] = OpenAIEmbedder,
) -> None:
    """시작된 실행을 단계대로 끝까지 진행한다.

    실패는 단계와 오류 코드를 남기고 마감한다. 예외를 밖으로 던지지 않는다.
    background task 가 호출하기 때문이다.
    """

    writer = IndexWriter(session)
    run = await session.get(IndexRun, index_run_id)
    if run is None:
        logger.error("존재하지 않는 색인 실행입니다: index_run_id=%s", index_run_id)
        return
    if run.status != ExecutionStatus.PROCESSING:
        logger.info(
            "이미 마감된 색인 실행을 건너뜁니다: index_run_id=%s, status=%s",
            index_run_id,
            run.status,
        )
        return

    if run.stage != IndexRunStage.APPLYING:
        built = await _build(session, writer, index_run_id, embedder_factory)
        if not built:
            return
    await _apply(session, writer, corpus_state, index_run_id)


async def _build(
    session: AsyncSession,
    writer: IndexWriter,
    index_run_id: int,
    embedder_factory: Callable[[], OpenAIEmbedder],
) -> bool:
    """조합을 적재하고 검증을 통과하면 후보를 READY로 만든다.

    임베딩은 접수 시점에 만들어지므로 보통 채울 것이 없다. 접수 이전에
    적재된 판이나 설정이 바뀐 경우에만 이 단계에서 채운다.
    """

    failed_stage = "BUILDING"
    error_code = INTERNAL_ERROR
    try:
        chunks = await load_indexable_chunks(session)
        missing = await writer.list_chunks_missing_embedding(
            index_run_id,
            chunks,
        )
        items = await _fill_missing_embeddings(
            session,
            writer,
            index_run_id,
            missing,
            embedder_factory,
        )
        await writer.store_index_items(index_run_id, chunks, items)
        await session.commit()

        failed_stage = "VALIDATING"
        remaining = await writer.list_chunks_missing_embedding(
            index_run_id,
            chunks,
        )
        if remaining:
            error_code = VALIDATION_FAILED
            raise MissingEmbeddingError(
                f"임베딩이 없는 Chunk가 {len(remaining)}개 남아 있습니다."
            )
        await writer.mark_index_ready(index_run_id)
        await session.commit()
        return True
    except Exception as error:
        logger.exception("색인 생성에 실패했습니다: index_run_id=%s", index_run_id)
        await session.rollback()
        await _record_failure(
            session,
            writer,
            index_run_id,
            error,
            failed_stage,
            error_code,
        )
        return False


async def _apply(
    session: AsyncSession,
    writer: IndexWriter,
    corpus_state: CorpusState,
    index_run_id: int,
) -> None:
    """READY 후보를 ACTIVE로 바꾸고 chat corpus까지 같은 세대로 교체한다.

    corpus 교체는 커밋 전에 끝낸다. 실패하면 전환을 롤백해 DB 와 메모리
    모두 이전 세대로 남기고 후보는 READY 로 유지한다.
    """

    corpus_replaced = False
    error_code = CORPUS_RELOAD_FAILED
    try:
        await writer.apply_index(index_run_id)
        chunks = await SearchReader(session).load_active_chunks()
        corpus_state.replace(chunks)
        corpus_replaced = True
        await writer.finish_apply_run(index_run_id)
        await session.commit()
        return
    except Exception as error:
        logger.exception("색인 적용에 실패했습니다: index_run_id=%s", index_run_id)
        await session.rollback()
        # corpus 를 바꾸기 전에 실패했으면 DB 와 메모리가 모두 이전 세대다.
        # 되돌릴 것이 없으므로 복구를 시도하지 않는다.
        if corpus_replaced and not await _restore_corpus(session, corpus_state):
            error_code = CORPUS_OUT_OF_SYNC
        await _record_failure(
            session,
            writer,
            index_run_id,
            error,
            "APPLYING",
            error_code,
        )


async def _record_failure(
    session: AsyncSession,
    writer: IndexWriter,
    index_run_id: int,
    error: Exception,
    failed_stage: str,
    error_code: str,
) -> None:
    try:
        await writer.fail_index(
            index_run_id,
            error,
            failed_stage=failed_stage,
            error_code=error_code,
        )
        await session.commit()
    except Exception:
        logger.exception(
            "색인 실패 로그를 마감하지 못했습니다: index_run_id=%s",
            index_run_id,
        )
        await session.rollback()


async def _restore_corpus(
    session: AsyncSession,
    corpus_state: CorpusState,
) -> bool:
    """롤백으로 되돌아간 ACTIVE 색인에 맞춰 corpus를 복구한다."""

    try:
        chunks = await SearchReader(session).load_active_chunks()
        corpus_state.replace(chunks)
        return True
    except Exception:
        # 복구는 최선 노력이다. 여기서 난 오류가 원래 실패를 가리면 안 된다.
        logger.warning("적용 실패 후 corpus를 복구하지 못했습니다.", exc_info=True)
        return False


async def _fill_missing_embeddings(
    session: AsyncSession,
    writer: IndexWriter,
    index_run_id: int,
    missing: List[RetrievalChunk],
    embedder_factory: Callable[[], OpenAIEmbedder],
) -> List[Tuple[RetrievalChunk, List[float]]]:
    """임베딩이 없는 Chunk만 채운다.

    같은 입력 해시로 이미 저장된 vector가 있으면 복사하고, 남은 것만
    한 번의 요청으로 만든다. 외부 호출을 하면 model_calls에 기록한다.
    """

    if not missing:
        return []

    embedding_config_id = await writer.get_embedding_config_id(index_run_id)
    inputs = [build_embedding_text(chunk) for chunk in missing]
    input_hashes = [sha256_hex(text) for text in inputs]
    reusable = await writer.load_reusable_embeddings(
        embedding_config_id,
        input_hashes,
    )

    positions = [
        position
        for position, input_hash in enumerate(input_hashes)
        if input_hash not in reusable
    ]
    if not positions:
        return [
            (chunk, list(reusable[input_hash]))
            for chunk, input_hash in zip(missing, input_hashes)
        ]

    call = await writer.start_embedding_model_call(index_run_id)
    model_call_id = call.id
    await session.commit()

    started_at = time.monotonic()
    try:
        response = await asyncio.to_thread(
            embedder_factory().embed_many_with_usage,
            [inputs[position] for position in positions],
        )
    except Exception as error:
        await writer.finish_embedding_model_call(
            model_call_id,
            status=ExecutionStatus.FAILED,
            latency_ms=int((time.monotonic() - started_at) * 1000),
            error_message=sanitize_error_message(error),
        )
        await session.commit()
        raise

    await writer.finish_embedding_model_call(
        model_call_id,
        status=ExecutionStatus.SUCCESS,
        latency_ms=int((time.monotonic() - started_at) * 1000),
        input_tokens=response.input_tokens,
        retry_count=response.retry_count,
    )

    generated = dict(zip(positions, response.embeddings))
    return [
        (
            chunk,
            list(generated[position])
            if position in generated
            else list(reusable[input_hashes[position]]),
        )
        for position, chunk in enumerate(missing)
    ]
