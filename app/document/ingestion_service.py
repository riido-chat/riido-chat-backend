"""관리자 Markdown 신규 업로드의 접수와 수집 수명주기를 관리한다."""

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Callable, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.document.job_gate import (
    acquire_group_job_gate,
    find_processing_job,
)
from app.document.document_key import (
    DEFAULT_DOCUMENT_GROUP_KEY,
    SOURCE_TYPE_UPLOAD,
    build_console_canonical_uri,
    build_upload_document_key,
)
from app.database.models import (
    DocumentGroup,
    DocumentSource,
    DocumentVersion,
    DocumentVersionStatus,
    ExecutionStatus,
    IndexRun,
    IngestionResultCode,
    IngestionRun,
    IngestionStage,
)
from app.database.session import get_session_factory
from app.document.models import NormalizedDocument
from app.document.clean import normalize_markdown
from app.document.chunker import create_chunks
from app.document.section_parser import parse_sections
from app.retrieval.embedding import OpenAIEmbedder
from app.retrieval.models import RetrievalChunk
from app.document.document_store import PARSER_NAME, PARSER_VERSION, DocumentStore
from app.document.ingestion import prepare_chunk_embeddings


logger = logging.getLogger(__name__)

ADMIN_SOURCE_TYPE = SOURCE_TYPE_UPLOAD
ADMIN_TRIGGER_TYPE = "ADMIN_UPLOAD"

INVALID_FILE = "INVALID_FILE"
FILE_TOO_LARGE = "FILE_TOO_LARGE"
DOCUMENT_NOT_REVISABLE = "DOCUMENT_NOT_REVISABLE"
JOB_IN_PROGRESS = "JOB_IN_PROGRESS"
NOT_FOUND = "NOT_FOUND"
INTERNAL_ERROR = "INTERNAL_ERROR"


class AdminApiError(RuntimeError):
    """Admin API가 상태 코드와 오류 응답으로 변환할 수 있는 예외."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        stage: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        # 접수 전 거절에만 붙는다. FE가 3-4 원인 문구 변형을 고를 때 쓴다.
        self.stage = stage


class InvalidUploadFileError(AdminApiError):
    def __init__(self, message: str = "올바른 Markdown 파일이 아닙니다.") -> None:
        super().__init__(
            INVALID_FILE,
            message,
            HTTPStatus.UNPROCESSABLE_ENTITY,
            stage=IngestionStage.VALIDATING.value,
        )


class UploadFileTooLargeError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            FILE_TOO_LARGE,
            "Markdown 파일은 5MB 이하여야 합니다.",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )


class DocumentNotFoundError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            NOT_FOUND,
            "존재하지 않는 문서입니다.",
            HTTPStatus.NOT_FOUND,
        )


class DocumentNotRevisableError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            DOCUMENT_NOT_REVISABLE,
            "GitBook 문서에는 수정본을 올릴 수 없습니다.",
            HTTPStatus.CONFLICT,
        )


class DocumentGroupNotFoundError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            NOT_FOUND,
            "존재하지 않는 문서 그룹입니다.",
            HTTPStatus.NOT_FOUND,
        )


class AdminJobInProgressError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            JOB_IN_PROGRESS,
            "다른 관리자 문서 작업을 처리 중입니다.",
            HTTPStatus.CONFLICT,
        )


class IngestionRunNotFoundError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            NOT_FOUND,
            "존재하지 않는 수집 실행입니다.",
            HTTPStatus.NOT_FOUND,
        )


class _UploadedMarkdownInvalidError(ValueError):
    """접수 뒤 문서 파이프라인에서 발견된 입력 오류."""


@dataclass(frozen=True)
class AcceptedIngestion:
    ingestion_run_id: int
    document_source_id: int


@dataclass(frozen=True)
class DuplicateDocument:
    """같은 본문을 이미 가진 문서."""

    document_id: int
    title: str


@dataclass(frozen=True)
class IngestionRunDetail:
    ingestion_run_id: int
    document_source_id: int
    status: ExecutionStatus
    stage: str
    document_version_id: Optional[int]
    version_no: Optional[int]
    section_count: Optional[int]
    chunk_count: Optional[int]
    error_code: Optional[str]
    error_message: Optional[str]
    started_at: datetime
    finished_at: Optional[datetime]
    result_code: Optional[str] = None
    chunk_stats: Optional[dict] = None
    duplicate_of: Optional[DuplicateDocument] = None


class AdminIngestionService:
    """요청 세션 안에서 신규 문서 접수와 실행 조회를 수행한다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start_new_document(
        self,
        *,
        group_id: int,
        title: str,
        category: Optional[str] = None,
        filename: str,
    ) -> AcceptedIngestion:
        """그룹 작업 gate 안에서 문서 원본과 PROCESSING 실행을 확정한다.

        같은 문서명의 콘솔 문서가 이미 있으면 거절하지 않고 그 문서의 새 판
        후보가 된다. 결과는 처리 뒤 result_code 로 구분한다.
        """

        try:
            group = await self._get_group(group_id)
            await self._acquire_group_gate(group.id)
            await self._ensure_no_processing_job(group.id)
            source = await self._find_or_create_upload_source(
                group,
                title=title,
                category=category,
            )
            return await self._accept(source, filename)
        except Exception:
            await self._session.rollback()
            raise

    async def start_document_revision(
        self,
        *,
        document_id: int,
        filename: str,
    ) -> AcceptedIngestion:
        """기존 콘솔 문서를 대상으로 수정본 실행을 확정한다.

        GitBook 문서는 재탐색으로만 새 판이 생기므로 거절한다.
        """

        try:
            source = await self._session.get(DocumentSource, document_id)
            if source is None:
                raise DocumentNotFoundError()

            await self._acquire_group_gate(source.document_group_id)
            await self._ensure_no_processing_job(source.document_group_id)
            if source.source_type != ADMIN_SOURCE_TYPE:
                raise DocumentNotRevisableError()

            has_ready_version = await self._session.scalar(
                select(DocumentVersion.id)
                .where(
                    DocumentVersion.document_source_id == source.id,
                    DocumentVersion.status == DocumentVersionStatus.READY,
                )
                .limit(1)
            )
            if has_ready_version is None:
                # 판이 없는 원본은 목록에 보이지 않으므로 대상이 될 수 없다.
                raise DocumentNotFoundError()

            return await self._accept(source, filename)
        except Exception:
            await self._session.rollback()
            raise

    async def _accept(
        self,
        source: DocumentSource,
        filename: str,
    ) -> AcceptedIngestion:
        now = datetime.now(timezone.utc)
        run = IngestionRun(
            document_source_id=source.id,
            trigger_type=ADMIN_TRIGGER_TYPE,
            parser_name=PARSER_NAME,
            parser_version=PARSER_VERSION,
            status=ExecutionStatus.PROCESSING,
            stage=IngestionStage.RECEIVING,
            summary={"stage": "RECEIVING", "filename": filename},
            started_at=now,
        )
        self._session.add(run)
        await self._session.flush()
        result = AcceptedIngestion(
            ingestion_run_id=run.id,
            document_source_id=source.id,
        )
        await self._session.commit()
        return result

    async def get_ingestion_run(self, ingestion_run_id: int) -> IngestionRunDetail:
        """수집 실행과 성공 시 생성한 문서 버전을 함께 조회한다."""

        statement = (
            select(IngestionRun, DocumentVersion)
            .outerjoin(
                DocumentVersion,
                DocumentVersion.id == IngestionRun.produced_version_id,
            )
            .where(IngestionRun.id == ingestion_run_id)
        )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            raise IngestionRunNotFoundError()

        run, version = row
        summary = run.summary or {}
        return IngestionRunDetail(
            ingestion_run_id=run.id,
            document_source_id=run.document_source_id,
            status=run.status,
            stage=run.stage.value,
            document_version_id=None if version is None else version.id,
            version_no=await self._version_no_for(run, version),
            section_count=summary.get("section_count"),
            chunk_count=summary.get("chunk_count"),
            error_code=run.error_code or summary.get("error_code"),
            error_message=run.error_message,
            started_at=run.started_at,
            finished_at=run.finished_at,
            result_code=(
                None if run.result_code is None else run.result_code.value
            ),
            chunk_stats=_chunk_stats_of(summary),
            duplicate_of=await self._duplicate_of(run),
        )

    async def _version_no_for(
        self,
        run: IngestionRun,
        version: Optional[DocumentVersion],
    ) -> Optional[int]:
        """새 판이 있으면 그 번호, 없으면 대상 문서의 현재 판 번호다."""

        if version is not None:
            return version.version_no
        if run.result_code not in (
            IngestionResultCode.NO_CHANGE,
            IngestionResultCode.DUPLICATE_CONTENT,
        ):
            return None
        return await self._session.scalar(
            select(func.max(DocumentVersion.version_no)).where(
                DocumentVersion.document_source_id == run.document_source_id,
                DocumentVersion.status == DocumentVersionStatus.READY,
            )
        )

    async def _duplicate_of(
        self,
        run: IngestionRun,
    ) -> Optional[DuplicateDocument]:
        if run.duplicate_of_document_source_id is None:
            return None
        source = await self._session.get(
            DocumentSource,
            run.duplicate_of_document_source_id,
        )
        if source is None:
            return None
        return DuplicateDocument(
            document_id=source.id,
            title=source.title or "",
        )

    async def _acquire_group_gate(self, group_id: int) -> None:
        await acquire_group_job_gate(self._session, group_id)

    async def _ensure_no_processing_job(self, group_id: int) -> None:
        if await find_processing_job(self._session, group_id) is not None:
            raise AdminJobInProgressError()

    async def _get_group(self, group_id: int) -> DocumentGroup:
        group = await self._get_document_group()
        if group.id != group_id:
            raise DocumentGroupNotFoundError()
        return group

    async def _get_document_group(self) -> DocumentGroup:
        """1차 문서 그룹을 조회한다. migration 20260904_06이 seed한다."""

        group = await self._session.scalar(
            select(DocumentGroup).where(
                DocumentGroup.group_key == DEFAULT_DOCUMENT_GROUP_KEY
            )
        )
        if group is None:
            raise RuntimeError(
                f"문서 그룹을 찾을 수 없습니다: {DEFAULT_DOCUMENT_GROUP_KEY}"
            )
        return group

    async def _find_or_create_upload_source(
        self,
        group: DocumentGroup,
        *,
        title: str,
        category: Optional[str],
    ) -> DocumentSource:
        document_key = build_upload_document_key(title)
        # 콘솔 문서는 밀어 넣는 문서라 수집 원천이 없다. 키는 그룹 안에서 유일하다.
        source = await self._session.scalar(
            select(DocumentSource).where(
                DocumentSource.document_group_id == group.id,
                DocumentSource.group_source_id.is_(None),
                DocumentSource.document_key == document_key,
            )
        )
        if source is None:
            now = datetime.now(timezone.utc)
            source = DocumentSource(
                document_group_id=group.id,
                document_key=document_key,
                source_type=ADMIN_SOURCE_TYPE,
                canonical_uri=build_console_canonical_uri(
                    group.group_key,
                    document_key,
                ),
                title=title,
                metadata_={
                    "document_id": f"admin-{uuid.uuid4().hex}",
                    "category": category,
                },
                enabled=True,
                created_at=now,
                updated_at=now,
            )
            self._session.add(source)
            await self._session.flush()
            return source

        if source.source_type != ADMIN_SOURCE_TYPE:
            # 업로드 키는 upload/ 접두가 붙어 GitBook 키와 겹치지 않는다.
            raise DocumentNotRevisableError()

        # 같은 문서명의 콘솔 문서는 그 문서의 새 판 후보가 된다.
        metadata = source.metadata_ or {}
        document_id = metadata.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            document_id = f"admin-{uuid.uuid4().hex}"
        source.title = title
        source.metadata_ = {"document_id": document_id, "category": category}
        source.enabled = True
        source.updated_at = datetime.now(timezone.utc)
        await self._session.flush()
        return source

async def run_admin_ingestion(
    ingestion_run_id: int,
    raw_content: str,
    embedder_factory: Callable[[], OpenAIEmbedder] = OpenAIEmbedder,
) -> None:
    """독립 세션에서 업로드 원문을 판정하고 결과까지 기록한다.

    같은 내용 재업로드와 다른 문서와 같은 내용은 오류가 아니라 결과다.
    판을 만들지 않고 result_code 로 구분해 마감한다.

    진행 단계는 폴링 조회가 볼 수 있도록 단계마다 커밋한다.
    """

    async with get_session_factory()() as session:
        store = DocumentStore(session)
        failed_stage = IngestionStage.RECEIVING.value
        try:
            source_values = await _load_processing_source_values(
                session,
                ingestion_run_id,
            )
            if source_values is None:
                return
            await session.rollback()

            failed_stage = IngestionStage.NORMALIZING.value
            await store.set_stage(ingestion_run_id, IngestionStage.NORMALIZING)
            await session.commit()
            normalized_content = await asyncio.to_thread(
                _normalize_uploaded_markdown,
                raw_content,
            )
            normalized_content_hash = _sha256(normalized_content)

            decided = await _decide_without_new_version(
                session,
                store,
                ingestion_run_id,
                source_values["document_source_id"],
                normalized_content_hash,
            )
            if decided:
                return

            failed_stage = IngestionStage.PARSING.value
            await store.set_stage(ingestion_run_id, IngestionStage.PARSING)
            await session.commit()
            document, sections = await asyncio.to_thread(
                _parse_uploaded_document,
                raw_content,
                normalized_content,
                normalized_content_hash,
                **{
                    key: value
                    for key, value in source_values.items()
                    if key != "document_source_id"
                },
            )

            failed_stage = IngestionStage.CHUNKING.value
            await store.set_stage(ingestion_run_id, IngestionStage.CHUNKING)
            await session.commit()
            chunks = await asyncio.to_thread(
                _chunk_uploaded_document,
                document,
                sections,
            )

            failed_stage = IngestionStage.EMBEDDING.value
            await store.set_stage(ingestion_run_id, IngestionStage.EMBEDDING)
            await session.commit()
            embeddings = await prepare_chunk_embeddings(
                session,
                store,
                ingestion_run_id,
                chunks,
                embedder_factory(),
            )

            failed_stage = IngestionStage.PERSISTING.value
            await store.set_stage(ingestion_run_id, IngestionStage.PERSISTING)
            result_code = await _result_code_for_new_version(
                store,
                source_values["document_source_id"],
            )
            await store.complete_ingestion(
                ingestion_run_id,
                document,
                chunks,
                embeddings,
                result_code=result_code,
            )
            await session.commit()
        except _UploadedMarkdownInvalidError as error:
            await session.rollback()
            await _record_ingestion_failure(
                session,
                store,
                ingestion_run_id,
                error,
                failed_stage=failed_stage,
                error_code=INVALID_FILE,
            )
        except Exception as error:
            logger.exception(
                "관리자 Markdown 수집에 실패했습니다: ingestion_run_id=%s",
                ingestion_run_id,
            )
            await session.rollback()
            await _record_ingestion_failure(
                session,
                store,
                ingestion_run_id,
                error,
                failed_stage=failed_stage,
                error_code=INTERNAL_ERROR,
            )


async def _decide_without_new_version(
    session: AsyncSession,
    store: DocumentStore,
    ingestion_run_id: int,
    document_source_id: int,
    normalized_content_hash: str,
) -> bool:
    """판을 만들지 않고 끝나는 결과인지 판정하고, 그렇다면 마감한다."""

    latest = await store.find_latest_ready_version(document_source_id)
    if (
        latest is not None
        and latest.normalized_content_hash == normalized_content_hash
    ):
        await store.complete_without_new_version(
            ingestion_run_id,
            IngestionResultCode.NO_CHANGE,
        )
        await session.commit()
        return True

    duplicate = await store.find_duplicate_source(
        document_source_id,
        normalized_content_hash,
    )
    if duplicate is not None:
        await store.complete_without_new_version(
            ingestion_run_id,
            IngestionResultCode.DUPLICATE_CONTENT,
            duplicate_of_document_source_id=duplicate.id,
        )
        await session.commit()
        return True

    return False


async def _result_code_for_new_version(
    store: DocumentStore,
    document_source_id: int,
) -> IngestionResultCode:
    """첫 판이면 CREATED, 이미 판이 있으면 UPDATED 다."""

    latest = await store.find_latest_ready_version(document_source_id)
    return (
        IngestionResultCode.CREATED
        if latest is None
        else IngestionResultCode.UPDATED
    )


async def _load_processing_source_values(
    session: AsyncSession,
    ingestion_run_id: int,
) -> Optional[dict]:
    statement = (
        select(IngestionRun, DocumentSource)
        .join(
            DocumentSource,
            DocumentSource.id == IngestionRun.document_source_id,
        )
        .where(IngestionRun.id == ingestion_run_id)
    )
    row = (await session.execute(statement)).one_or_none()
    if row is None:
        logger.error("수집 실행을 찾을 수 없습니다: ingestion_run_id=%s", ingestion_run_id)
        return None

    run, source = row
    if run.status != ExecutionStatus.PROCESSING:
        logger.info(
            "이미 마감된 관리자 수집 실행을 건너뜁니다: ingestion_run_id=%s, status=%s",
            ingestion_run_id,
            run.status,
        )
        return None

    metadata = source.metadata_ or {}
    document_id = metadata.get("document_id")
    category = metadata.get("category")
    if not isinstance(document_id, str) or not document_id:
        raise RuntimeError("DocumentSource에 document_id metadata가 없습니다.")
    if source.title is None:
        raise RuntimeError("DocumentSource에 title이 없습니다.")
    if category is not None and not isinstance(category, str):
        raise RuntimeError("DocumentSource category metadata 형식이 올바르지 않습니다.")

    return {
        "document_source_id": source.id,
        "document_id": document_id,
        "title": source.title,
        "source_url": source.canonical_uri,
        "category": category,
    }


def _normalize_uploaded_markdown(raw_content: str) -> str:
    """업로드 원문을 정제한다. 결과가 비면 접수 뒤 실패로 다룬다."""

    try:
        normalized_content, _ = normalize_markdown(raw_content)
    except ValueError as error:
        raise _UploadedMarkdownInvalidError(str(error)) from error

    if not normalized_content.strip():
        raise _UploadedMarkdownInvalidError("정제 후 유효한 본문이 없습니다.")
    return normalized_content


def _parse_uploaded_document(
    raw_content: str,
    normalized_content: str,
    normalized_content_hash: str,
    *,
    document_id: str,
    title: str,
    source_url: str,
    category: Optional[str],
) -> tuple[NormalizedDocument, list]:
    """정제 문서를 만들고 Section 구조를 분석한다."""

    document = NormalizedDocument(
        document_id=document_id,
        title=title,
        source_url=source_url,
        category=category,
        content=normalized_content,
        raw_content_uri=None,
        raw_content_hash=_sha256(raw_content),
        normalized_content_hash=normalized_content_hash,
        raw_content=raw_content,
    )
    try:
        sections = parse_sections(document)
    except ValueError as error:
        raise _UploadedMarkdownInvalidError(str(error)) from error
    return document, sections


def _chunk_uploaded_document(
    document: NormalizedDocument,
    sections: list,
) -> list[RetrievalChunk]:
    """Section을 검색 단위 Chunk로 나눈다."""

    try:
        chunks = [
            RetrievalChunk.from_document_chunk(document, chunk)
            for chunk in create_chunks(sections)
        ]
    except ValueError as error:
        raise _UploadedMarkdownInvalidError(str(error)) from error

    chunks = [chunk for chunk in chunks if chunk.content.strip()]
    if not chunks:
        raise _UploadedMarkdownInvalidError(
            "정제와 청킹 후 유효한 본문이 없습니다."
        )
    return chunks


async def _record_ingestion_failure(
    session: AsyncSession,
    store: DocumentStore,
    ingestion_run_id: int,
    error: Exception,
    *,
    failed_stage: str,
    error_code: str,
) -> None:
    try:
        await store.fail_ingestion(
            ingestion_run_id,
            error,
            failed_stage=failed_stage,
            error_code=error_code,
        )
        await session.commit()
    except Exception:
        logger.exception(
            "관리자 수집 실패 로그를 마감하지 못했습니다: ingestion_run_id=%s",
            ingestion_run_id,
        )
        await session.rollback()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _chunk_stats_of(summary: dict) -> Optional[dict]:
    """실행 summary에 청크 통계가 모두 있으면 그대로 돌려준다."""

    keys = ("added", "changed", "deleted", "reused")
    if not all(key in summary for key in keys):
        return None
    return {key: summary[key] for key in keys}
