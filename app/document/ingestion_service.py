"""관리자 Markdown 신규 업로드의 접수와 수집 수명주기를 관리한다."""

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

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
    ExecutionStatus,
    IndexRun,
    IngestionRun,
)
from app.database.session import get_session_factory
from app.document.models import NormalizedDocument
from app.document.clean import normalize_markdown
from app.retrieval.corpus import build_document_retrieval_chunks
from app.retrieval.models import RetrievalChunk
from app.document.document_store import PARSER_NAME, PARSER_VERSION, DocumentStore


logger = logging.getLogger(__name__)

ADMIN_JOB_LOCK_KEY = 0x524949444F
ADMIN_SOURCE_TYPE = SOURCE_TYPE_UPLOAD
ADMIN_TRIGGER_TYPE = "ADMIN_UPLOAD"
DOCUMENT_SOURCE_KEY_CONSTRAINT = "uq_document_sources_document_group_id_document_key"

INVALID_FILE = "INVALID_FILE"
FILE_TOO_LARGE = "FILE_TOO_LARGE"
DOCUMENT_ALREADY_EXISTS = "DOCUMENT_ALREADY_EXISTS"
JOB_IN_PROGRESS = "JOB_IN_PROGRESS"
NOT_FOUND = "NOT_FOUND"
INTERNAL_ERROR = "INTERNAL_ERROR"


class AdminApiError(RuntimeError):
    """Admin API가 상태 코드와 오류 응답으로 변환할 수 있는 예외."""

    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class InvalidUploadFileError(AdminApiError):
    def __init__(self, message: str = "올바른 Markdown 파일이 아닙니다.") -> None:
        super().__init__(INVALID_FILE, message, HTTPStatus.UNPROCESSABLE_ENTITY)


class UploadFileTooLargeError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            FILE_TOO_LARGE,
            "Markdown 파일은 5MB 이하여야 합니다.",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )


class DocumentAlreadyExistsError(AdminApiError):
    def __init__(self) -> None:
        super().__init__(
            DOCUMENT_ALREADY_EXISTS,
            "같은 문서명의 문서가 이미 존재합니다.",
            HTTPStatus.CONFLICT,
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
class IngestionRunDetail:
    ingestion_run_id: int
    document_source_id: int
    status: ExecutionStatus
    document_version_id: Optional[int]
    version_no: Optional[int]
    section_count: Optional[int]
    chunk_count: Optional[int]
    error_code: Optional[str]
    error_message: Optional[str]
    started_at: datetime
    finished_at: Optional[datetime]


class AdminIngestionService:
    """요청 세션 안에서 신규 문서 접수와 실행 조회를 수행한다."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start_new_document(
        self,
        *,
        title: str,
        source_url: Optional[str] = None,
        category: Optional[str],
        filename: str,
    ) -> AcceptedIngestion:
        """전역 작업 gate 안에서 Source와 PROCESSING 실행을 확정한다.

        문서는 (문서 그룹, document_key)로 식별하므로 요청의 source_url은
        받기만 하고 사용하지 않는다. 입력 필드 제거는 후속 계약 변경에서 한다.
        """

        try:
            await self._acquire_admin_job_gate()
            await self._ensure_no_processing_job()
            source = await self._find_or_create_new_source(
                title=title,
                category=category,
            )
            now = datetime.now(timezone.utc)
            run = IngestionRun(
                document_source_id=source.id,
                trigger_type=ADMIN_TRIGGER_TYPE,
                parser_name=PARSER_NAME,
                parser_version=PARSER_VERSION,
                status=ExecutionStatus.PROCESSING,
                summary={"stage": "RECEIVED", "filename": filename},
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
        except IntegrityError as error:
            await self._session.rollback()
            if self._constraint_name(error) == DOCUMENT_SOURCE_KEY_CONSTRAINT:
                raise DocumentAlreadyExistsError() from error
            raise
        except Exception:
            await self._session.rollback()
            raise

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
            document_version_id=None if version is None else version.id,
            version_no=None if version is None else version.version_no,
            section_count=summary.get("section_count"),
            chunk_count=summary.get("chunk_count"),
            error_code=summary.get("error_code"),
            error_message=run.error_message,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )

    async def _acquire_admin_job_gate(self) -> None:
        await self._session.execute(
            select(func.pg_advisory_xact_lock(ADMIN_JOB_LOCK_KEY))
        )

    async def _ensure_no_processing_job(self) -> None:
        ingestion_run_id = await self._session.scalar(
            select(IngestionRun.id)
            .where(IngestionRun.status == ExecutionStatus.PROCESSING)
            .limit(1)
        )
        index_run_id = await self._session.scalar(
            select(IndexRun.id)
            .where(IndexRun.status == ExecutionStatus.PROCESSING)
            .limit(1)
        )
        if ingestion_run_id is not None or index_run_id is not None:
            raise AdminJobInProgressError()

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

    async def _find_or_create_new_source(
        self,
        *,
        title: str,
        category: Optional[str],
    ) -> DocumentSource:
        group = await self._get_document_group()
        document_key = build_upload_document_key(title)
        source = await self._session.scalar(
            select(DocumentSource).where(
                DocumentSource.document_group_id == group.id,
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

        version_count = await self._session.scalar(
            select(func.count())
            .select_from(DocumentVersion)
            .where(DocumentVersion.document_source_id == source.id)
        )
        if source.source_type != ADMIN_SOURCE_TYPE or version_count:
            raise DocumentAlreadyExistsError()

        # 이전 신규 업로드가 실패해 Source만 남은 경우 같은 문서로 재시도한다.
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

    @staticmethod
    def _constraint_name(error: IntegrityError) -> Optional[str]:
        original = getattr(error, "orig", None)
        diagnostic = getattr(original, "diag", None)
        return getattr(diagnostic, "constraint_name", None)


async def run_admin_ingestion(
    ingestion_run_id: int,
    raw_content: str,
) -> None:
    """독립 세션에서 업로드 원문을 READY DocumentVersion까지 처리한다."""

    async with get_session_factory()() as session:
        store = DocumentStore(session)
        failed_stage = "LOADING"
        try:
            source_values = await _load_processing_source_values(
                session,
                ingestion_run_id,
            )
            if source_values is None:
                return
            await session.rollback()

            failed_stage = "NORMALIZING"
            document, chunks = await asyncio.to_thread(
                _build_uploaded_document,
                raw_content,
                **source_values,
            )

            failed_stage = "PERSISTING"
            await store.complete_ingestion(
                ingestion_run_id,
                document,
                chunks,
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


async def _load_processing_source_values(
    session: AsyncSession,
    ingestion_run_id: int,
) -> Optional[dict[str, Optional[str]]]:
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
        "document_id": document_id,
        "title": source.title,
        "source_url": source.canonical_uri,
        "category": category,
    }


def _build_uploaded_document(
    raw_content: str,
    *,
    document_id: str,
    title: str,
    source_url: str,
    category: Optional[str],
) -> tuple[NormalizedDocument, list[RetrievalChunk]]:
    try:
        normalized_content, _ = normalize_markdown(raw_content)
        document = NormalizedDocument(
            document_id=document_id,
            title=title,
            source_url=source_url,
            category=category,
            content=normalized_content,
            raw_content_uri=None,
            raw_content_hash=_sha256(raw_content),
            normalized_content_hash=_sha256(normalized_content),
            raw_content=raw_content,
        )
        chunks = [
            chunk
            for chunk in build_document_retrieval_chunks(document)
            if chunk.content.strip()
        ]
    except ValueError as error:
        raise _UploadedMarkdownInvalidError(str(error)) from error

    if not chunks:
        raise _UploadedMarkdownInvalidError(
            "정제·청킹 후 유효한 본문이 없습니다."
        )
    return document, chunks


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
