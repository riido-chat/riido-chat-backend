"""Admin Markdown 업로드와 검색 반영, GitBook 수집 endpoint."""

from pathlib import Path
from typing import Annotated, Callable

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.admin.dependencies import (
    get_admin_ingestion_service,
    get_chunk_embedder_factory,
    get_document_group_service,
    get_index_reindex_service,
    get_recollect_service,
)
from app.document.ingestion_service import (
    AdminIngestionService,
    IngestionFailedError,
    InvalidUploadFileError,
    UploadFileTooLargeError,
    run_admin_ingestion,
)
from app.admin.group_service import (
    DocumentGroupService,
    GroupDetail,
    GroupSummary,
)
from app.admin.schema import (
    AdminActiveIndexVersion,
    AdminDocumentGroupDetailResponse,
    AdminDocumentGroupListResponse,
    AdminDocumentGroupSummary,
    AdminGroupDocument,
    AdminGroupInfo,
    AdminGroupSourceItem,
    AdminGroupSummary,
    AdminChunkStats,
    AdminDocumentRevisionRequest,
    AdminDocumentUploadRequest,
    AdminIndexVersionSummary,
    AdminReindexResultResponse,
    AdminUploadResultResponse,
    AdminErrorResponse,
    AdminGitBookSyncRequest,
    AdminGitBookSyncResultResponse,
    AdminRecollectCounts,
    AdminRecollectFailure,
)
from app.chat.dependencies import get_corpus_state
from app.document.recollect import run_recollect_batch
from app.document.recollect_service import RecollectService
from app.indexing.index_job import run_admin_index_job
from app.indexing.index_service import (
    IndexReindexService,
    IndexRunFailedError,
)
from app.retrieval.corpus_state import CorpusState
from app.retrieval.embedding import OpenAIEmbedder
from app.database.models import ExecutionStatus


router = APIRouter(prefix="/api/admin", tags=["admin"])

MAX_MARKDOWN_FILE_BYTES = 5 * 1024 * 1024
UPLOAD_READ_CHUNK_BYTES = 64 * 1024
ADMIN_ERROR_RESPONSES = {
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {
        "model": AdminErrorResponse,
        "description": (
            "`FILE_TOO_LARGE`: 업로드한 Markdown 파일이 5MB를 초과한 경우입니다."
        ),
    },
    status.HTTP_409_CONFLICT: {
        "model": AdminErrorResponse,
        "description": (
            "`DOCUMENT_ALREADY_EXISTS`: 같은 이름의 콘솔 문서에 READY 판이 "
            "있는 경우입니다. "
            "`DUPLICATE_CONTENT`: 그룹 안 다른 콘솔 문서와 본문이 같은 "
            "경우입니다. "
            "`NO_CHANGE`: 수정본이 대상 문서의 직전 판과 본문이 같은 "
            "경우입니다. "
            "`JOB_IN_PROGRESS`: 같은 그룹에 진행 중 작업이 있는 경우입니다. "
            "`DOCUMENT_NOT_REVISABLE`: GitBook 문서에 수정본을 올린 경우입니다."
        ),
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "model": AdminErrorResponse,
        "description": (
            "`INVALID_FILE`: .md 파일이 아니거나 UTF-8이 아니거나, "
            "파일 내용이 비어 있거나, 문서명을 정규화한 결과가 "
            "빈 문자열인 경우입니다."
        ),
    },
    status.HTTP_404_NOT_FOUND: {
        "model": AdminErrorResponse,
        "description": "`NOT_FOUND`: 대상 그룹 또는 문서가 없는 경우입니다.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": AdminErrorResponse,
        "description": (
            "`INVALID_FILE`: 접수 뒤 본문을 처리할 수 없는 경우입니다. "
            "`INTERNAL_ERROR`: 그 밖의 처리 중 실패입니다."
        ),
    },
}


@router.post(
    "/document-groups/{group_id}/documents",
    response_model=AdminUploadResultResponse,
    status_code=status.HTTP_200_OK,
    responses=ADMIN_ERROR_RESPONSES,
    summary="Markdown 신규 문서 업로드",
)
async def create_admin_document(
    group_id: int,
    upload: Annotated[AdminDocumentUploadRequest, File()],
    service: AdminIngestionService = Depends(get_admin_ingestion_service),
    embedder_factory: Callable[[], OpenAIEmbedder] = Depends(
        get_chunk_embedder_factory
    ),
) -> AdminUploadResultResponse:
    """파일을 검증하고 임베딩까지 끝낸 뒤 결과를 돌려준다."""

    filename, raw_content = await _read_markdown_file(upload.file)
    accepted = await service.start_new_document(
        group_id=group_id,
        title=upload.title,
        filename=filename,
    )
    return await _run_ingestion(service, accepted, raw_content, embedder_factory)


@router.post(
    "/documents/{document_id}/versions",
    response_model=AdminUploadResultResponse,
    status_code=status.HTTP_200_OK,
    responses=ADMIN_ERROR_RESPONSES,
    summary="Markdown 수정본 업로드",
)
async def create_admin_document_version(
    document_id: int,
    upload: Annotated[AdminDocumentRevisionRequest, File()],
    service: AdminIngestionService = Depends(get_admin_ingestion_service),
    embedder_factory: Callable[[], OpenAIEmbedder] = Depends(
        get_chunk_embedder_factory
    ),
) -> AdminUploadResultResponse:
    """대상 문서를 고정하고 파일만 받아 새 판을 만든다."""

    filename, raw_content = await _read_markdown_file(upload.file)
    accepted = await service.start_document_revision(
        document_id=document_id,
        filename=filename,
    )
    return await _run_ingestion(service, accepted, raw_content, embedder_factory)


async def _run_ingestion(
    service: AdminIngestionService,
    accepted,
    raw_content: str,
    embedder_factory: Callable[[], OpenAIEmbedder],
) -> AdminUploadResultResponse:
    """파이프라인을 요청 안에서 끝내고 결과를 응답으로 만든다.

    run_admin_ingestion 은 실패를 실행 기록에 남기고 예외를 밖으로 던지지
    않는다. 기록을 다시 읽어 성공이면 결과를, 실패면 오류를 내보낸다.
    """

    await run_admin_ingestion(
        accepted.ingestion_run_id,
        raw_content,
        embedder_factory,
    )
    detail = await service.read_finished_run(accepted.ingestion_run_id)
    if detail.status == ExecutionStatus.FAILED:
        raise IngestionFailedError(detail.error_code, detail.error_message)

    return AdminUploadResultResponse(
        ingestionRunId=detail.ingestion_run_id,
        documentId=detail.document_source_id,
        documentVersionId=detail.document_version_id,
        versionNo=detail.version_no,
        sectionCount=detail.section_count,
        chunkCount=detail.chunk_count,
        chunkStats=AdminChunkStats(**detail.chunk_stats),
    )


async def _read_markdown_file(upload_file: UploadFile) -> tuple[str, str]:
    filename = Path(upload_file.filename or "").name
    if Path(filename).suffix.lower() != ".md":
        await upload_file.close()
        raise InvalidUploadFileError(".md 확장자의 Markdown 파일만 업로드할 수 있습니다.")

    content = bytearray()
    try:
        while True:
            chunk = await upload_file.read(UPLOAD_READ_CHUNK_BYTES)
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > MAX_MARKDOWN_FILE_BYTES:
                raise UploadFileTooLargeError()
    finally:
        await upload_file.close()

    try:
        raw_content = bytes(content).decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise InvalidUploadFileError("UTF-8 Markdown 파일만 업로드할 수 있습니다.") from error
    if not raw_content.strip():
        raise InvalidUploadFileError("빈 Markdown 파일은 업로드할 수 없습니다.")
    return filename, raw_content


INDEX_RUN_ERROR_RESPONSES = {
    status.HTTP_404_NOT_FOUND: {
        "model": AdminErrorResponse,
        "description": "`NOT_FOUND`: 대상 그룹이 존재하지 않는 경우입니다.",
    },
    status.HTTP_409_CONFLICT: {
        "model": AdminErrorResponse,
        "description": (
            "`JOB_IN_PROGRESS`: 같은 그룹에 실행 중 작업이 있는 경우입니다. "
            "`REINDEX_NOT_REQUIRED`: 반영할 변경이 없는 경우입니다. "
            "`NO_READY_DOCUMENTS`: 준비된 문서가 없는 경우입니다."
        ),
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": AdminErrorResponse,
        "description": (
            "`INTERNAL_ERROR`: 처리 중 실패입니다. 원인은 index_runs 의 "
            "error_code 에 남고 응답에서는 구분하지 않습니다."
        ),
    },
}


@router.post(
    "/document-groups/{group_id}/reindex",
    response_model=AdminReindexResultResponse,
    status_code=status.HTTP_200_OK,
    responses=INDEX_RUN_ERROR_RESPONSES,
    summary="검색에 반영하기",
)
async def start_reindex(
    group_id: int,
    service: IndexReindexService = Depends(get_index_reindex_service),
    corpus_state: CorpusState = Depends(get_corpus_state),
    embedder_factory: Callable[[], OpenAIEmbedder] = Depends(
        get_chunk_embedder_factory
    ),
) -> AdminReindexResultResponse:
    """최신 READY 문서 조합으로 후보를 만들고 ACTIVE 전환까지 마친다."""

    accepted = await service.start_reindex(group_id)
    await run_admin_index_job(
        accepted.index_run_id,
        corpus_state,
        embedder_factory,
    )
    detail = await service.read_finished_run(accepted.index_run_id)
    if detail.status == ExecutionStatus.FAILED:
        raise IndexRunFailedError(detail.error_message)

    return AdminReindexResultResponse(
        indexRunId=detail.index_run_id,
        indexVersion=_to_index_version_summary(detail.index_version),
        previousIndexVersion=(
            None
            if detail.previous_index_version is None
            else _to_index_version_summary(detail.previous_index_version)
        ),
    )


def _to_index_version_summary(summary) -> AdminIndexVersionSummary:
    return AdminIndexVersionSummary(
        indexVersionId=summary.index_version_id,
        versionNo=summary.version_no,
    )


RECOLLECT_ERROR_RESPONSES = {
    status.HTTP_404_NOT_FOUND: {
        "model": AdminErrorResponse,
        "description": "`NOT_FOUND`: 대상 그룹이 없는 경우입니다.",
    },
    status.HTTP_409_CONFLICT: {
        "model": AdminErrorResponse,
        "description": "`JOB_IN_PROGRESS`: 같은 그룹에 진행 중 작업이 있는 경우입니다.",
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "model": AdminErrorResponse,
        "description": "`INVALID_REQUEST`: `sourceUrl` 이 https 가 아닌 경우입니다.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": AdminErrorResponse,
        "description": (
            "`INTERNAL_ERROR`: 처리 중 실패입니다. 페이지 단위 실패는 여기 "
            "오지 않고 `200` 응답의 `failures` 로 갑니다."
        ),
    },
    status.HTTP_502_BAD_GATEWAY: {
        "model": AdminErrorResponse,
        "description": (
            "`SOURCE_LIST_FAILED`: GitBook 페이지 목록을 읽지 못한 "
            "경우입니다. 수집을 시작하지 않습니다."
        ),
    },
}


@router.post(
    "/document-groups/{group_id}/gitbook-sync",
    response_model=AdminGitBookSyncResultResponse,
    status_code=status.HTTP_200_OK,
    responses=RECOLLECT_ERROR_RESPONSES,
    summary="GitBook 수집",
)
async def start_gitbook_sync(
    group_id: int,
    request: AdminGitBookSyncRequest,
    service: RecollectService = Depends(get_recollect_service),
    embedder_factory: Callable[[], OpenAIEmbedder] = Depends(
        get_chunk_embedder_factory
    ),
) -> AdminGitBookSyncResultResponse:
    """페이지 목록과 원문을 읽어 페이지마다 처리하고 집계를 돌려준다.

    같은 루트로 다시 부르면 재수집이 된다. 페이지 하나가 실패해도 계속
    진행하고 counts.failed 로 센다.
    """

    accepted = await service.start_sync(group_id, request.source_url)
    await run_recollect_batch(accepted.batch_id, embedder_factory)
    detail = await service.read_finished_batch(accepted.batch_id)

    counts = detail.counts or {}
    return AdminGitBookSyncResultResponse(
        groupSourceId=detail.group_source_id,
        rootUrl=detail.root_url,
        counts=AdminRecollectCounts(
            total=counts.get("total", 0),
            created=counts.get("created", 0),
            updated=counts.get("updated", 0),
            noChange=counts.get("no_change", 0),
            removed=counts.get("removed", 0),
            failed=counts.get("failed", 0),
        ),
        failures=[
            AdminRecollectFailure(
                documentKey=failure.document_key,
                title=failure.title,
                ingestionRunId=failure.ingestion_run_id,
                message=failure.message,
            )
            for failure in detail.failures
        ],
    )


@router.get(
    "/document-groups",
    response_model=AdminDocumentGroupListResponse,
    status_code=status.HTTP_200_OK,
    summary="문서 그룹 목록 조회",
)
async def list_document_groups(
    service: DocumentGroupService = Depends(get_document_group_service),
) -> AdminDocumentGroupListResponse:
    summaries = await service.list_groups()
    return AdminDocumentGroupListResponse(
        groups=[_to_group_summary(summary) for summary in summaries],
    )


@router.get(
    "/document-groups/{group_id}",
    response_model=AdminDocumentGroupDetailResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": AdminErrorResponse,
            "description": "`NOT_FOUND`: 존재하지 않는 groupId 입니다.",
        }
    },
    summary="문서 그룹 상세 조회",
)
async def get_document_group(
    group_id: int,
    service: DocumentGroupService = Depends(get_document_group_service),
) -> AdminDocumentGroupDetailResponse:
    return _to_group_detail(await service.get_group_detail(group_id))


def _to_group_summary(summary: GroupSummary) -> AdminDocumentGroupSummary:
    return AdminDocumentGroupSummary(
        groupId=summary.group_id,
        groupKey=summary.group_key,
        name=summary.name,
        consumerKey=summary.consumer_key,
        documentCount=summary.document_count,
        activeIndexVersionNo=summary.active_index_version_no,
        searchStatus=summary.search_status,
    )


def _to_group_detail(detail: GroupDetail) -> AdminDocumentGroupDetailResponse:
    active = detail.active_index_version
    return AdminDocumentGroupDetailResponse(
        group=AdminGroupInfo(
            groupId=detail.group_id,
            groupKey=detail.group_key,
            name=detail.name,
            consumerKey=detail.consumer_key,
        ),
        sources=[
            AdminGroupSourceItem(
                groupSourceId=source.group_source_id,
                provider=source.provider,
                rootUrl=source.root_url,
                enabled=source.enabled,
                documentCount=source.document_count,
            )
            for source in detail.sources
        ],
        summary=AdminGroupSummary(
            activeIndexVersion=(
                None
                if active is None
                else AdminActiveIndexVersion(
                    indexVersionId=active.index_version_id,
                    versionNo=active.version_no,
                )
            ),
            pendingCount=detail.pending_count,
            searchStatus=detail.search_status,
        ),
        documents=[
            AdminGroupDocument(
                documentId=document.document_id,
                documentKey=document.document_key,
                title=document.title,
                sourceType=document.source_type,
                groupSourceId=document.group_source_id,
                documentVersionNo=document.document_version_no,
                appliedVersionNo=document.applied_version_no,
                appliedStatus=document.applied_status,
            )
            for document in detail.documents
        ],
        jobInProgress=detail.job_in_progress,
    )
