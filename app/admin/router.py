"""Admin Markdown 업로드와 검색 반영, GitBook 수집 endpoint."""

import asyncio
from pathlib import Path
from typing import Annotated, Callable

from fastapi import APIRouter, Depends, File, Request, UploadFile, status

from app.admin.dependencies import (
    get_admin_ingestion_service,
    get_chunk_embedder_factory,
    get_document_group_service,
    get_index_reindex_service,
    get_recollect_service,
)
from app.document.ingestion_service import (
    AdminIngestionService,
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
    AdminDocumentRevisionRequest,
    AdminDocumentUploadRequest,
    AdminIndexRunAcceptedResponse,
    AdminErrorResponse,
    AdminIngestionAcceptedResponse,
    AdminIngestionStatus,
    AdminGitBookSyncRequest,
    AdminRecollectAcceptedResponse,
    IngestionStageValue,
    RecollectStageValue,
)
from app.chat.dependencies import get_corpus_state
from app.core.task_registry import register_pipeline_task
from app.document.recollect import run_recollect_batch
from app.document.recollect_service import RecollectService
from app.indexing.index_job import run_admin_index_job
from app.indexing.index_service import IndexReindexService
from app.retrieval.corpus_state import CorpusState
from app.retrieval.embedding import OpenAIEmbedder


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
            "`JOB_IN_PROGRESS`: 같은 그룹에 진행 중 작업이 있는 경우입니다. "
            "`DOCUMENT_NOT_REVISABLE`: GitBook 문서에 수정본을 올린 경우입니다."
        ),
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "model": AdminErrorResponse,
        "description": (
            "`INVALID_FILE`: .md 파일이 아니거나 UTF-8이 아니거나, "
            "파일 내용이 비어 있는 경우입니다. 본문에 `stage`가 함께 붙습니다."
        ),
    },
    status.HTTP_404_NOT_FOUND: {
        "model": AdminErrorResponse,
        "description": "`NOT_FOUND`: 대상 그룹 또는 문서가 없는 경우입니다.",
    },
}


@router.post(
    "/document-groups/{group_id}/documents",
    response_model=AdminIngestionAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ADMIN_ERROR_RESPONSES,
    summary="Markdown 신규 문서 업로드",
)
async def create_admin_document(
    group_id: int,
    upload: Annotated[AdminDocumentUploadRequest, File()],
    http_request: Request,
    service: AdminIngestionService = Depends(get_admin_ingestion_service),
    embedder_factory: Callable[[], OpenAIEmbedder] = Depends(
        get_chunk_embedder_factory
    ),
) -> AdminIngestionAcceptedResponse:
    """파일을 검증한 뒤 수집 실행을 확정하고 background task를 시작한다."""

    filename, raw_content = await _read_markdown_file(upload.file)
    accepted = await service.start_new_document(
        group_id=group_id,
        title=upload.title,
        category=upload.category,
        filename=filename,
    )
    _start_ingestion_job(
        http_request,
        accepted.ingestion_run_id,
        raw_content,
        embedder_factory,
    )
    return _to_accepted_ingestion_response(accepted)


@router.post(
    "/documents/{document_id}/versions",
    response_model=AdminIngestionAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ADMIN_ERROR_RESPONSES,
    summary="Markdown 수정본 업로드",
)
async def create_admin_document_version(
    document_id: int,
    upload: Annotated[AdminDocumentRevisionRequest, File()],
    http_request: Request,
    service: AdminIngestionService = Depends(get_admin_ingestion_service),
    embedder_factory: Callable[[], OpenAIEmbedder] = Depends(
        get_chunk_embedder_factory
    ),
) -> AdminIngestionAcceptedResponse:
    """대상 문서를 고정하고 파일만 받아 새 판 후보를 접수한다."""

    filename, raw_content = await _read_markdown_file(upload.file)
    accepted = await service.start_document_revision(
        document_id=document_id,
        filename=filename,
    )
    _start_ingestion_job(
        http_request,
        accepted.ingestion_run_id,
        raw_content,
        embedder_factory,
    )
    return _to_accepted_ingestion_response(accepted)


def _start_ingestion_job(
    http_request: Request,
    ingestion_run_id: int,
    raw_content: str,
    embedder_factory: Callable[[], OpenAIEmbedder],
) -> None:
    task = asyncio.create_task(
        run_admin_ingestion(ingestion_run_id, raw_content, embedder_factory)
    )
    register_pipeline_task(http_request.app, task)


def _to_accepted_ingestion_response(accepted) -> AdminIngestionAcceptedResponse:
    return AdminIngestionAcceptedResponse(
        ingestionRunId=accepted.ingestion_run_id,
        documentId=accepted.document_source_id,
        status=AdminIngestionStatus.PROCESSING,
        stage=IngestionStageValue.RECEIVING,
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
        "description": "`NOT_FOUND`: 대상 그룹 또는 실행이 존재하지 않는 경우입니다.",
    },
    status.HTTP_409_CONFLICT: {
        "model": AdminErrorResponse,
        "description": (
            "`JOB_IN_PROGRESS`: 같은 그룹에 실행 중 작업이 있는 경우입니다. "
            "`REINDEX_NOT_REQUIRED`: 반영할 변경이 없는 경우입니다. "
            "`NO_READY_DOCUMENTS`: 준비된 문서가 없는 경우입니다. "
            "`RETRY_NOT_ALLOWED`: 적용 단계 실패가 아니거나 후보가 READY가 "
            "아닌 경우입니다."
        ),
    },
}


@router.post(
    "/document-groups/{group_id}/reindex",
    response_model=AdminIndexRunAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=INDEX_RUN_ERROR_RESPONSES,
    summary="검색에 반영하기",
)
async def start_reindex(
    group_id: int,
    http_request: Request,
    service: IndexReindexService = Depends(get_index_reindex_service),
    corpus_state: CorpusState = Depends(get_corpus_state),
    embedder_factory: Callable[[], OpenAIEmbedder] = Depends(
        get_chunk_embedder_factory
    ),
) -> AdminIndexRunAcceptedResponse:
    """최신 READY 문서 조합으로 후보 색인을 만들고 적용까지 진행한다."""

    accepted = await service.start_reindex(group_id)
    _start_index_job(
        http_request,
        accepted.index_run_id,
        corpus_state,
        embedder_factory,
    )
    return _to_accepted_response(accepted)


def _start_index_job(
    http_request: Request,
    index_run_id: int,
    corpus_state: CorpusState,
    embedder_factory: Callable[[], OpenAIEmbedder],
) -> None:
    task = asyncio.create_task(
        run_admin_index_job(index_run_id, corpus_state, embedder_factory)
    )
    register_pipeline_task(http_request.app, task)


def _to_accepted_response(accepted) -> AdminIndexRunAcceptedResponse:
    return AdminIndexRunAcceptedResponse(
        indexRunId=accepted.index_run_id,
        indexVersionId=accepted.index_version_id,
        groupId=accepted.group_id,
        operationType=accepted.operation_type,
        triggerType=accepted.trigger_type,
        status=AdminIngestionStatus.PROCESSING,
        stage=accepted.stage,
    )


RECOLLECT_ERROR_RESPONSES = {
    status.HTTP_404_NOT_FOUND: {
        "model": AdminErrorResponse,
        "description": "`NOT_FOUND`: 대상 그룹 또는 배치가 없는 경우입니다.",
    },
    status.HTTP_409_CONFLICT: {
        "model": AdminErrorResponse,
        "description": "`JOB_IN_PROGRESS`: 같은 그룹에 진행 중 작업이 있는 경우입니다.",
    },
    status.HTTP_502_BAD_GATEWAY: {
        "model": AdminErrorResponse,
        "description": (
            "`SOURCE_LIST_FAILED`: docs.riido.io 페이지 목록을 읽지 못한 "
            "경우입니다. 배치를 시작하지 않습니다."
        ),
    },
}


@router.post(
    "/document-groups/{group_id}/gitbook-sync",
    response_model=AdminRecollectAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=RECOLLECT_ERROR_RESPONSES,
    summary="GitBook 수집",
)
async def start_gitbook_sync(
    group_id: int,
    request: AdminGitBookSyncRequest,
    http_request: Request,
    service: RecollectService = Depends(get_recollect_service),
    embedder_factory: Callable[[], OpenAIEmbedder] = Depends(
        get_chunk_embedder_factory
    ),
) -> AdminRecollectAcceptedResponse:
    """루트 URL의 페이지 목록을 읽어 페이지별 실행을 만들고 배치를 시작한다.

    같은 루트로 다시 부르면 재탐색이 된다.
    """

    accepted = await service.start_sync(group_id, request.source_url)
    task = asyncio.create_task(
        run_recollect_batch(accepted.batch_id, embedder_factory)
    )
    register_pipeline_task(http_request.app, task)
    return AdminRecollectAcceptedResponse(
        batchId=accepted.batch_id,
        groupId=accepted.group_id,
        groupSourceId=accepted.group_source_id,
        rootUrl=accepted.root_url,
        status=AdminIngestionStatus.PROCESSING,
        stage=RecollectStageValue.PROCESSING,
        pageCount=accepted.page_count,
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
