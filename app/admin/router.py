"""Admin Markdown 신규 업로드와 수집 상태 조회 endpoint."""

import asyncio
from pathlib import Path
from typing import Annotated, Callable, Optional

from fastapi import APIRouter, Depends, File, Request, UploadFile, status

from app.admin.dependencies import (
    get_admin_ingestion_service,
    get_chunk_embedder_factory,
    get_index_reindex_service,
)
from app.document.ingestion_service import (
    INTERNAL_ERROR,
    INVALID_FILE,
    AdminIngestionService,
    IngestionRunDetail,
    InvalidUploadFileError,
    UploadFileTooLargeError,
    run_admin_ingestion,
)
from app.admin.schema import (
    AdminDocumentUploadRequest,
    AdminIndexRunAcceptedResponse,
    AdminIndexRunFailedResponse,
    AdminIndexRunProcessingResponse,
    AdminIndexRunResponse,
    AdminIndexRunError,
    AdminIndexRunSuccessResponse,
    AdminIndexVersionSummary,
    IndexRunErrorCode,
    AdminError,
    AdminErrorCode,
    AdminErrorResponse,
    AdminIngestionAcceptedResponse,
    AdminIngestionFailedResponse,
    AdminIngestionProcessingResponse,
    AdminIngestionRunResponse,
    AdminIngestionStatus,
    AdminIngestionSuccessResponse,
)
from app.chat.dependencies import get_corpus_state
from app.core.task_registry import register_pipeline_task
from app.indexing.index_job import run_admin_index_job
from app.indexing.index_service import (
    IndexReindexService,
    IndexRunDetail,
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
            "`DOCUMENT_ALREADY_EXISTS`: 같은 문서명의 문서를 다시 업로드한 경우입니다. "
            "`JOB_IN_PROGRESS`: 다른 업로드나 재색인이 진행 중인 경우입니다."
        ),
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: {
        "model": AdminErrorResponse,
        "description": (
            "`INVALID_FILE`: .md 파일이 아니거나 UTF-8이 아니거나, "
            "파일 내용이 비어 있는 경우입니다."
        ),
    },
}


@router.post(
    "/documents",
    response_model=AdminIngestionAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ADMIN_ERROR_RESPONSES,
    summary="Markdown 신규 문서 업로드",
)
async def create_admin_document(
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
        title=upload.title,
        source_url=None if upload.source_url is None else str(upload.source_url),
        category=upload.category,
        filename=filename,
    )
    task = asyncio.create_task(
        run_admin_ingestion(
            accepted.ingestion_run_id,
            raw_content,
            embedder_factory,
        )
    )
    register_pipeline_task(http_request.app, task)
    return AdminIngestionAcceptedResponse(
        ingestionRunId=accepted.ingestion_run_id,
        documentId=accepted.document_source_id,
        status=AdminIngestionStatus.PROCESSING,
    )


@router.get(
    "/ingestion-runs/{ingestion_run_id}",
    response_model=AdminIngestionRunResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": AdminErrorResponse,
            "description": (
                "`NOT_FOUND`: 요청한 ingestionRunId에 해당하는 수집 실행이 "
                "존재하지 않는 경우입니다."
            ),
        }
    },
    summary="문서 수집 실행 상태 조회",
)
async def get_admin_ingestion_run(
    ingestion_run_id: int,
    service: AdminIngestionService = Depends(get_admin_ingestion_service),
) -> AdminIngestionRunResponse:
    detail = await service.get_ingestion_run(ingestion_run_id)
    return _to_ingestion_response(detail)


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


def _to_ingestion_response(detail: IngestionRunDetail) -> AdminIngestionRunResponse:
    common = {
        "ingestionRunId": detail.ingestion_run_id,
        "documentId": detail.document_source_id,
        "startedAt": detail.started_at,
    }
    if detail.status == ExecutionStatus.PROCESSING:
        return AdminIngestionProcessingResponse(
            **common,
            status=AdminIngestionStatus.PROCESSING,
        )
    if detail.status == ExecutionStatus.SUCCESS:
        if (
            detail.document_version_id is None
            or detail.version_no is None
            or detail.section_count is None
            or detail.chunk_count is None
            or detail.finished_at is None
        ):
            raise RuntimeError("SUCCESS 수집 실행의 결과 정보가 불완전합니다.")
        return AdminIngestionSuccessResponse(
            **common,
            status=AdminIngestionStatus.SUCCESS,
            documentVersionId=detail.document_version_id,
            versionNo=detail.version_no,
            changed=True,
            sectionCount=detail.section_count,
            chunkCount=detail.chunk_count,
            finishedAt=detail.finished_at,
        )
    if detail.status == ExecutionStatus.FAILED:
        if detail.finished_at is None:
            raise RuntimeError("FAILED 수집 실행에 finished_at이 없습니다.")
        code = detail.error_code or INTERNAL_ERROR
        try:
            external_code = AdminErrorCode(code)
        except ValueError:
            external_code = AdminErrorCode.INTERNAL_ERROR
        if code == INVALID_FILE and detail.error_message:
            message = detail.error_message
        else:
            message = "문서를 처리하는 중 오류가 발생했습니다."
        return AdminIngestionFailedResponse(
            **common,
            status=AdminIngestionStatus.FAILED,
            error=AdminError(code=external_code, message=message),
            finishedAt=detail.finished_at,
        )
    raise RuntimeError(f"지원하지 않는 수집 실행 상태입니다: {detail.status}")


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


@router.post(
    "/index-runs/{index_run_id}/retry-apply",
    response_model=AdminIndexRunAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=INDEX_RUN_ERROR_RESPONSES,
    summary="적용 다시 시도",
)
async def retry_apply_index_run(
    index_run_id: int,
    http_request: Request,
    service: IndexReindexService = Depends(get_index_reindex_service),
    corpus_state: CorpusState = Depends(get_corpus_state),
    embedder_factory: Callable[[], OpenAIEmbedder] = Depends(
        get_chunk_embedder_factory
    ),
) -> AdminIndexRunAcceptedResponse:
    """적용 단계에서 실패한 실행의 READY 후보에 적용만 다시 시도한다."""

    accepted = await service.start_retry_apply(index_run_id)
    _start_index_job(
        http_request,
        accepted.index_run_id,
        corpus_state,
        embedder_factory,
    )
    return _to_accepted_response(accepted)


@router.get(
    "/index-runs/{index_run_id}",
    response_model=AdminIndexRunResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": AdminErrorResponse,
            "description": (
                "`NOT_FOUND`: 요청한 indexRunId에 해당하는 실행이 존재하지 "
                "않는 경우입니다."
            ),
        }
    },
    summary="검색 반영 실행 상태 조회",
)
async def get_index_run(
    index_run_id: int,
    service: IndexReindexService = Depends(get_index_reindex_service),
) -> AdminIndexRunResponse:
    detail = await service.get_index_run(index_run_id)
    return _to_index_run_response(detail)


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
        retryOfIndexRunId=accepted.retry_of_index_run_id,
    )


def _to_index_version_summary(summary) -> AdminIndexVersionSummary:
    return AdminIndexVersionSummary(
        indexVersionId=summary.index_version_id,
        versionNo=summary.version_no,
        status=summary.status,
        activatedAt=summary.activated_at,
    )


def _to_index_run_response(detail: IndexRunDetail) -> AdminIndexRunResponse:
    if detail.status == ExecutionStatus.PROCESSING:
        return AdminIndexRunProcessingResponse(
            indexRunId=detail.index_run_id,
            groupId=detail.group_id,
            indexVersionId=detail.index_version_id,
            operationType=detail.operation_type,
            triggerType=detail.trigger_type,
            status=AdminIngestionStatus.PROCESSING,
            stage=detail.stage,
            startedAt=detail.started_at,
        )

    if detail.status == ExecutionStatus.FAILED:
        return AdminIndexRunFailedResponse(
            indexRunId=detail.index_run_id,
            groupId=detail.group_id,
            indexVersionId=detail.index_version_id,
            operationType=detail.operation_type,
            triggerType=detail.trigger_type,
            status=AdminIngestionStatus.FAILED,
            stage=detail.stage,
            error=AdminIndexRunError(
                code=_to_index_run_error_code(detail.error_code),
                message=detail.error_message or "검색 반영에 실패했습니다.",
            ),
            indexVersion=_to_index_version_summary(detail.index_version),
            retryable=bool(detail.retryable),
            startedAt=detail.started_at,
            finishedAt=detail.finished_at,
        )

    return AdminIndexRunSuccessResponse(
        indexRunId=detail.index_run_id,
        groupId=detail.group_id,
        indexVersionId=detail.index_version_id,
        operationType=detail.operation_type,
        triggerType=detail.trigger_type,
        status=AdminIngestionStatus.SUCCESS,
        stage=detail.stage,
        indexVersion=_to_index_version_summary(detail.index_version),
        previousIndexVersion=(
            None
            if detail.previous_index_version is None
            else _to_index_version_summary(detail.previous_index_version)
        ),
        documentCount=detail.document_count or 0,
        chunkCount=detail.chunk_count or 0,
        startedAt=detail.started_at,
        finishedAt=detail.finished_at,
    )


def _to_index_run_error_code(error_code: Optional[str]) -> IndexRunErrorCode:
    """기록되지 않았거나 모르는 코드는 내부 오류로 내린다."""

    try:
        return IndexRunErrorCode(error_code)
    except ValueError:
        return IndexRunErrorCode.INTERNAL_ERROR
