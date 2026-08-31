"""Admin Markdown 신규 업로드와 수집 상태 조회 endpoint."""

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile, status

from app.admin.dependencies import get_admin_ingestion_service
from app.admin.ingestion_service import (
    INTERNAL_ERROR,
    INVALID_FILE,
    AdminIngestionService,
    IngestionRunDetail,
    InvalidUploadFileError,
    UploadFileTooLargeError,
    run_admin_ingestion,
)
from app.api.admin_schema import (
    AdminDocumentUploadRequest,
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
from app.core.task_registry import register_pipeline_task
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
            "`DOCUMENT_ALREADY_EXISTS`: 같은 sourceUrl의 문서를 다시 업로드한 경우입니다. "
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
) -> AdminIngestionAcceptedResponse:
    """파일을 검증한 뒤 수집 실행을 확정하고 background task를 시작한다."""

    filename, raw_content = await _read_markdown_file(upload.file)
    accepted = await service.start_new_document(
        title=upload.title,
        source_url=str(upload.source_url),
        category=upload.category,
        filename=filename,
    )
    task = asyncio.create_task(
        run_admin_ingestion(accepted.ingestion_run_id, raw_content)
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
