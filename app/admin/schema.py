"""Admin 문서 업로드와 수집 상태 조회 HTTP DTO."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from fastapi import UploadFile
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator

from app.chat.schema import HTTP_DTO_CONFIG


ADMIN_UPLOAD_DTO_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    populate_by_name=True,
    arbitrary_types_allowed=True,
)


class AdminDocumentUploadRequest(BaseModel):
    """POST /api/admin/documents multipart 요청."""

    model_config = ADMIN_UPLOAD_DTO_CONFIG

    title: str = Field(min_length=1, max_length=300)
    # 문서는 (문서 그룹, document_key)로 식별한다. sourceUrl은 선택 입력으로만 남긴다.
    source_url: Optional[AnyHttpUrl] = Field(
        default=None, alias="sourceUrl", max_length=1_000
    )
    category: Optional[str] = Field(default=None, max_length=100)
    file: UploadFile

    @field_validator("title", mode="before")
    @classmethod
    def strip_title(cls, title: object) -> object:
        if isinstance(title, str):
            return title.strip()
        return title

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, category: object) -> object:
        if isinstance(category, str):
            return category.strip() or None
        return category


class AdminIngestionStatus(str, Enum):
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class AdminErrorCode(str, Enum):
    INVALID_FILE = "INVALID_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_ALREADY_EXISTS = "DOCUMENT_ALREADY_EXISTS"
    JOB_IN_PROGRESS = "JOB_IN_PROGRESS"
    REINDEX_NOT_REQUIRED = "REINDEX_NOT_REQUIRED"
    NO_READY_DOCUMENTS = "NO_READY_DOCUMENTS"
    RETRY_NOT_ALLOWED = "RETRY_NOT_ALLOWED"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AdminError(BaseModel):
    model_config = HTTP_DTO_CONFIG

    code: AdminErrorCode
    message: str


class AdminErrorResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    code: AdminErrorCode
    message: str


class AdminIngestionAcceptedResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    ingestion_run_id: int = Field(alias="ingestionRunId")
    document_id: int = Field(alias="documentId")
    status: Literal[AdminIngestionStatus.PROCESSING]


class AdminIngestionProcessingResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    ingestion_run_id: int = Field(alias="ingestionRunId")
    document_id: int = Field(alias="documentId")
    status: Literal[AdminIngestionStatus.PROCESSING]
    started_at: datetime = Field(alias="startedAt")


class AdminIngestionSuccessResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    ingestion_run_id: int = Field(alias="ingestionRunId")
    document_id: int = Field(alias="documentId")
    status: Literal[AdminIngestionStatus.SUCCESS]
    document_version_id: int = Field(alias="documentVersionId")
    version_no: int = Field(alias="versionNo", ge=1)
    changed: Literal[True]
    section_count: int = Field(alias="sectionCount", ge=1)
    chunk_count: int = Field(alias="chunkCount", ge=1)
    started_at: datetime = Field(alias="startedAt")
    finished_at: datetime = Field(alias="finishedAt")


class AdminIngestionFailedResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    ingestion_run_id: int = Field(alias="ingestionRunId")
    document_id: int = Field(alias="documentId")
    status: Literal[AdminIngestionStatus.FAILED]
    error: AdminError
    started_at: datetime = Field(alias="startedAt")
    finished_at: datetime = Field(alias="finishedAt")


AdminIngestionRunResponse = Annotated[
    Union[
        AdminIngestionProcessingResponse,
        AdminIngestionSuccessResponse,
        AdminIngestionFailedResponse,
    ],
    Field(discriminator="status"),
]


class IndexRunStageValue(str, Enum):
    BUILDING = "BUILDING"
    VALIDATING = "VALIDATING"
    APPLYING = "APPLYING"


class IndexOperationTypeValue(str, Enum):
    BUILD_AND_APPLY = "BUILD_AND_APPLY"
    BUILD = "BUILD"
    APPLY = "APPLY"


class AdminIndexRunAcceptedResponse(BaseModel):
    """검색 반영 시작과 적용 재시도 접수 결과."""

    model_config = HTTP_DTO_CONFIG

    index_run_id: int = Field(alias="indexRunId")
    index_version_id: int = Field(alias="indexVersionId")
    group_id: int = Field(alias="groupId")
    operation_type: IndexOperationTypeValue = Field(alias="operationType")
    trigger_type: str = Field(alias="triggerType")
    status: AdminIngestionStatus
    stage: IndexRunStageValue
    retry_of_index_run_id: Optional[int] = Field(
        default=None,
        alias="retryOfIndexRunId",
    )


class IndexRunErrorCode(str, Enum):
    """실행 이력에 남는 검색 반영 실패 원인."""

    VALIDATION_FAILED = "VALIDATION_FAILED"
    CORPUS_RELOAD_FAILED = "CORPUS_RELOAD_FAILED"
    CORPUS_OUT_OF_SYNC = "CORPUS_OUT_OF_SYNC"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AdminIndexRunError(BaseModel):
    """검색 반영 실행의 실패 원인. HTTP 오류 코드와 별개다."""

    model_config = HTTP_DTO_CONFIG

    code: IndexRunErrorCode
    message: str


class AdminIndexVersionSummary(BaseModel):
    """실행이 다룬 검색 버전 요약."""

    model_config = HTTP_DTO_CONFIG

    index_version_id: int = Field(alias="indexVersionId")
    version_no: Optional[int] = Field(alias="versionNo")
    status: str
    activated_at: Optional[datetime] = Field(default=None, alias="activatedAt")


class AdminIndexRunProcessingResponse(BaseModel):
    """진행 중인 검색 반영 실행."""

    model_config = HTTP_DTO_CONFIG

    index_run_id: int = Field(alias="indexRunId")
    group_id: int = Field(alias="groupId")
    index_version_id: int = Field(alias="indexVersionId")
    operation_type: IndexOperationTypeValue = Field(alias="operationType")
    trigger_type: str = Field(alias="triggerType")
    status: AdminIngestionStatus
    stage: IndexRunStageValue
    started_at: datetime = Field(alias="startedAt")


class AdminIndexRunSuccessResponse(BaseModel):
    """반영이 끝난 검색 반영 실행."""

    model_config = HTTP_DTO_CONFIG

    index_run_id: int = Field(alias="indexRunId")
    group_id: int = Field(alias="groupId")
    index_version_id: int = Field(alias="indexVersionId")
    operation_type: IndexOperationTypeValue = Field(alias="operationType")
    trigger_type: str = Field(alias="triggerType")
    status: AdminIngestionStatus
    stage: IndexRunStageValue
    index_version: AdminIndexVersionSummary = Field(alias="indexVersion")
    previous_index_version: Optional[AdminIndexVersionSummary] = Field(
        alias="previousIndexVersion"
    )
    document_count: int = Field(alias="documentCount", ge=0)
    chunk_count: int = Field(alias="chunkCount", ge=0)
    started_at: datetime = Field(alias="startedAt")
    finished_at: datetime = Field(alias="finishedAt")


class AdminIndexRunFailedResponse(BaseModel):
    """실패로 마감된 검색 반영 실행."""

    model_config = HTTP_DTO_CONFIG

    index_run_id: int = Field(alias="indexRunId")
    group_id: int = Field(alias="groupId")
    index_version_id: int = Field(alias="indexVersionId")
    operation_type: IndexOperationTypeValue = Field(alias="operationType")
    trigger_type: str = Field(alias="triggerType")
    status: AdminIngestionStatus
    stage: IndexRunStageValue
    error: AdminIndexRunError
    index_version: AdminIndexVersionSummary = Field(alias="indexVersion")
    retryable: bool
    started_at: datetime = Field(alias="startedAt")
    finished_at: datetime = Field(alias="finishedAt")


AdminIndexRunResponse = Union[
    AdminIndexRunProcessingResponse,
    AdminIndexRunSuccessResponse,
    AdminIndexRunFailedResponse,
]
