"""Admin 문서 업로드와 수집 상태 조회 HTTP DTO."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from fastapi import UploadFile
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator

from app.api.chat_schema import HTTP_DTO_CONFIG


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
    source_url: AnyHttpUrl = Field(alias="sourceUrl", max_length=1_000)
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
