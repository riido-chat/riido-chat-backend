"""Admin 문서 업로드와 수집 상태 조회 HTTP DTO."""

from datetime import datetime
from enum import Enum
from typing import Annotated, List, Literal, Optional, Union
from uuid import UUID

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.chat.schema import HTTP_DTO_CONFIG


ADMIN_UPLOAD_DTO_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    populate_by_name=True,
    arbitrary_types_allowed=True,
)


class AdminDocumentUploadRequest(BaseModel):
    """신규 문서 업로드 multipart 요청.

    문서는 (문서 그룹, document_key)로 식별하므로 sourceUrl은 받지 않는다.
    콘솔 문서의 canonical_uri는 서버가 만들고 API로 노출하지 않는다.
    """

    model_config = ADMIN_UPLOAD_DTO_CONFIG

    title: str = Field(min_length=1, max_length=300)
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


class AdminDocumentRevisionRequest(BaseModel):
    """수정본 업로드 multipart 요청. 대상은 경로가 정하므로 파일만 받는다."""

    model_config = ADMIN_UPLOAD_DTO_CONFIG

    file: UploadFile


class AdminIngestionStatus(str, Enum):
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class AdminErrorCode(str, Enum):
    INVALID_FILE = "INVALID_FILE"
    # 요청 본문이나 필드 형식이 잘못된 경우. 422 를 다른 오류와 같은 형식으로 돌려준다.
    INVALID_REQUEST = "INVALID_REQUEST"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    DOCUMENT_NOT_REVISABLE = "DOCUMENT_NOT_REVISABLE"
    JOB_IN_PROGRESS = "JOB_IN_PROGRESS"
    REINDEX_NOT_REQUIRED = "REINDEX_NOT_REQUIRED"
    NO_READY_DOCUMENTS = "NO_READY_DOCUMENTS"
    RETRY_NOT_ALLOWED = "RETRY_NOT_ALLOWED"
    SOURCE_LIST_FAILED = "SOURCE_LIST_FAILED"
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
    # 접수 전 거절에만 붙는다. FE가 3-4 원인 문구 변형을 고를 때 쓴다.
    stage: Optional[str] = None


class IngestionStageValue(str, Enum):
    """업로드 실행의 진행 단계. FE는 이 값으로 3-4 원인 문구를 고른다."""

    RECEIVING = "RECEIVING"
    VALIDATING = "VALIDATING"
    NORMALIZING = "NORMALIZING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    PERSISTING = "PERSISTING"


class AdminIngestionAcceptedResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    ingestion_run_id: int = Field(alias="ingestionRunId")
    document_id: int = Field(alias="documentId")
    status: Literal[AdminIngestionStatus.PROCESSING]
    stage: IngestionStageValue


class IngestionResultCodeValue(str, Enum):
    """업로드 결과. 같은 내용 재업로드와 중복은 오류가 아니라 결과다."""

    CREATED = "CREATED"
    UPDATED = "UPDATED"
    NO_CHANGE = "NO_CHANGE"
    DUPLICATE_CONTENT = "DUPLICATE_CONTENT"


class IngestionErrorCode(str, Enum):
    """업로드 실행 이력에 남는 실패 원인."""

    INVALID_FILE = "INVALID_FILE"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AdminIngestionError(BaseModel):
    """업로드 실행의 실패 원인. HTTP 오류 코드와 별개다."""

    model_config = HTTP_DTO_CONFIG

    code: IngestionErrorCode
    message: str


class AdminChunkStats(BaseModel):
    """이전 판 대비 청크 변화."""

    model_config = HTTP_DTO_CONFIG

    added: int = Field(ge=0)
    changed: int = Field(ge=0)
    deleted: int = Field(ge=0)
    reused: int = Field(ge=0)


class AdminDuplicateDocument(BaseModel):
    """같은 본문을 이미 가진 문서."""

    model_config = HTTP_DTO_CONFIG

    document_id: int = Field(alias="documentId")
    title: str


class AdminIngestionProcessingResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    ingestion_run_id: int = Field(alias="ingestionRunId")
    document_id: int = Field(alias="documentId")
    status: Literal[AdminIngestionStatus.PROCESSING]
    stage: IngestionStageValue
    started_at: datetime = Field(alias="startedAt")


class AdminIngestionSuccessResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    ingestion_run_id: int = Field(alias="ingestionRunId")
    document_id: int = Field(alias="documentId")
    status: Literal[AdminIngestionStatus.SUCCESS]
    result_code: IngestionResultCodeValue = Field(alias="resultCode")
    stage: IngestionStageValue
    document_version_id: Optional[int] = Field(alias="documentVersionId")
    version_no: Optional[int] = Field(alias="versionNo", ge=1)
    section_count: Optional[int] = Field(alias="sectionCount", ge=0)
    chunk_count: Optional[int] = Field(alias="chunkCount", ge=0)
    chunk_stats: Optional[AdminChunkStats] = Field(alias="chunkStats")
    duplicate_of: Optional[AdminDuplicateDocument] = Field(alias="duplicateOf")
    started_at: datetime = Field(alias="startedAt")
    finished_at: datetime = Field(alias="finishedAt")


class AdminIngestionFailedResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    ingestion_run_id: int = Field(alias="ingestionRunId")
    document_id: int = Field(alias="documentId")
    status: Literal[AdminIngestionStatus.FAILED]
    stage: IngestionStageValue
    error: AdminIngestionError
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


class AdminGitBookSyncRequest(BaseModel):
    """GitBook 수집 요청. 루트 URL 하나를 받는다."""

    model_config = HTTP_DTO_CONFIG

    source_url: str = Field(alias="sourceUrl", min_length=1, max_length=1_000)

    @field_validator("source_url")
    @classmethod
    def require_https(cls, source_url: str) -> str:
        value = source_url.strip().rstrip("/")
        if not value.startswith("https://"):
            raise ValueError("GitBook 루트 URL은 https 여야 합니다.")
        return value


class RecollectStageValue(str, Enum):
    """재탐색 배치의 단계."""

    LISTING = "LISTING"
    PROCESSING = "PROCESSING"


class AdminRecollectAcceptedResponse(BaseModel):
    """재탐색 접수 결과."""

    model_config = HTTP_DTO_CONFIG

    batch_id: UUID = Field(alias="batchId")
    group_id: int = Field(alias="groupId")
    group_source_id: int = Field(alias="groupSourceId")
    root_url: str = Field(alias="rootUrl")
    status: Literal[AdminIngestionStatus.PROCESSING]
    stage: RecollectStageValue
    # 빈 목록은 list_pages 가 502 SOURCE_LIST_FAILED 로 먼저 막으므로 0 은 오지 않는다.
    page_count: int = Field(alias="pageCount", ge=1)


class AdminRecollectProgress(BaseModel):
    """페이지 처리 진행."""

    model_config = HTTP_DTO_CONFIG

    total: int = Field(ge=0)
    processed: int = Field(ge=0)


class AdminRecollectCounts(BaseModel):
    """배치 결과 집계."""

    model_config = HTTP_DTO_CONFIG

    total: int = Field(ge=0)
    created: int = Field(ge=0)
    updated: int = Field(ge=0)
    no_change: int = Field(alias="noChange", ge=0)
    removed: int = Field(ge=0)
    failed: int = Field(ge=0)


class AdminRecollectFailure(BaseModel):
    """실패한 페이지 한 건."""

    model_config = HTTP_DTO_CONFIG

    document_key: str = Field(alias="documentKey")
    title: str
    ingestion_run_id: int = Field(alias="ingestionRunId")
    stage: IngestionStageValue
    error_code: Optional[IngestionErrorCode] = Field(alias="errorCode")


class AdminRecollectProcessingResponse(BaseModel):
    """진행 중인 재탐색 배치."""

    model_config = HTTP_DTO_CONFIG

    batch_id: UUID = Field(alias="batchId")
    group_id: int = Field(alias="groupId")
    group_source_id: Optional[int] = Field(alias="groupSourceId")
    root_url: Optional[str] = Field(alias="rootUrl")
    status: Literal[AdminIngestionStatus.PROCESSING]
    stage: RecollectStageValue
    progress: AdminRecollectProgress
    started_at: datetime = Field(alias="startedAt")


class AdminRecollectSuccessResponse(BaseModel):
    """끝난 재탐색 배치. 일부 페이지가 실패해도 배치는 성공이다."""

    model_config = HTTP_DTO_CONFIG

    batch_id: UUID = Field(alias="batchId")
    group_id: int = Field(alias="groupId")
    group_source_id: Optional[int] = Field(alias="groupSourceId")
    root_url: Optional[str] = Field(alias="rootUrl")
    status: Literal[AdminIngestionStatus.SUCCESS]
    stage: RecollectStageValue
    counts: AdminRecollectCounts
    failures: List[AdminRecollectFailure]
    started_at: datetime = Field(alias="startedAt")
    finished_at: datetime = Field(alias="finishedAt")


AdminRecollectBatchResponse = Union[
    AdminRecollectProcessingResponse,
    AdminRecollectSuccessResponse,
]


class SearchStatusValue(str, Enum):
    """그룹 단위 검색 반영 상태. 저장하지 않고 계산한다."""

    UP_TO_DATE = "UP_TO_DATE"
    REINDEX_REQUIRED = "REINDEX_REQUIRED"
    IN_PROGRESS = "IN_PROGRESS"
    NO_DOCUMENTS = "NO_DOCUMENTS"


class ChangeTypeValue(str, Enum):
    """반영 대기 문서의 변경 종류."""

    NEW = "NEW"
    UPDATED = "UPDATED"
    REMOVED = "REMOVED"


class SourceTypeValue(str, Enum):
    GITBOOK = "GITBOOK"
    UPLOAD = "UPLOAD"


class JobTypeValue(str, Enum):
    INGESTION = "INGESTION"
    RECOLLECT = "RECOLLECT"
    INDEX = "INDEX"


class AdminDocumentGroupSummary(BaseModel):
    """문서 그룹 목록 한 줄."""

    model_config = HTTP_DTO_CONFIG

    group_id: int = Field(alias="groupId")
    group_key: str = Field(alias="groupKey")
    name: str
    consumer_key: str = Field(alias="consumerKey")
    document_count: int = Field(alias="documentCount", ge=0)
    active_index_version_no: Optional[int] = Field(alias="activeIndexVersionNo")
    search_status: SearchStatusValue = Field(alias="searchStatus")


class AdminDocumentGroupListResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    groups: List[AdminDocumentGroupSummary]


class AdminGroupSourceItem(BaseModel):
    """그룹이 문서를 끌어오는 수집 원천 하나."""

    model_config = HTTP_DTO_CONFIG

    group_source_id: int = Field(alias="groupSourceId")
    provider: str
    root_url: str = Field(alias="rootUrl")
    enabled: bool
    document_count: int = Field(alias="documentCount", ge=0)


class AdminGroupInfo(BaseModel):
    model_config = HTTP_DTO_CONFIG

    group_id: int = Field(alias="groupId")
    group_key: str = Field(alias="groupKey")
    name: str
    consumer_key: str = Field(alias="consumerKey")


class AdminActiveIndexVersion(BaseModel):
    model_config = HTTP_DTO_CONFIG

    index_version_id: int = Field(alias="indexVersionId")
    version_no: Optional[int] = Field(alias="versionNo")
    activated_at: Optional[datetime] = Field(alias="activatedAt")


class AdminPendingDocument(BaseModel):
    model_config = HTTP_DTO_CONFIG

    document_id: int = Field(alias="documentId")
    title: str
    change_type: ChangeTypeValue = Field(alias="changeType")


class AdminGroupSummary(BaseModel):
    """상세 화면의 요약 카드."""

    model_config = HTTP_DTO_CONFIG

    active_index_version: Optional[AdminActiveIndexVersion] = Field(
        alias="activeIndexVersion"
    )
    pending_count: int = Field(alias="pendingCount", ge=0)
    pending_documents: List[AdminPendingDocument] = Field(
        alias="pendingDocuments"
    )
    search_status: SearchStatusValue = Field(alias="searchStatus")


class AdminGroupDocument(BaseModel):
    """상세 표의 한 행."""

    model_config = HTTP_DTO_CONFIG

    document_id: int = Field(alias="documentId")
    document_key: str = Field(alias="documentKey")
    title: str
    source_type: SourceTypeValue = Field(alias="sourceType")
    # 어느 수집 원천에서 왔는지. 콘솔 업로드 문서는 null 이다.
    group_source_id: Optional[int] = Field(alias="groupSourceId")
    document_version_no: int = Field(alias="documentVersionNo", ge=1)
    applied_version_no: Optional[int] = Field(alias="appliedVersionNo")
    processing_status: str = Field(alias="processingStatus")


class AdminRunningJob(BaseModel):
    """진행 중인 작업. 있으면 콘솔의 실행 버튼이 모두 비활성이다."""

    model_config = HTTP_DTO_CONFIG

    job_type: JobTypeValue = Field(alias="jobType")
    stage: str
    ingestion_run_id: Optional[int] = Field(default=None, alias="ingestionRunId")
    document_id: Optional[int] = Field(default=None, alias="documentId")
    index_run_id: Optional[int] = Field(default=None, alias="indexRunId")
    batch_id: Optional[UUID] = Field(default=None, alias="batchId")
    group_source_id: Optional[int] = Field(default=None, alias="groupSourceId")
    root_url: Optional[str] = Field(default=None, alias="rootUrl")


class AdminLatestIndexRun(BaseModel):
    """재진입 시 4-2 또는 4-4 모달 복원 근거."""

    model_config = HTTP_DTO_CONFIG

    index_run_id: int = Field(alias="indexRunId")
    index_version_id: int = Field(alias="indexVersionId")
    operation_type: IndexOperationTypeValue = Field(alias="operationType")
    status: AdminIngestionStatus
    stage: IndexRunStageValue
    error_code: Optional[IndexRunErrorCode] = Field(alias="errorCode")
    started_at: datetime = Field(alias="startedAt")
    finished_at: Optional[datetime] = Field(alias="finishedAt")


class AdminDocumentGroupDetailResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    group: AdminGroupInfo
    sources: List[AdminGroupSourceItem]
    summary: AdminGroupSummary
    documents: List[AdminGroupDocument]
    running_job: Optional[AdminRunningJob] = Field(alias="runningJob")
    latest_index_run: Optional[AdminLatestIndexRun] = Field(
        alias="latestIndexRun"
    )
