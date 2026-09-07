"""Admin 문서 업로드와 검색 반영, GitBook 수집 HTTP DTO."""

from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional
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
    file: UploadFile

    @field_validator("title", mode="before")
    @classmethod
    def strip_title(cls, title: object) -> object:
        if isinstance(title, str):
            return title.strip()
        return title


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
    # 같은 이름의 콘솔 문서에 READY 판이 있다. 새 판은 수정본 업로드로 만든다.
    DOCUMENT_ALREADY_EXISTS = "DOCUMENT_ALREADY_EXISTS"
    # 그룹 안 다른 콘솔 문서와 본문이 같다. 판을 만들지 않는다.
    DUPLICATE_CONTENT = "DUPLICATE_CONTENT"
    DOCUMENT_NOT_REVISABLE = "DOCUMENT_NOT_REVISABLE"
    JOB_IN_PROGRESS = "JOB_IN_PROGRESS"
    REINDEX_NOT_REQUIRED = "REINDEX_NOT_REQUIRED"
    NO_READY_DOCUMENTS = "NO_READY_DOCUMENTS"
    SOURCE_LIST_FAILED = "SOURCE_LIST_FAILED"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AdminError(BaseModel):
    model_config = HTTP_DTO_CONFIG

    code: AdminErrorCode
    message: str


class AdminErrorResponse(BaseModel):
    """오류 본문은 code 와 message 둘이다.

    화면 문구는 message 를 그대로 보인다. code 는 어느 화면에 보일지만 정한다.
    실행의 단계는 ingestion_runs 에 기록하고 API 로 노출하지 않는다.
    """

    model_config = HTTP_DTO_CONFIG

    code: AdminErrorCode
    message: str


class IngestionStageValue(str, Enum):
    """업로드 실행의 진행 단계. 기록용이고 응답으로 내보내지 않는다."""

    RECEIVING = "RECEIVING"
    VALIDATING = "VALIDATING"
    NORMALIZING = "NORMALIZING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    PERSISTING = "PERSISTING"


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


class AdminChunkStats(BaseModel):
    """이전 판 대비 청크 변화."""

    model_config = HTTP_DTO_CONFIG

    added: int = Field(ge=0)
    changed: int = Field(ge=0)
    deleted: int = Field(ge=0)
    reused: int = Field(ge=0)


class AdminUploadResultResponse(BaseModel):
    """업로드 처리를 끝내고 돌려주는 결과.

    실행이 동기라 접수 응답이 없다. 요청의 응답이 곧 결과다.
    """

    model_config = HTTP_DTO_CONFIG

    ingestion_run_id: int = Field(alias="ingestionRunId")
    document_id: int = Field(alias="documentId")
    result_code: IngestionResultCodeValue = Field(alias="resultCode")
    document_version_id: Optional[int] = Field(alias="documentVersionId")
    version_no: Optional[int] = Field(alias="versionNo", ge=1)
    section_count: Optional[int] = Field(alias="sectionCount", ge=0)
    chunk_count: Optional[int] = Field(alias="chunkCount", ge=0)
    chunk_stats: Optional[AdminChunkStats] = Field(alias="chunkStats")


class IndexRunStageValue(str, Enum):
    BUILDING = "BUILDING"
    VALIDATING = "VALIDATING"
    APPLYING = "APPLYING"


class IndexOperationTypeValue(str, Enum):
    BUILD_AND_APPLY = "BUILD_AND_APPLY"
    BUILD = "BUILD"
    APPLY = "APPLY"


class AdminIndexRunAcceptedResponse(BaseModel):
    """검색 반영 시작 접수 결과."""

    model_config = HTTP_DTO_CONFIG

    index_run_id: int = Field(alias="indexRunId")
    index_version_id: int = Field(alias="indexVersionId")
    group_id: int = Field(alias="groupId")
    operation_type: IndexOperationTypeValue = Field(alias="operationType")
    trigger_type: str = Field(alias="triggerType")
    status: AdminIngestionStatus
    stage: IndexRunStageValue


class IndexRunErrorCode(str, Enum):
    """실행 이력에 남는 검색 반영 실패 원인."""

    VALIDATION_FAILED = "VALIDATION_FAILED"
    CORPUS_RELOAD_FAILED = "CORPUS_RELOAD_FAILED"
    CORPUS_OUT_OF_SYNC = "CORPUS_OUT_OF_SYNC"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AdminIndexVersionSummary(BaseModel):
    """실행이 다룬 검색 버전 요약."""

    model_config = HTTP_DTO_CONFIG

    index_version_id: int = Field(alias="indexVersionId")
    version_no: Optional[int] = Field(alias="versionNo")
    status: str
    activated_at: Optional[datetime] = Field(default=None, alias="activatedAt")


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


class SearchStatusValue(str, Enum):
    """그룹 단위 검색 반영 상태. 저장하지 않고 계산한다."""

    UP_TO_DATE = "UP_TO_DATE"
    REINDEX_REQUIRED = "REINDEX_REQUIRED"
    IN_PROGRESS = "IN_PROGRESS"
    NO_DOCUMENTS = "NO_DOCUMENTS"
    FAILED = "FAILED"


class AppliedStatusValue(str, Enum):
    """문서 표의 반영 여부 뱃지가 쓰는 값."""

    APPLIED = "APPLIED"
    UNAPPLIED = "UNAPPLIED"


class SourceTypeValue(str, Enum):
    GITBOOK = "GITBOOK"
    UPLOAD = "UPLOAD"


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


class AdminGroupSummary(BaseModel):
    """상세 화면의 요약 카드."""

    model_config = HTTP_DTO_CONFIG

    active_index_version: Optional[AdminActiveIndexVersion] = Field(
        alias="activeIndexVersion"
    )
    # 반영 대기 건수. documents[] 중 appliedStatus 가 UNAPPLIED 인 문서 수다.
    pending_count: int = Field(alias="pendingCount", ge=0)
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
    applied_status: AppliedStatusValue = Field(alias="appliedStatus")


class AdminDocumentGroupDetailResponse(BaseModel):
    model_config = HTTP_DTO_CONFIG

    group: AdminGroupInfo
    sources: List[AdminGroupSourceItem]
    summary: AdminGroupSummary
    documents: List[AdminGroupDocument]
    # 그룹에 PROCESSING 인 실행이 있는지. 업로드와 검색 반영과 GitBook 수집이
    # 그룹 잠금을 공유하므로 종류를 구분하지 않는다.
    job_in_progress: bool = Field(alias="jobInProgress")
