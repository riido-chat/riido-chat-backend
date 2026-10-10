"""프로필 버전 관리 요청과 응답. 지시문 본문은 노출하지 않는다."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.database.models import ChatProfileRevisionStatus, ProfileRevisionComponentStage
from app.profiles.lifecycle import RevisionDetail
from app.profiles.stage_settings import ComponentSpec, STAGE_ORDER


class ComponentDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, from_attributes=True)

    stage: ProfileRevisionComponentStage
    model_name: str = Field(alias="modelName", min_length=1, max_length=150)
    prompt_key: str = Field(alias="promptKey", min_length=1, max_length=100)
    prompt_version: str = Field(alias="promptVersion", min_length=1, max_length=50)
    prompt_sha256: str | None = Field(alias="promptSha256", pattern=r"^[0-9a-f]{64}$")
    params: dict[str, Any] | None = None

    def to_spec(self) -> ComponentSpec:
        return ComponentSpec(**self.model_dump())


class CreateComponentRequest(ComponentDTO):
    prompt_sha256: str = Field(alias="promptSha256", pattern=r"^[0-9a-f]{64}$")


class CreateRevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    components: list[CreateComponentRequest] | None = Field(default=None, min_length=7, max_length=7)
    semantic_cache_enabled: bool | None = Field(default=None, alias="semanticCacheEnabled")
    exact_cache_enabled: bool | None = Field(default=None, alias="exactCacheEnabled")


class RevisionDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: int
    profile_id: int = Field(alias="profileId")
    version: int
    document_group_id: int = Field(alias="documentGroupId")
    status: ChatProfileRevisionStatus
    semantic_cache_enabled: bool = Field(alias="semanticCacheEnabled")
    exact_cache_enabled: bool = Field(alias="exactCacheEnabled")
    created_at: datetime = Field(alias="createdAt")


class RevisionDetailDTO(RevisionDTO):
    components: list[ComponentDTO]

    @classmethod
    def from_detail(cls, detail: RevisionDetail) -> "RevisionDetailDTO":
        positions = {stage: index for index, stage in enumerate(STAGE_ORDER)}
        return cls(
            **RevisionDTO.model_validate(detail.revision).model_dump(),
            components=[ComponentDTO.model_validate(row) for row in sorted(
                detail.components, key=lambda row: positions[row.stage]
            )],
        )


class RevisionListResponse(BaseModel):
    revisions: list[RevisionDTO]
