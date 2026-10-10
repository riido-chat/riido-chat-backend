"""프로필 버전 관리의 얇은 HTTP 어댑터.

PROFILE_REVISION_ADMIN_ENABLED 는 인증을 대체하지 않는다. 인증을 적용하기 전에는
공개 환경에서 활성화하지 않는다. 기본값 false 일 때 라우터 자체를 등록하지 않는다.
"""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.profile_revisions.schema import (
    CreateRevisionRequest, RevisionDetailDTO, RevisionDTO, RevisionListResponse,
)
from app.database.session import get_db_session
from app.document.ingestion_service import AdminApiError
from app.profiles.lifecycle import ProfileRevisionError, ProfileRevisionService


router = APIRouter(prefix="/api/admin/profiles/{profile_key}/revisions", tags=["admin-profile-revisions"])


def get_profile_revision_service(session: AsyncSession = Depends(get_db_session)) -> ProfileRevisionService:
    return ProfileRevisionService(session)


async def _result(operation):
    try:
        return await operation
    except ProfileRevisionError as error:
        raise AdminApiError(error.code, error.message, error.status_code) from error


@router.get("", response_model=RevisionListResponse)
async def list_revisions(profile_key: str, service: ProfileRevisionService = Depends(get_profile_revision_service)):
    revisions = await _result(service.list(profile_key))
    return RevisionListResponse(revisions=[RevisionDTO.model_validate(row) for row in revisions])


@router.get("/{revision_id}", response_model=RevisionDetailDTO)
async def revision_detail(profile_key: str, revision_id: int, service: ProfileRevisionService = Depends(get_profile_revision_service)):
    return RevisionDetailDTO.from_detail(await _result(service.detail(profile_key, revision_id)))


@router.post("", response_model=RevisionDetailDTO, status_code=201)
async def create_revision(profile_key: str, request: CreateRevisionRequest, service: ProfileRevisionService = Depends(get_profile_revision_service)):
    detail = await _result(service.create(
        profile_key,
        components=None if request.components is None else [row.to_spec() for row in request.components],
        semantic_cache_enabled=request.semantic_cache_enabled,
        exact_cache_enabled=request.exact_cache_enabled,
    ))
    return RevisionDetailDTO.from_detail(detail)


@router.post("/{revision_id}/testing", response_model=RevisionDetailDTO)
async def testing_revision(profile_key: str, revision_id: int, service: ProfileRevisionService = Depends(get_profile_revision_service)):
    return RevisionDetailDTO.from_detail(await _result(service.testing(profile_key, revision_id)))


@router.post("/{revision_id}/publish", response_model=RevisionDetailDTO)
async def publish_revision(profile_key: str, revision_id: int, service: ProfileRevisionService = Depends(get_profile_revision_service)):
    return RevisionDetailDTO.from_detail(await _result(service.publish(profile_key, revision_id)))


@router.post("/{revision_id}/rollback", response_model=RevisionDetailDTO)
async def rollback_revision(profile_key: str, revision_id: int, service: ProfileRevisionService = Depends(get_profile_revision_service)):
    return RevisionDetailDTO.from_detail(await _result(service.rollback(profile_key, revision_id)))


@router.delete("/{revision_id}", status_code=204)
async def delete_revision(profile_key: str, revision_id: int, service: ProfileRevisionService = Depends(get_profile_revision_service)):
    await _result(service.delete(profile_key, revision_id))
    return Response(status_code=204)
