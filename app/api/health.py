import logging
from typing import Union

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.build_info import BUILD_VERSION
from app.database.session import get_db_session
from app.profiles.readiness import check_published_profile_versions


logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", summary="애플리케이션 상태 확인")
async def health_check() -> dict[str, str]:
    """애플리케이션 liveness와 빌드 커밋을 반환한다."""

    return {"status": "ok", "buildCommit": BUILD_VERSION}


@router.get(
    "/health/db",
    response_model=None,
    summary="데이터베이스 연결 상태 확인",
)
async def database_health_check(
    session: AsyncSession = Depends(get_db_session),
) -> Union[dict[str, str], JSONResponse]:
    """간단한 쿼리로 데이터베이스 연결 상태를 확인한다."""

    try:
        await session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "database": "unavailable"},
        )

    return {"status": "ok", "database": "connected"}


@router.get(
    "/health/profiles",
    response_model=None,
    summary="PUBLISHED 프로필 버전 서빙 가능 여부 확인",
)
async def profile_versions_health_check(
    session: AsyncSession = Depends(get_db_session),
) -> Union[dict[str, object], JSONResponse]:
    """등록된 서비스마다 PUBLISHED 프로필 버전을 실행 중인 코드로 서빙할 수 있는지 확인한다.

    배포 후 점검(scripts/deploy.sh)이 부른다. 공개 주소로도 열려 있으므로 서비스별 사유는 로그에 남기고,
    정상일 때 서빙 프로필 식별자를 응답한다.
    """

    try:
        results = await check_published_profile_versions(session)
    except SQLAlchemyError:
        logger.exception("프로필 버전 점검 중 데이터베이스 오류가 발생했습니다.")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "database": "unavailable"},
        )

    unservable = [result for result in results if not result.servable]
    for result in unservable:
        logger.error(
            "PUBLISHED 프로필 버전을 서빙할 수 없습니다: service=%s, revision_id=%s, problems=%s",
            result.profile_key,
            result.profile_version_id,
            " | ".join(result.problems),
        )
    if unservable:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "profiles": "unservable"},
        )
    return {
        "status": "ok",
        "profiles": "servable",
        "publishedProfiles": [
            {
                "profileKey": result.profile_key,
                "profileRevisionId": result.profile_version_id,
                "version": result.version,
            }
            for result in results
        ],
    }
