"""배포 후 점검: 등록된 서비스마다 PUBLISHED 프로필 버전을 실행 중인 코드로 서빙할 수 있는지.

서비스마다 PUBLISHED 프로필 버전이 정확히 하나이고, 그 구성 행이 모두 있으며, 프롬프트 버전과
묶음 버전이 등록부에 있고 지문이 같아야 한다(app/profiles/stage_settings.py 의 서빙 조건).
채팅 턴이 쓰는 검사와 같은 함수를 쓰므로, 여기서 통과한 프로필 버전은 턴에서도 서빙된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ChatProfile,
    ChatProfileRevision,
    ChatProfileRevisionStatus,
)
from app.profiles.services import REGISTERED_SERVICES, ChatServiceRoute
from app.profiles.stage_settings import component_problems, read_profile_components


@dataclass(frozen=True)
class ServiceReadiness:
    """서비스 하나의 점검 결과. problems 가 비어 있으면 서빙할 수 있다."""

    profile_key: str
    profile_version_id: Optional[int]
    problems: Tuple[str, ...]

    @property
    def servable(self) -> bool:
        return not self.problems


async def check_service(
    session: AsyncSession,
    service: ChatServiceRoute,
) -> ServiceReadiness:
    revisions = list(
        (
            await session.scalars(
                select(ChatProfileRevision)
                .join(ChatProfile, ChatProfile.id == ChatProfileRevision.profile_id)
                .where(
                    ChatProfile.profile_key == service.profile_key,
                    ChatProfileRevision.status == ChatProfileRevisionStatus.PUBLISHED,
                )
            )
        ).all()
    )
    if len(revisions) != 1:
        return ServiceReadiness(
            profile_key=service.profile_key,
            profile_version_id=None,
            problems=(
                f"PUBLISHED 프로필 버전이 정확히 하나가 아님({len(revisions)}개)",
            ),
        )
    revision = revisions[0]
    rows = await read_profile_components(session, revision.id)
    return ServiceReadiness(
        profile_key=service.profile_key,
        profile_version_id=revision.id,
        problems=component_problems(revision, rows),
    )


async def check_published_profile_versions(
    session: AsyncSession,
    services: Sequence[ChatServiceRoute] = REGISTERED_SERVICES,
) -> Tuple[ServiceReadiness, ...]:
    """등록된 서비스마다 점검한다. 읽기만 한다."""

    return tuple([await check_service(session, service) for service in services])
