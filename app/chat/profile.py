"""Resolution of the chat profile revision used by a request."""

import logging
from typing import Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ChatProfile,
    ChatProfileRevision,
    ChatProfileRevisionStatus,
    Conversation,
)
from app.profiles.services import ChatServiceRoute
from app.profiles.stage_settings import (
    ProfileStageSettings,
    ProfileStageSettingsError,
    load_profile_stage_settings,
)


logger = logging.getLogger(__name__)


class ChatProfileUnavailableError(LookupError):
    """The requested profile has no uniquely selectable revision."""


class ChatProfileConfigurationError(RuntimeError):
    """The selected revision cannot be served by the running code."""


class ConversationProfileMismatchError(LookupError):
    """A conversation belongs to another endpoint's service."""


async def resolve_chat_profile_revision(
    session: AsyncSession,
    *,
    service: ChatServiceRoute,
    conversation_id: Optional[uuid.UUID] = None,
) -> ChatProfileRevision:
    """이번 턴에 쓸 프로필 버전. 턴마다 서비스의 현재 PUBLISHED 프로필 버전을 고른다.

    The endpoint supplies only its service, so a caller can never select a raw
    profile, group, or revision id from HTTP input.

    이어 가는 대화도 시작한 프로필 버전이 아니라 지금 PUBLISHED 프로필 버전으로 답한다. 게시와
    되돌리기는 진행 중인 대화에도 다음 턴부터 적용된다. conversations.chat_profile_revision_id 는
    대화를 시작한 프로필 버전의 기록이고, 턴이 쓴 프로필 버전은 rag_runs.profile_revision_id 에
    남는다. 대화는 자기 서비스(conversations.profile_id)의 주소로만 이어진다.
    """

    if conversation_id is not None:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None or conversation.chat_profile_revision_id is None:
            raise ConversationProfileMismatchError(
                f"프로필이 연결되지 않은 대화입니다: {conversation_id}"
            )
        # 대화는 자기 서비스의 주소로만 이어진다.  서비스 칸이 없는 객체(옛 스키마나
        # 잘못된 행)는 어느 주소에도 붙이지 않는다.
        profile_id = getattr(conversation, "profile_id", None)
        profile = (
            None if profile_id is None else await session.get(ChatProfile, profile_id)
        )
        if profile is None or profile.profile_key != service.profile_key:
            raise ConversationProfileMismatchError(
                f"대화가 {service.profile_key} 서비스에 속하지 않습니다: {conversation_id}"
            )
        # 시작한 프로필 버전은 기록일 뿐 이번 턴에 쓰지 않는다. 같은 서비스의 것인지만 본다.
        started_revision = await session.get(
            ChatProfileRevision,
            conversation.chat_profile_revision_id,
        )
        if started_revision is None or started_revision.profile_id != profile.id:
            raise ConversationProfileMismatchError(
                f"대화의 프로필 버전이 서비스와 다릅니다: {conversation_id}"
            )

    statement = (
        select(ChatProfileRevision)
        .join(ChatProfile, ChatProfile.id == ChatProfileRevision.profile_id)
        .where(
            ChatProfile.profile_key == service.profile_key,
            ChatProfileRevision.status == ChatProfileRevisionStatus.PUBLISHED,
        )
    )
    revisions = list((await session.scalars(statement)).all())
    if len(revisions) != 1:
        raise ChatProfileUnavailableError(
            f"{service.profile_key}의 PUBLISHED 프로필 버전이 정확히 하나가 아닙니다."
        )
    return revisions[0]


async def load_turn_stage_settings(
    session: AsyncSession,
    revision: ChatProfileRevision,
) -> ProfileStageSettings:
    """선택한 프로필 버전의 단계별 설정. 실행 중인 코드로 서빙할 수 없으면 fail closed 한다.

    구성 행이 없거나, 프롬프트 버전이나 묶음 버전이 등록부에 없거나, 지문이 다르면
    ChatProfileConfigurationError 로 올린다. 턴 기록을 시작하기 전에 확인한다.
    """

    try:
        return await load_profile_stage_settings(session, revision)
    except ProfileStageSettingsError as error:
        logger.error(
            "프로필 버전을 실행 중인 코드로 서빙할 수 없어 답하지 않습니다: "
            "revision_id=%s, problems=%s",
            error.revision_id,
            " | ".join(error.problems),
        )
        raise ChatProfileConfigurationError(str(error)) from error
