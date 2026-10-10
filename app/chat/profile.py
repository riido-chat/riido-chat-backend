"""Resolution of the chat profile revision used by a request."""

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


class ChatProfileUnavailableError(LookupError):
    """The requested profile has no uniquely selectable revision."""


class ChatProfileConfigurationError(RuntimeError):
    """The selected revision cannot be served by the configured model clients."""


class ConversationProfileMismatchError(LookupError):
    """A conversation belongs to another endpoint's service."""


async def resolve_chat_profile_revision(
    session: AsyncSession,
    *,
    service: ChatServiceRoute,
    conversation_id: Optional[uuid.UUID] = None,
) -> ChatProfileRevision:
    """Resolve a pinned conversation revision or the service's PUBLISHED revision.

    The endpoint supplies only its service, so a caller can never select a raw
    profile, group, or revision id from HTTP input.
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
        revision = await session.get(
            ChatProfileRevision,
            conversation.chat_profile_revision_id,
        )
        if revision is None or revision.profile_id != profile.id:
            raise ConversationProfileMismatchError(
                f"대화의 프로필 버전이 서비스와 다릅니다: {conversation_id}"
            )
        # A conversation keeps using the exact immutable revision it pinned,
        # regardless of later lifecycle transitions.  revision.status only
        # selects new conversations.
        return revision

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


def validate_runtime_model_configuration(
    revision: ChatProfileRevision,
    *,
    generation_service: object,
    query_rewrite_service: object,
) -> None:
    """Fail closed when observed model clients differ from profile intent."""

    generation_model = getattr(generation_service, "model_name", None)
    generation_prompt = getattr(generation_service, "prompt_version", None)
    rewrite_model = getattr(query_rewrite_service, "model_name", None)
    rewrite_prompt = getattr(query_rewrite_service, "prompt_version", None)
    expected = (
        revision.generation_model_name,
        revision.generation_prompt_version,
        revision.query_rewrite_model_name,
        revision.query_rewrite_prompt_version,
    )
    observed = (generation_model, generation_prompt, rewrite_model, rewrite_prompt)
    if observed != expected:
        raise ChatProfileConfigurationError(
            "선택한 chat profile revision과 실제 모델 설정이 일치하지 않습니다."
        )
