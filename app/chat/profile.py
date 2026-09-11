"""Resolution of the chat profile revision used by a request."""

from typing import Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    ChatProfile,
    ChatProfileRevision,
    ChatProfileRevisionStatus,
    ConversationChannel,
    Conversation,
)


HELP_CHATBOT_PROFILE_KEY = "HELP_CHATBOT"


class ChatProfileUnavailableError(LookupError):
    """The requested profile has no uniquely selectable revision."""


class ChatProfileConfigurationError(RuntimeError):
    """The selected revision cannot be served by the configured model clients."""


class ConversationProfileMismatchError(LookupError):
    """A conversation is pinned to another endpoint's profile revision."""


async def resolve_chat_profile_revision(
    session: AsyncSession,
    *,
    status: ChatProfileRevisionStatus,
    channel: Optional[ConversationChannel] = None,
    conversation_id: Optional[uuid.UUID] = None,
) -> ChatProfileRevision:
    """Resolve a pinned conversation revision or the current endpoint revision.

    The endpoint supplies only the lifecycle status (PUBLISHED for production or
    TESTING for the internal endpoint).  A caller can therefore never select a
    raw profile, group, or revision id from HTTP input.
    """

    if conversation_id is not None:
        requested_channel = channel or (
            ConversationChannel.INTERNAL_TEST
            if status == ChatProfileRevisionStatus.TESTING
            else ConversationChannel.PUBLIC
        )
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None or conversation.chat_profile_revision_id is None:
            raise ConversationProfileMismatchError(
                f"프로필이 연결되지 않은 대화입니다: {conversation_id}"
            )
        revision = await session.get(
            ChatProfileRevision,
            conversation.chat_profile_revision_id,
        )
        if revision is None:
            raise ConversationProfileMismatchError(
                f"프로필 판을 찾을 수 없는 대화입니다: {conversation_id}"
            )
        profile = await session.get(ChatProfile, revision.profile_id)
        if profile is None or profile.profile_key != HELP_CHATBOT_PROFILE_KEY:
            raise ConversationProfileMismatchError(
                f"대화가 HELP_CHATBOT 프로필에 연결되지 않았습니다: {conversation_id}"
            )
        # The migration makes channel non-null.  Treat an object that predates
        # that schema (or a malformed row) as invalid instead of silently
        # assigning it to the public endpoint.
        try:
            persisted_channel = conversation.channel
        except AttributeError:
            persisted_channel = None
        if persisted_channel is None or persisted_channel != requested_channel:
            raise ConversationProfileMismatchError(
                f"대화의 endpoint channel이 요청과 다릅니다: {conversation_id}"
            )
        # A conversation keeps using the exact immutable revision it pinned,
        # regardless of later lifecycle transitions.  Channel is the stable
        # endpoint boundary; revision.status only selects new conversations.
        return revision

    statement = (
        select(ChatProfileRevision)
        .join(ChatProfile, ChatProfile.id == ChatProfileRevision.profile_id)
        .where(
            ChatProfile.profile_key == HELP_CHATBOT_PROFILE_KEY,
            ChatProfileRevision.status == status,
        )
    )
    revisions = list((await session.scalars(statement)).all())
    if len(revisions) != 1:
        raise ChatProfileUnavailableError(
            f"HELP_CHATBOT의 {status.value} 프로필 판이 정확히 하나가 아닙니다."
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
