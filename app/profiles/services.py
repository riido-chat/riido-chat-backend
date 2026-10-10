"""주소마다 서빙하는 서비스.

서비스 하나는 chat_profiles 행 하나이고 전용 문서 그룹 하나를 쓴다. 요청은 서비스를 고르지
못하고 주소가 정한다. 서비스 키를 쓰는 실행 경로는 이 module 의 값만 쓴다.
"""

from dataclasses import dataclass

from app.database.models import ConversationChannel


@dataclass(frozen=True)
class ChatServiceRoute:
    """주소 하나가 서빙하는 서비스.

    channel 은 conversations.channel 칸의 옛 표기다. 4단계에서 칸과 함께 지운다.
    """

    profile_key: str
    channel: ConversationChannel


# POST /api/chat (SSE 포함)
HELP_CHATBOT = ChatServiceRoute(
    profile_key="HELP_CHATBOT",
    channel=ConversationChannel.PUBLIC,
)
# POST /api/internal/test-chat
HELP_CHATBOT_TEST = ChatServiceRoute(
    profile_key="HELP_CHATBOT_TEST",
    channel=ConversationChannel.INTERNAL_TEST,
)
