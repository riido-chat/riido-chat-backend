import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.chat.profile import (
    ConversationProfileMismatchError,
    resolve_chat_profile_revision,
)
from app.chat.service import ChatService
from app.database.models import (
    ConversationChannel,
    ChatProfileRevisionStatus,
)


class ChatProfileRoutingTest(unittest.IsolatedAsyncioTestCase):
    def _revision(self, *, status: ChatProfileRevisionStatus) -> SimpleNamespace:
        return SimpleNamespace(
            id=11,
            profile_id=7,
            document_group_id=23,
            status=status,
            generation_model_name="gpt-5.6-terra",
            generation_prompt_version="v24",
            query_rewrite_model_name="gpt-5.4-mini",
            query_rewrite_prompt_version="v7",
        )

    async def test_pinned_retired_revision_remains_valid_for_production(self) -> None:
        conversation_id = uuid.uuid4()
        revision = self._revision(status=ChatProfileRevisionStatus.RETIRED)
        conversation = SimpleNamespace(
            id=conversation_id,
            chat_profile_revision_id=revision.id,
            channel=ConversationChannel.PUBLIC,
        )
        profile = SimpleNamespace(
            id=revision.profile_id,
            profile_key="HELP_CHATBOT",
        )
        session = SimpleNamespace(
            get=AsyncMock(side_effect=[conversation, revision, profile])
        )

        resolved = await resolve_chat_profile_revision(
            session,
            status=ChatProfileRevisionStatus.PUBLISHED,
            conversation_id=conversation_id,
        )

        self.assertIs(revision, resolved)

    async def test_testing_endpoint_rejects_production_pinned_conversation(self) -> None:
        conversation_id = uuid.uuid4()
        revision = self._revision(status=ChatProfileRevisionStatus.PUBLISHED)
        conversation = SimpleNamespace(
            id=conversation_id,
            chat_profile_revision_id=revision.id,
            channel=ConversationChannel.PUBLIC,
        )
        profile = SimpleNamespace(
            id=revision.profile_id,
            profile_key="HELP_CHATBOT",
        )
        session = SimpleNamespace(
            get=AsyncMock(side_effect=[conversation, revision, profile])
        )

        with self.assertRaises(ConversationProfileMismatchError):
            await resolve_chat_profile_revision(
                session,
                status=ChatProfileRevisionStatus.TESTING,
                conversation_id=conversation_id,
            )

    async def test_retired_former_testing_revision_stays_on_testing_channel(self) -> None:
        conversation_id = uuid.uuid4()
        revision = self._revision(status=ChatProfileRevisionStatus.RETIRED)
        conversation = SimpleNamespace(
            id=conversation_id,
            chat_profile_revision_id=revision.id,
            channel=ConversationChannel.INTERNAL_TEST,
        )
        profile = SimpleNamespace(id=revision.profile_id, profile_key="HELP_CHATBOT")
        session = SimpleNamespace(
            get=AsyncMock(side_effect=[conversation, revision, profile])
        )

        resolved = await resolve_chat_profile_revision(
            session,
            status=ChatProfileRevisionStatus.TESTING,
            channel=ConversationChannel.INTERNAL_TEST,
            conversation_id=conversation_id,
        )
        self.assertIs(revision, resolved)

        session.get.reset_mock()
        session.get.side_effect = [conversation, revision, profile]
        with self.assertRaises(ConversationProfileMismatchError):
            await resolve_chat_profile_revision(
                session,
                status=ChatProfileRevisionStatus.PUBLISHED,
                channel=ConversationChannel.PUBLIC,
                conversation_id=conversation_id,
            )

    async def test_promoted_testing_revision_stays_pinned_to_testing_channel(self) -> None:
        conversation_id = uuid.uuid4()
        # The revision was TESTING when the conversation was created and is
        # now PUBLISHED.  Its persisted channel remains the source of truth.
        revision = self._revision(status=ChatProfileRevisionStatus.PUBLISHED)
        conversation = SimpleNamespace(
            id=conversation_id,
            chat_profile_revision_id=revision.id,
            channel=ConversationChannel.INTERNAL_TEST,
        )
        profile = SimpleNamespace(id=revision.profile_id, profile_key="HELP_CHATBOT")
        session = SimpleNamespace(
            get=AsyncMock(side_effect=[conversation, revision, profile])
        )

        resolved = await resolve_chat_profile_revision(
            session,
            status=ChatProfileRevisionStatus.TESTING,
            channel=ConversationChannel.INTERNAL_TEST,
            conversation_id=conversation_id,
        )
        self.assertIs(revision, resolved)

        session.get.reset_mock()
        session.get.side_effect = [conversation, revision, profile]
        with self.assertRaises(ConversationProfileMismatchError):
            await resolve_chat_profile_revision(
                session,
                status=ChatProfileRevisionStatus.PUBLISHED,
                channel=ConversationChannel.PUBLIC,
                conversation_id=conversation_id,
            )

    async def test_retired_former_production_revision_stays_on_public_channel(self) -> None:
        conversation_id = uuid.uuid4()
        revision = self._revision(status=ChatProfileRevisionStatus.RETIRED)
        conversation = SimpleNamespace(
            id=conversation_id,
            chat_profile_revision_id=revision.id,
            channel=ConversationChannel.PUBLIC,
        )
        profile = SimpleNamespace(id=revision.profile_id, profile_key="HELP_CHATBOT")
        session = SimpleNamespace(
            get=AsyncMock(side_effect=[conversation, revision, profile])
        )

        resolved = await resolve_chat_profile_revision(
            session,
            status=ChatProfileRevisionStatus.PUBLISHED,
            channel=ConversationChannel.PUBLIC,
            conversation_id=conversation_id,
        )
        self.assertIs(revision, resolved)

        session.get.reset_mock()
        session.get.side_effect = [conversation, revision, profile]
        with self.assertRaises(ConversationProfileMismatchError):
            await resolve_chat_profile_revision(
                session,
                status=ChatProfileRevisionStatus.TESTING,
                channel=ConversationChannel.INTERNAL_TEST,
                conversation_id=conversation_id,
            )

    async def test_missing_or_null_channel_is_rejected(self) -> None:
        revision = self._revision(status=ChatProfileRevisionStatus.PUBLISHED)
        profile = SimpleNamespace(id=revision.profile_id, profile_key="HELP_CHATBOT")
        for conversation in (
            SimpleNamespace(id=uuid.uuid4(), chat_profile_revision_id=revision.id),
            SimpleNamespace(
                id=uuid.uuid4(),
                chat_profile_revision_id=revision.id,
                channel=None,
            ),
        ):
            with self.subTest(conversation=conversation):
                session = SimpleNamespace(
                    get=AsyncMock(side_effect=[conversation, revision, profile])
                )
                with self.assertRaises(ConversationProfileMismatchError):
                    await resolve_chat_profile_revision(
                        session,
                        status=ChatProfileRevisionStatus.PUBLISHED,
                        channel=ConversationChannel.PUBLIC,
                        conversation_id=conversation.id,
                    )

    async def test_runtime_model_mismatch_fails_before_conversation_creation(
        self,
    ) -> None:
        revision = self._revision(status=ChatProfileRevisionStatus.PUBLISHED)
        log_store = SimpleNamespace(create_conversation=AsyncMock())
        session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
        service = ChatService(
            retriever=None,
            generation_service=SimpleNamespace(
                model_name="wrong-model",
                prompt_version="v24",
            ),
            query_rewrite_service=SimpleNamespace(
                model_name="gpt-5.4-mini",
                prompt_version="v7",
            ),
            log_store=log_store,
            session=session,
            index_version_id=None,
            profile_status=ChatProfileRevisionStatus.PUBLISHED,
            retriever_factory=lambda _: (object(), 91),
        )
        with patch(
            "app.chat.service.resolve_chat_profile_revision",
            new=AsyncMock(return_value=revision),
        ):
            with self.assertRaisesRegex(RuntimeError, "모델 설정"):
                await service._start_turn("질문", None)

        log_store.create_conversation.assert_not_awaited()

    async def test_new_turn_pins_revision_and_uses_one_retriever_snapshot(self) -> None:
        conversation_id = uuid.uuid4()
        revision = SimpleNamespace(
            id=11,
            document_group_id=23,
            status=ChatProfileRevisionStatus.PUBLISHED,
            generation_model_name="gpt-5.6-terra",
            generation_prompt_version="v24",
            query_rewrite_model_name="gpt-5.4-mini",
            query_rewrite_prompt_version="v7",
        )
        conversation = SimpleNamespace(id=conversation_id)
        run = SimpleNamespace(id=uuid.uuid4(), turn_no=1)
        log_store = SimpleNamespace(
            create_conversation=AsyncMock(return_value=conversation),
            start_rag_run=AsyncMock(return_value=run),
        )
        session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
        generation = SimpleNamespace(model_name="gpt-5.6-terra", prompt_version="v24")
        rewrite = SimpleNamespace(model_name="gpt-5.4-mini", prompt_version="v7")
        retriever = object()

        service = ChatService(
            retriever=None,
            generation_service=generation,
            query_rewrite_service=rewrite,
            log_store=log_store,
            session=session,
            index_version_id=None,
            profile_status=ChatProfileRevisionStatus.PUBLISHED,
            retriever_factory=lambda group_id: (retriever, 91),
        )
        with patch(
            "app.chat.service.resolve_chat_profile_revision",
            new=AsyncMock(return_value=revision),
        ):
            turn = await service._start_turn("질문", None)

        log_store.create_conversation.assert_awaited_once_with(
            chat_profile_revision_id=11,
            channel=ConversationChannel.PUBLIC,
        )
        log_store.start_rag_run.assert_awaited_once_with(
            conversation_id,
            user_query="질문",
            index_version_id=91,
        )
        self.assertIs(retriever, turn.retriever)
        self.assertEqual(11, turn.profile_revision_id)
        self.assertEqual(23, turn.document_group_id)
        self.assertEqual(91, turn.index_version_id)


if __name__ == "__main__":
    unittest.main()
