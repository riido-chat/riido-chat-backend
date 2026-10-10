import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.chat.profile import (
    ChatProfileUnavailableError,
    ConversationProfileMismatchError,
    resolve_chat_profile_revision,
)
from app.chat.service import ChatService
from app.database.models import (
    ConversationChannel,
    ChatProfileRevisionStatus,
)
from app.profiles.services import HELP_CHATBOT, HELP_CHATBOT_TEST
from app.question_grouping.exact_question import exact_question_hash


class ChatProfileRoutingTest(unittest.IsolatedAsyncioTestCase):
    def _revision(
        self,
        *,
        status: ChatProfileRevisionStatus,
        profile_id: int = 7,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=11,
            profile_id=profile_id,
            document_group_id=23,
            status=status,
            generation_model_name="gpt-5.6-terra",
            generation_prompt_version="v24",
            query_rewrite_model_name="gpt-5.4-mini",
            query_rewrite_prompt_version="v7",
        )

    @staticmethod
    def _conversation(revision: SimpleNamespace, **overrides) -> SimpleNamespace:
        values = dict(
            id=uuid.uuid4(),
            chat_profile_revision_id=revision.id,
            profile_id=revision.profile_id,
            channel=ConversationChannel.PUBLIC,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    async def _resolve(self, service, conversation, profile, revision):
        session = SimpleNamespace(
            get=AsyncMock(side_effect=[conversation, profile, revision])
        )
        return await resolve_chat_profile_revision(
            session,
            service=service,
            conversation_id=conversation.id,
        )

    async def test_pinned_retired_revision_remains_valid_for_its_service(self) -> None:
        for service in (HELP_CHATBOT, HELP_CHATBOT_TEST):
            with self.subTest(service=service.profile_key):
                revision = self._revision(status=ChatProfileRevisionStatus.RETIRED)
                profile = SimpleNamespace(id=7, profile_key=service.profile_key)
                conversation = self._conversation(revision, channel=service.channel)

                resolved = await self._resolve(service, conversation, profile, revision)

                self.assertIs(revision, resolved)

    async def test_conversation_continues_only_on_its_own_service_address(self) -> None:
        # The persisted channel is ignored: conversations.profile_id decides.
        for owner, other in (
            (HELP_CHATBOT, HELP_CHATBOT_TEST),
            (HELP_CHATBOT_TEST, HELP_CHATBOT),
        ):
            for channel in ConversationChannel:
                with self.subTest(owner=owner.profile_key, channel=channel):
                    revision = self._revision(
                        status=ChatProfileRevisionStatus.PUBLISHED
                    )
                    profile = SimpleNamespace(id=7, profile_key=owner.profile_key)
                    conversation = self._conversation(revision, channel=channel)

                    with self.assertRaises(ConversationProfileMismatchError):
                        await self._resolve(other, conversation, profile, revision)
                    self.assertIs(
                        revision,
                        await self._resolve(owner, conversation, profile, revision),
                    )

    async def test_conversation_without_service_is_rejected(self) -> None:
        revision = self._revision(status=ChatProfileRevisionStatus.PUBLISHED)
        for conversation in (
            SimpleNamespace(id=uuid.uuid4(), chat_profile_revision_id=revision.id),
            self._conversation(revision, profile_id=None),
        ):
            with self.subTest(conversation=conversation):
                session = SimpleNamespace(get=AsyncMock(side_effect=[conversation]))
                with self.assertRaises(ConversationProfileMismatchError):
                    await resolve_chat_profile_revision(
                        session,
                        service=HELP_CHATBOT,
                        conversation_id=conversation.id,
                    )

    async def test_revision_of_another_profile_is_rejected(self) -> None:
        revision = self._revision(
            status=ChatProfileRevisionStatus.PUBLISHED,
            profile_id=8,
        )
        profile = SimpleNamespace(id=7, profile_key=HELP_CHATBOT.profile_key)
        conversation = self._conversation(revision, profile_id=7)

        with self.assertRaises(ConversationProfileMismatchError):
            await self._resolve(HELP_CHATBOT, conversation, profile, revision)

    async def test_new_conversation_uses_the_services_published_revision(self) -> None:
        revision = self._revision(status=ChatProfileRevisionStatus.PUBLISHED)
        session = SimpleNamespace(
            scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: [revision]))
        )

        resolved = await resolve_chat_profile_revision(
            session,
            service=HELP_CHATBOT_TEST,
        )

        self.assertIs(revision, resolved)
        statement = str(
            session.scalars.await_args.args[0].compile(
                compile_kwargs={"literal_binds": True}
            )
        )
        self.assertIn("'HELP_CHATBOT_TEST'", statement)
        self.assertIn("'PUBLISHED'", statement)
        self.assertNotIn("'TESTING'", statement)

        session.scalars = AsyncMock(return_value=SimpleNamespace(all=lambda: []))
        with self.assertRaises(ChatProfileUnavailableError):
            await resolve_chat_profile_revision(session, service=HELP_CHATBOT_TEST)

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
            service=HELP_CHATBOT,
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
        for service in (HELP_CHATBOT, HELP_CHATBOT_TEST):
            with self.subTest(service=service.profile_key):
                await self._assert_new_turn_pins_revision(service)

    async def _assert_new_turn_pins_revision(self, service) -> None:
        conversation_id = uuid.uuid4()
        revision = SimpleNamespace(
            id=11,
            profile_id=7,
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
        requested_groups = []

        def retriever_factory(group_id):
            requested_groups.append(group_id)
            return retriever, 91

        chat_service = ChatService(
            retriever=None,
            generation_service=generation,
            query_rewrite_service=rewrite,
            log_store=log_store,
            session=session,
            index_version_id=None,
            service=service,
            retriever_factory=retriever_factory,
        )
        resolve = AsyncMock(return_value=revision)
        with patch("app.chat.service.resolve_chat_profile_revision", new=resolve):
            turn = await chat_service._start_turn("질문", None)

        self.assertIs(service, resolve.await_args.kwargs["service"])
        log_store.create_conversation.assert_awaited_once_with(
            chat_profile_revision_id=11,
            profile_id=7,
            channel=service.channel,
        )
        log_store.start_rag_run.assert_awaited_once_with(
            conversation_id,
            user_query="질문",
            index_version_id=91,
            query_hash=exact_question_hash("질문"),
            profile_revision_id=11,
        )
        self.assertEqual([23], requested_groups)
        self.assertIs(retriever, turn.retriever)
        self.assertEqual(11, turn.profile_revision_id)
        self.assertEqual(23, turn.document_group_id)
        self.assertEqual(91, turn.index_version_id)


if __name__ == "__main__":
    unittest.main()
