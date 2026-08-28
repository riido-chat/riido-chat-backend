"""SSE 전달 계층의 asyncio 계약과 코어 훅 무영향을 검증한다.

TestClient는 스트림을 버퍼링해 disconnect·drain을 관측할 수 없으므로
producer와 lifespan 헬퍼를 직접 호출해 확인한다.
"""

import asyncio
import unittest
import uuid
from contextlib import asynccontextmanager
from itertools import count
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.chat_schema import (
    ChatAnswer,
    ChatCitation,
    ChatCompletedResponse,
    ChatResponseStatus,
)
from app.api.chat_stream import (
    ErrorEvent,
    ResultEvent,
    RunEvent,
    StageEvent,
    TurnStartFailed,
    _Sentinel,
    produce_turn,
    register_pipeline_task,
)
from app.database.models import ConversationStatus
from app.main import _drain_pipeline_tasks
from app.rag.chat_service import ChatService
from app.rag.generation_service import GenerationService
from app.rag.query_rewrite import QueryRewriteService
from app.rag.log_store import RagLogStore
from app.rag.model_trace import ModelCallTrace
from app.rag.progress import ProgressStage
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import HybridSearchCall
from generation.models import (
    Citation,
    FinalAnswerStatus,
    FinalGenerationResult,
)


CONVERSATION_ID = uuid.UUID("6b401388-b1ca-410a-9430-dd9beee85460")
RAG_RUN_ID = uuid.UUID("d49dc6fb-25f1-4782-a1db-659fe1c55892")
INDEX_VERSION_ID = 7


def completed_response() -> ChatCompletedResponse:
    return ChatCompletedResponse(
        status=ChatResponseStatus.COMPLETED,
        conversation_id=CONVERSATION_ID,
        rag_run_id=RAG_RUN_ID,
        answer=ChatAnswer(answer_markdown="답변입니다. [1]"),
        citations=[
            ChatCitation(
                citation_number=1,
                document_title="문서",
                section_path=[],
                source_url="https://docs.riido.io/1",
            )
        ],
    )


async def _drain(queue: "asyncio.Queue"):
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


class ProduceTurnTest(unittest.IsolatedAsyncioTestCase):
    """producer의 sentinel 보장과 이벤트 순서를 검증한다."""

    def setUp(self) -> None:
        self.queue: "asyncio.Queue" = asyncio.Queue()
        self.service = AsyncMock(spec=ChatService)

        @asynccontextmanager
        async def scope(*_args, **_kwargs):
            yield self.service

        patcher = patch("app.api.chat_stream._chat_service_scope", scope)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _run(self) -> None:
        await produce_turn(
            self.queue,
            question="질문",
            conversation_id=None,
            corpus_state=object(),
            embedder=object(),
            generation_service=object(),
            query_rewrite_service=object(),
        )

    def _pipeline(self, *, stages=(), result=None, error=None, start_turn=True):
        async def answer(_question, _conversation_id=None, *, on_turn_started=None,
                         on_progress_stage=None):
            if start_turn:
                await on_turn_started(CONVERSATION_ID, RAG_RUN_ID)
            for stage in stages:
                await on_progress_stage(stage)
            if error is not None:
                raise error
            return result

        self.service.answer_question.side_effect = answer

    # ---------------- sentinel 3경로 ----------------

    async def test_sentinel_on_normal_completion(self) -> None:
        self._pipeline(stages=(ProgressStage.RETRIEVING,), result=completed_response())

        await self._run()

        items = await _drain(self.queue)
        self.assertIsInstance(items[-1], _Sentinel)
        self.assertIsInstance(items[-2], ResultEvent)

    async def test_sentinel_on_pipeline_exception(self) -> None:
        self._pipeline(
            stages=(ProgressStage.RETRIEVING,), error=RuntimeError("파손")
        )

        await self._run()

        items = await _drain(self.queue)
        self.assertIsInstance(items[-1], _Sentinel)
        self.assertIsInstance(items[-2], ErrorEvent)

    async def test_sentinel_on_turn_start_failure(self) -> None:
        self._pipeline(start_turn=False, error=RuntimeError("턴 생성 실패"))

        await self._run()

        items = await _drain(self.queue)
        self.assertIsInstance(items[-1], _Sentinel)
        self.assertIsInstance(items[-2], TurnStartFailed)
        self.assertIsNotNone(items[-2].error)

    # ---------------- 완주와 순서 ----------------

    async def test_producer_completes_without_consumer(self) -> None:
        """소비자가 없어도(= 클라이언트 단절) 터미널까지 완주한다."""

        self._pipeline(
            stages=(
                ProgressStage.RETRIEVING,
                ProgressStage.GENERATING,
                ProgressStage.VALIDATING,
            ),
            result=completed_response(),
        )

        task = asyncio.create_task(self._run())
        await task

        items = await _drain(self.queue)
        self.assertTrue(task.done())
        self.assertIsInstance(items[-2], ResultEvent)
        self.assertIsInstance(items[-1], _Sentinel)

    async def test_queue_preserves_event_order(self) -> None:
        self._pipeline(
            stages=(
                ProgressStage.RETRIEVING,
                ProgressStage.GENERATING,
                ProgressStage.VALIDATING,
            ),
            result=completed_response(),
        )

        await self._run()

        items = await _drain(self.queue)
        self.assertIsInstance(items[0], RunEvent)
        self.assertEqual(
            [
                ProgressStage.RETRIEVING,
                ProgressStage.GENERATING,
                ProgressStage.VALIDATING,
            ],
            [item.stage for item in items[1:4] if isinstance(item, StageEvent)],
        )


class DrainTest(unittest.IsolatedAsyncioTestCase):
    """shutdown drain이 실행 중 task를 기다리고 engine보다 먼저 끝나는지 확인한다."""

    async def test_drain_waits_for_pending_task(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(pipeline_tasks=set()))
        finished = []

        async def slow() -> None:
            await asyncio.sleep(0.05)
            finished.append("done")

        register_pipeline_task(app, asyncio.create_task(slow()))
        self.assertEqual([], finished)

        await _drain_pipeline_tasks(app)

        self.assertEqual(["done"], finished)
        self.assertEqual(set(), app.state.pipeline_tasks)

    async def test_drain_runs_before_dispose_engine(self) -> None:
        from app import main

        order = []

        async def fake_drain(app) -> None:
            order.append("drain")

        async def fake_dispose() -> None:
            order.append("dispose")

        async def fake_load(_corpus_state) -> None:
            return None

        with patch.object(main, "_drain_pipeline_tasks", fake_drain), \
                patch.object(main, "dispose_engine", fake_dispose), \
                patch.object(main, "_load_corpus_if_available", fake_load), \
                patch.object(main, "CorpusState", lambda _dir: object()), \
                patch.object(main, "OpenAIEmbedder", lambda: object()), \
                patch.object(main, "OpenAIGenerator", lambda: object()), \
                patch.object(main, "GenerationService", lambda _g: object()), \
                patch.object(main, "QueryRewriteService", lambda: object()):
            app = FastAPI()
            async with main.lifespan(app):
                self.assertEqual(set(), app.state.pipeline_tasks)

        self.assertEqual(["drain", "dispose"], order)

    async def test_register_creates_registry_when_lifespan_is_replaced(self) -> None:
        """빈 lifespan으로 만든 앱에서도 등록이 실패하지 않는다."""

        app = SimpleNamespace(state=SimpleNamespace())

        async def noop() -> None:
            return None

        register_pipeline_task(app, asyncio.create_task(noop()))

        self.assertEqual(1, len(app.state.pipeline_tasks))
        await _drain_pipeline_tasks(app)
        self.assertEqual(set(), app.state.pipeline_tasks)


class CoreHookNeutralityTest(unittest.IsolatedAsyncioTestCase):
    """훅을 넣지 않으면 코어 동작이 기존과 같은지 확인한다."""

    def setUp(self) -> None:
        self.conversation_id = CONVERSATION_ID
        self.rag_run_id = RAG_RUN_ID

        self.retriever = AsyncMock(spec=HybridRetriever)
        self.generation_service = AsyncMock(spec=GenerationService)
        self.query_rewrite_service = AsyncMock(spec=QueryRewriteService)
        self.log_store = AsyncMock(spec=RagLogStore)
        self.session = AsyncMock(spec=AsyncSession)

        self.log_store.create_conversation.return_value = SimpleNamespace(
            id=self.conversation_id,
            status=ConversationStatus.ACTIVE,
        )
        self.log_store.start_rag_run.return_value = SimpleNamespace(
            id=self.rag_run_id,
            turn_no=1,
        )
        ids = count(1)
        self.log_store.start_model_call.side_effect = (
            lambda **_kwargs: SimpleNamespace(id=next(ids))
        )

        async def search(_question, *, before_model_call):
            await before_model_call("openai", "text-embedding-3-large", None)
            return HybridSearchCall(
                embedding_call=ModelCallTrace(
                    provider="openai",
                    model_name="text-embedding-3-large",
                    succeeded=True,
                    latency_ms=30,
                )
            )

        async def generate(_question, _results, *, before_model_call,
                           on_progress_stage=None):
            await before_model_call("openai", "gpt-5.4-mini", "v3")
            # 실제 GenerationService가 검증 직전에 발행하는 지점을 흉내낸다
            if on_progress_stage is not None:
                await on_progress_stage(ProgressStage.VALIDATING)
            return self._generation_result()

        self.retriever.search_with_trace.side_effect = search
        self.generation_service.generate_answer.side_effect = generate

        self.service = ChatService(
            retriever=self.retriever,
            generation_service=self.generation_service,
            query_rewrite_service=self.query_rewrite_service,
            log_store=self.log_store,
            session=self.session,
            index_version_id=INDEX_VERSION_ID,
        )

    @staticmethod
    def _generation_result() -> FinalGenerationResult:
        return FinalGenerationResult(
            status=FinalAnswerStatus.COMPLETED,
            answer_markdown="답변입니다. [1]",
            citations=(
                Citation(
                    citation_number=1,
                    document_title="문서 1",
                    section_path=("문서 1", "섹션 1"),
                    source_url="https://docs.riido.io/1",
                    chunk_id=1,
                    document_version_id=101,
                ),
            ),
            model_call=ModelCallTrace(
                provider="openai",
                model_name="gpt-5.4-mini",
                succeeded=True,
                latency_ms=900,
                prompt_version="v3",
            ),
        )

    async def test_hooks_are_not_invoked_when_not_supplied(self) -> None:
        on_turn_started = AsyncMock()
        on_progress_stage = AsyncMock()

        response = await self.service.answer_question("질문")

        self.assertEqual(ChatResponseStatus.COMPLETED, response.status)
        on_turn_started.assert_not_awaited()
        on_progress_stage.assert_not_awaited()
        # 훅 미주입 시 generate_answer 호출 인자가 기존과 동일해야 한다
        _, kwargs = self.generation_service.generate_answer.call_args
        self.assertEqual({"before_model_call"}, set(kwargs))

    async def test_response_identical_with_and_without_hooks(self) -> None:
        plain = await self.service.answer_question("질문")

        async def noop_turn(_conversation_id, _rag_run_id) -> None:
            return None

        async def noop_stage(_stage) -> None:
            return None

        hooked = await self.service.answer_question(
            "질문",
            on_turn_started=noop_turn,
            on_progress_stage=noop_stage,
        )

        self.assertEqual(
            plain.model_dump(mode="json", by_alias=True),
            hooked.model_dump(mode="json", by_alias=True),
        )

    async def test_hooks_invoked_in_expected_order(self) -> None:
        seen = []

        async def on_turn_started(conversation_id, rag_run_id) -> None:
            seen.append(("run", conversation_id, rag_run_id))

        async def on_progress_stage(stage) -> None:
            seen.append(stage)

        await self.service.answer_question(
            "질문",
            on_turn_started=on_turn_started,
            on_progress_stage=on_progress_stage,
        )

        self.assertEqual("run", seen[0][0])
        self.assertEqual(self.conversation_id, seen[0][1])
        self.assertEqual(self.rag_run_id, seen[0][2])
        self.assertEqual(
            [
                ProgressStage.RETRIEVING,
                ProgressStage.GENERATING,
                ProgressStage.VALIDATING,
            ],
            seen[1:],
        )


if __name__ == "__main__":
    unittest.main()
