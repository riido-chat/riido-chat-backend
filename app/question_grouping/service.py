"""턴 하나의 질문 판별과 정본 캐시 게이트를 묶는 오케스트레이터.

ChatService 가 턴 흐름 사이사이에서 부른다(계획 1절 5~11단계). 이 모듈은 판별 후보 읽기,
판별 호출, 게이트, 판별 행·캐시 시도 쓰기의 순서와 실패 규칙만 책임지고 턴 마감
(complete_rag_run 등)과 SSE 는 호출자가 한다.

공개 API 와 트랜잭션 소유:

| 메서드 | commit | 설명 |
| --- | --- | --- |
| open_online_run | 직접 commit | 열린 ONLINE 분류 실행 조회/생성. 턴 잠금 없음 |
| record_exact_question | 하지 않음 | 턴 원문이 같은 문서 그룹 과거 첫 턴 질문과 정확히 같으면 최신 CONNECT 분류로 판별 행과 캐시 시도. 일치 없으면 쓰지 않음 |
| prepare | 재임베딩 checkpoint 만 직접 commit | 질문 벡터, 카탈로그, 후보, payload. 읽기는 열린 채 둔다 |
| judge | 판별 checkpoint 를 직접 commit | 대기 중인 재임베딩 호출 마감 + 호출자 쓰기 + 판별 model_call 시작 |
| record_judgment_and_gate | 하지 않음 | 호출 마감, 질문 임베딩, 게이트, 판별 행, 캐시 시도 |
| record_preparation_failure | 하지 않음 | 후보를 만들지 못한 턴의 실패 행과 FAILED 시도 |
| record_serve_failure | rollback 후 쓰고 commit 하지 않음 | SERVED 쓰기 실패 뒤 REJECTED[CANONICAL_SERVE_FAILED] 로 다시 쓴다 |
| finalize_attribution | 하지 않음 | 턴 끝 CITATION 귀속. complete_rag_run 과 같은 트랜잭션, 그보다 먼저 |

실패 규칙:

- fail-open(판별 실패로 기록하고 턴은 생성으로 진행): 질문 재임베딩 API 오류, 카탈로그·후보
  데이터 오류(CatalogDataError, IndexScopeNotFoundError, 후보 조립 ValueError), 판별 API 오류·
  timeout·미완료 응답, 판별 출력 무효(R7). 판별 행은 UNCLASSIFIED + NO_DOCUMENT/NONE,
  캐시 시도는 FAILED 다.
- fail-closed(예외를 그대로 올린다. 호출자는 기존 _fail_quietly 로 턴을 ERROR 로 닫는다):
  checkpoint·기록 쓰기와 commit 실패, DB 연결 오류 같은 판별 외 DB 오류.
- R13: 검색(질문 임베딩) 실패로 턴이 ERROR 로 끝나면 이 서비스를 부르지 않는다.

질문 벡터(결정 1): 검색이 실제로 넣은 질의(search.retrieval_query)가 resolved_query 와 같으면
검색 벡터를 재사용한다. 다르면(검색어 보충) resolved_query 를 따로 임베딩해 question_embeddings 와
세부 문제 top5 에 쓴다. 문서 후보는 어느 경우든 검색 결과에서 고른다.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.log_store import CitationLog, RagLogStore
from app.core.model_trace import ModelCallTrace
from app.core.openai_error import is_transient_openai_error
from app.database.models import (
    EMBEDDING_DIMENSIONS,
    CacheAttemptOutcome,
    ClassificationDecision,
    ExecutionStatus,
    ModelCallPurpose,
)
from app.question_grouping.catalog_reader import (
    CatalogDataError,
    IndexScopeNotFoundError,
    QuestionCatalogReader,
)
from app.question_grouping.constants import (
    JUDGE_REASONING_EFFORT,
    JUDGMENT_INPUT_SCHEMA_VERSION,
    REJECT_CANONICAL_DATA_INVALID,
    REJECT_CANONICAL_SERVE_FAILED,
)
from app.question_grouping.decision import build_turn_judgment, failed_turn_judgment
from app.question_grouping.document_candidates import (
    attach_outlines,
    document_retrieval_settings,
    rank_documents_from_search,
)
from app.question_grouping.gate import (
    evaluate_cache_gate,
    gate_judgment_input,
    resolve_citations,
)
from app.question_grouping.models import (
    CitationResolution,
    GateInputs,
    GateResult,
    IndexScope,
    JudgeCall,
    JudgeFailure,
    JudgeFailureKind,
    JudgePresentation,
    PresentedSubproblem,
    SubproblemCatalog,
    TurnJudgment,
)
from app.question_grouping.outline_reader import DocumentOutlineReader
from app.question_grouping.payload import (
    build_judge_presentation,
    presentation_judgment_input,
    presentation_seed,
)
from app.question_grouping.store import QuestionGroupingStore, served_citation_logs
from app.question_grouping.attribution import subproblem_attribution
from app.question_grouping.subproblem_search import rank_subproblems
from app.retrieval.embedding import OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_PROVIDER
from app.retrieval.models import HybridSearchCall
from app.retrieval.vector_retriever import (
    QUERY_EMBEDDING_INITIAL_RETRY_DELAY_SECONDS,
    QUERY_EMBEDDING_MAX_ATTEMPTS,
    QUERY_EMBEDDING_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# judgment_input.failure.stage
FAILURE_STAGE_QUESTION_EMBEDDING = "QUESTION_EMBEDDING"
FAILURE_STAGE_CANDIDATES = "CANDIDATES"
FAILURE_STAGE_JUDGE = "JUDGE"

# 파싱하지 못한 판별 출력을 judgment_input.output 에 남길 때의 최대 길이.
MAX_UNPARSED_OUTPUT_CHARS = 4000

BeforeJudgeCheckpoint = Callable[[], Awaitable[None]]
"""판별 checkpoint commit 직전에 호출자가 같은 트랜잭션에 쓰기를 더하는 hook."""


def _elapsed_ms(started: float, now: float) -> int:
    return max(0, int((now - started) * 1000))


# ---------------------------------------------------------------------------
# 결과 모델
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupingTurn:
    """판별하는 턴과 그 턴이 속한 열린 ONLINE 분류 실행."""

    rag_run_id: uuid.UUID
    document_group_id: int
    index_version_id: int
    classification_run_id: int


@dataclass(frozen=True)
class QuestionVector:
    """판별과 question_embeddings 에 쓰는 resolved_query 벡터와 출처."""

    embedding: Optional[Tuple[float, ...]]
    reused_retrieval_embedding: bool
    retrieval_query: Optional[str]
    # 재임베딩했을 때만 채운다. 마감은 판별 checkpoint 또는 기록 트랜잭션이 한다.
    model_call_id: Optional[int] = None
    trace: Optional[ModelCallTrace] = None


@dataclass(frozen=True)
class PreparedJudgment:
    """판별 호출 직전까지의 준비 결과.

    failure 가 없으면 presentation 이 있고 judge 를 부를 수 있다. failure 가 있으면 판별을
    부르지 않고 record_preparation_failure 로 실패 행을 쓴다.
    """

    turn: GroupingTurn
    resolved_query: str
    seed: str
    model: str
    prompt_version: str
    vector: QuestionVector
    started: float
    scope: Optional[IndexScope] = None
    catalog: Optional[SubproblemCatalog] = None
    presentation: Optional[JudgePresentation] = None
    failure: Optional[JudgeFailure] = None
    failure_stage: Optional[str] = None
    exact_match_metadata: Optional[Dict[str, Any]] = None

    @property
    def ready(self) -> bool:
        return self.failure is None and self.presentation is not None


@dataclass(frozen=True)
class JudgedTurn:
    """판별 호출 한 건과 정규화한 턴 판별."""

    call: JudgeCall
    judgment: TurnJudgment
    model_call_id: Optional[int]
    # 판별 checkpoint 에서 재임베딩 model_call 을 이미 마감(commit)했는가.
    embedding_call_finished: bool = False


@dataclass(frozen=True)
class ExactQuestionResult:
    """정확 일치 기록과 SERVED 실패 복구에 필요한 내부 입력."""

    recorded: "RecordedJudgment"
    prepared: PreparedJudgment
    judged: JudgedTurn


@dataclass(frozen=True)
class RecordedJudgment:
    """기록한 판별 행과 캐시 시도. SERVED 면 호출자가 같은 트랜잭션에서 턴을 완료한다."""

    classification_id: int
    cache_attempt_id: int
    judgment: TurnJudgment
    gate: GateResult
    latency_ms: int
    citation_resolutions: Tuple[CitationResolution, ...] = ()
    canonical_answer_id: Optional[uuid.UUID] = None
    served_citations: Tuple[CitationLog, ...] = ()
    served_answer_markdown: Optional[str] = None

    @property
    def served(self) -> bool:
        return self.gate.outcome == CacheAttemptOutcome.SERVED


@dataclass(frozen=True)
class _GateOutcome:
    gate: GateResult
    resolutions: Tuple[CitationResolution, ...] = ()
    inputs: GateInputs = field(default_factory=GateInputs)
    data_error: Optional[str] = None


# ---------------------------------------------------------------------------
# judgment_input
# ---------------------------------------------------------------------------


def _usage_block(trace: Optional[ModelCallTrace]) -> Optional[Dict[str, Any]]:
    if trace is None:
        return None
    return {
        "input": trace.input_tokens,
        "output": trace.output_tokens,
        "cached": trace.cached_input_tokens,
        "reasoning": trace.reasoning_tokens,
        "latencyMs": trace.latency_ms,
        "retryCount": trace.retry_count,
    }


def _output_block(prepared: PreparedJudgment, judged: Optional[JudgedTurn]) -> Any:
    if judged is None:
        return None
    normalized = judged.judgment.normalized
    if normalized is not None and normalized.raw_output is not None:
        return dict(normalized.raw_output)
    text = judged.call.output_text
    if text is None or judged.call.failure is not None:
        return None
    return text[:MAX_UNPARSED_OUTPUT_CHARS]


def build_judgment_input(
    prepared: PreparedJudgment,
    judged: Optional[JudgedTurn],
    gate_block: Dict[str, Any],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """판별 행 judgment_input(계획 3절). 한 번 쓰고 고치지 않는다.

    UUID 는 문자열로 바꿔 JSON 으로 직렬화할 수 있게 한다. presentationSeed 는 rag_run_id 로
    만든 섞기 seed 라 같은 턴을 다시 판별하면 같은 제시 순서가 나온다(결정 10).
    """

    turn = prepared.turn
    vector = prepared.vector
    scope = prepared.scope
    catalog = prepared.catalog
    judgment = None if judged is None else judged.judgment

    failure: Optional[JudgeFailure] = prepared.failure
    stage = prepared.failure_stage
    retry_count: Optional[int] = None
    if failure is not None:
        if stage == FAILURE_STAGE_QUESTION_EMBEDDING and vector.trace is not None:
            retry_count = vector.trace.retry_count
    elif judgment is not None and judgment.failure is not None:
        failure = judgment.failure
        stage = FAILURE_STAGE_JUDGE
        retry_count = judged.call.trace.retry_count if judged is not None else None

    data: Dict[str, Any] = {
        "schemaVersion": JUDGMENT_INPUT_SCHEMA_VERSION,
        "resolvedQuery": prepared.resolved_query,
        "promptVersion": prepared.prompt_version,
        "model": prepared.model,
        "reasoningEffort": JUDGE_REASONING_EFFORT,
        "indexVersionId": turn.index_version_id,
        "documentGroupId": turn.document_group_id,
        "classificationRunId": turn.classification_run_id,
        "presentationSeed": prepared.seed,
        "embedding": {
            "embeddingConfigId": None if scope is None else scope.embedding_config_id,
            "subproblemTextVersion": (
                [] if catalog is None else list(catalog.embedding_text_versions)
            ),
            "reusedRetrievalEmbedding": vector.reused_retrieval_embedding,
            "retrievalQuery": vector.retrieval_query,
            "usage": _usage_block(vector.trace),
        },
        "catalog": (
            None
            if catalog is None
            else {
                "itemCount": len(catalog.items),
                "skippedMissingEmbedding": catalog.skipped_missing_embedding,
                "skippedEmbeddingConfigMismatch": (
                    catalog.skipped_embedding_config_mismatch
                ),
            }
        ),
        "subproblemCandidates": None,
        "documentCandidates": None,
        "output": _output_block(prepared, judged),
        "normalization": (
            None
            if judgment is None or judgment.normalized is None
            else judgment.normalized.normalization_judgment_input()
        ),
        "failure": (
            None
            if failure is None
            else {
                "stage": stage,
                "kind": failure.kind.value,
                "safeMessage": failure.safe_message,
                "retryCount": retry_count,
            }
        ),
        "usage": None if judged is None else _usage_block(judged.call.trace),
        "gate": gate_block,
    }
    if prepared.presentation is not None:
        data.update(
            presentation_judgment_input(
                prepared.presentation,
                document_retrieval=document_retrieval_settings(),
            )
        )
    if extra:
        data.update(extra)
    # jsonb 에 넣기 전에 직렬화 가능 여부(UUID, NaN 등)를 확인한다.
    json.dumps(data, ensure_ascii=False, allow_nan=False)
    return data


def _safe_embedding_error_message(error: BaseException) -> str:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return f"질문 임베딩 호출 실패: HTTP {status}"
    return f"질문 임베딩 호출 실패: {type(error).__name__}"


def _safe_candidate_error_message(error: BaseException) -> str:
    # 카탈로그 데이터 오류 문구는 key·id 만 담는다. 그 외 예외는 질문이 섞일 수 있어 이름만 쓴다.
    if isinstance(error, (CatalogDataError, IndexScopeNotFoundError)):
        return f"판별 후보 데이터 오류: {error}"
    return f"판별 후보를 만들지 못했습니다: {type(error).__name__}"


# ---------------------------------------------------------------------------
# 서비스
# ---------------------------------------------------------------------------


class QuestionGroupingService:
    """턴 하나의 판별, 게이트, 기록 순서를 조율한다.

    session, log_store 는 ChatService 와 같은 객체를 받는다. store, reader 는 같은 세션으로
    만든다(생략하면 기본 구현). judge_client 는 provider/model_name/prompt_version 속성과
    judge(payload, before_model_call=) 를, embedder 는 embed_many_with_usage 를 제공한다.
    """

    def __init__(
        self,
        session: AsyncSession,
        log_store: RagLogStore,
        judge_client: Any,
        embedder: Any,
        *,
        store: Optional[QuestionGroupingStore] = None,
        catalog_reader: Optional[QuestionCatalogReader] = None,
        outline_reader: Optional[DocumentOutlineReader] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._session = session
        self._log_store = log_store
        self._judge_client = judge_client
        self._embedder = embedder
        self._store = store or QuestionGroupingStore(session, log_store)
        self._catalog_reader = catalog_reader or QuestionCatalogReader(session)
        self._outline_reader = outline_reader or DocumentOutlineReader(session)
        self._clock = clock

    # ------------------------------------------------------------------
    # 1. 분류 실행
    # ------------------------------------------------------------------

    async def open_online_run(
        self,
        *,
        rag_run_id: uuid.UUID,
        document_group_id: int,
        index_version_id: int,
    ) -> GroupingTurn:
        """같은 설정의 열린 ONLINE 실행을 찾거나 열고 commit 한다.

        검색 임베딩 checkpoint(QUERY_EMBEDDING, rag_run_id + classification_run_id)보다 먼저
        부른다. 턴 잠금을 잡지 않는 공유 행이라 턴 쓰기가 섞이지 않은 시점에 부르고, 쓰기·commit
        실패는 그대로 올린다(fail-closed).
        """

        run_id = await self._store.get_or_open_online_run(
            document_group_id=document_group_id,
            index_version_id=index_version_id,
            model=self._judge_client.model_name,
            prompt_version=self._judge_client.prompt_version,
        )
        await self._session.commit()
        return GroupingTurn(
            rag_run_id=rag_run_id,
            document_group_id=document_group_id,
            index_version_id=index_version_id,
            classification_run_id=run_id,
        )

    async def record_exact_question(
        self,
        turn: GroupingTurn,
        question: str,
        *,
        semantic_cache_enabled: bool,
    ) -> Optional[ExactQuestionResult]:
        """턴 원문이 같은 문서 그룹의 과거 첫 턴 질문과 정확히 같으면 LLM·검색 없이 판별과 게이트를 기록한다.

        첫 턴과 후속 턴 모두에서 Query Rewrite 전에 사용자 원문으로 부른다. 매핑 원천은
        과거 첫 턴 로그뿐이다.
        일치한 로그 중 가장 최근에 확정된 현재 CONNECT 분류가 가리키는 세부 문제를 현재
        개정으로 연결한다(로그끼리 세부 문제가 달라도 최신 분류가 이긴다). 일치가 없으면
        아무 행도 쓰지 않고 None 을 돌려 호출자가 일반 판별로 진행한다. 게이트가 거절하면 기록만 반환하고
        호출자가 일반 흐름(후속 턴은 Query Rewrite 포함)의 검색·생성으로 이어간다.
        """

        match = await self._store.find_exact_question_log_match(
            turn.document_group_id,
            question,
            exclude_rag_run_id=turn.rag_run_id,
        )
        if match is None:
            return None
        scope = await self._catalog_reader.load_index_scope(turn.index_version_id)
        if scope.document_group_id != turn.document_group_id:
            return None
        presented = PresentedSubproblem(
            key=match.key,
            subproblem_id=match.subproblem_id,
            subproblem_version=match.current_version,
            problem_group_id=match.problem_group_id,
            document_source_id=match.document_source_id,
            document_key=match.document_key or "",
            canonical_answer_id=None,
            similarity=1.0,
            retrieval_rank=1,
            presented_order=1,
            inclusion_count=0,
            exclusion_count=0,
        )
        prepared = PreparedJudgment(
            turn=turn,
            resolved_query=question,
            seed=presentation_seed(turn.rag_run_id),
            model=self._judge_client.model_name,
            prompt_version=self._judge_client.prompt_version,
            vector=QuestionVector(
                embedding=None,
                reused_retrieval_embedding=False,
                retrieval_query=None,
            ),
            started=self._clock(),
            scope=scope,
            exact_match_metadata={
                "matchSource": "QUESTION_LOG_EXACT",
                "exactQuestionMatch": {
                    "normalizedQuestion": match.normalized_question,
                    "sourceRagRunId": str(match.source_rag_run_id),
                    "sourceClassificationId": match.classification_id,
                    "matchedCount": match.matched_count,
                },
            },
        )
        judgment = TurnJudgment(
            decision=ClassificationDecision.CONNECT,
            attribution=subproblem_attribution(match.problem_group_id),
            subproblem=presented,
        )
        outcome = await self._evaluate_gate(
            prepared, judgment, semantic_cache_enabled
        )
        canonical = (
            None if outcome.inputs is None else outcome.inputs.canonical_answer
        )
        canonical_answer_id = None if canonical is None else canonical.canonical_answer_id
        gate_block = gate_judgment_input(
            outcome.gate,
            citation_resolutions=outcome.resolutions,
            canonical_answer_id=canonical_answer_id,
        )
        if outcome.data_error is not None:
            gate_block["dataError"] = outcome.data_error
        synthetic_judged = JudgedTurn(
            call=JudgeCall(
                trace=ModelCallTrace(
                    provider=self._judge_client.provider,
                    model_name=self._judge_client.model_name,
                    succeeded=True,
                    latency_ms=0,
                    prompt_version=self._judge_client.prompt_version,
                )
            ),
            judgment=judgment,
            model_call_id=None,
        )
        recorded = await self._insert_rows(
            prepared,
            None,
            outcome.gate,
            gate_block,
            latency_ms=_elapsed_ms(prepared.started, self._clock()),
            resolutions=outcome.resolutions,
            canonical_answer_id=canonical_answer_id,
            judgment=judgment,
            extra=prepared.exact_match_metadata,
        )
        if not recorded.served:
            return ExactQuestionResult(
                recorded=recorded, prepared=prepared, judged=synthetic_judged
            )
        assert canonical is not None
        return ExactQuestionResult(
            recorded=replace(
                recorded,
                served_citations=served_citation_logs(outcome.gate),
                served_answer_markdown=canonical.content_markdown,
            ),
            prepared=prepared,
            judged=synthetic_judged,
        )

    # ------------------------------------------------------------------
    # 2. 준비
    # ------------------------------------------------------------------

    async def prepare(
        self,
        turn: GroupingTurn,
        resolved_query: str,
        search: HybridSearchCall,
    ) -> PreparedJudgment:
        """질문 벡터, 세부 문제 top5, 문서 top5 와 outline, payload 를 만든다.

        - 검색이 성공한 턴에서만 부른다(R13).
        - 재임베딩할 때만 QUERY_EMBEDDING model_call 을 시작하고 commit 한다. 호출 결과는
          판별 checkpoint 또는 기록 트랜잭션에서 마감한다.
        - 재임베딩 API 오류와 후보 데이터 오류는 failure 로 담아 돌려준다(fail-open).
        - 읽기 트랜잭션은 열어 둔 채 돌려준다. 다음 judge checkpoint 나 기록 트랜잭션이 닫는다.
        """

        started = self._clock()
        base = PreparedJudgment(
            turn=turn,
            resolved_query=resolved_query,
            seed=presentation_seed(turn.rag_run_id),
            model=self._judge_client.model_name,
            prompt_version=self._judge_client.prompt_version,
            vector=QuestionVector(
                embedding=None,
                reused_retrieval_embedding=False,
                retrieval_query=search.retrieval_query,
            ),
            started=started,
        )

        vector, embedding_failure = await self._question_vector(
            turn, resolved_query, search
        )
        prepared = replace(base, vector=vector)
        if embedding_failure is not None:
            return replace(
                prepared,
                failure=embedding_failure,
                failure_stage=FAILURE_STAGE_QUESTION_EMBEDDING,
            )
        assert vector.embedding is not None

        scope: Optional[IndexScope] = None
        catalog: Optional[SubproblemCatalog] = None
        try:
            scope = await self._catalog_reader.load_index_scope(turn.index_version_id)
            if scope.document_group_id != turn.document_group_id:
                raise CatalogDataError(
                    "색인 판의 문서 그룹이 턴의 문서 그룹과 다릅니다: "
                    f"index={scope.document_group_id}, turn={turn.document_group_id}"
                )
            catalog = await self._catalog_reader.load_subproblem_catalog(scope)
            subproblems = rank_subproblems(vector.embedding, catalog.items)
            ranked_documents = rank_documents_from_search(search)
            outlines = await self._outline_reader.load_outlines(
                [item.document_version_id for item in ranked_documents],
                chunking_config_id=scope.chunking_config_id,
            )
            documents = attach_outlines(ranked_documents, outlines)
            presentation = build_judge_presentation(
                resolved_query, subproblems, documents, seed=prepared.seed
            )
        except (CatalogDataError, IndexScopeNotFoundError, ValueError) as error:
            logger.warning(
                "판별 후보를 만들지 못해 판별 실패로 기록합니다: rag_run_id=%s",
                turn.rag_run_id,
                exc_info=error,
            )
            return replace(
                prepared,
                scope=scope,
                catalog=catalog,
                failure=JudgeFailure(
                    kind=JudgeFailureKind.INTERNAL_ERROR,
                    safe_message=_safe_candidate_error_message(error),
                ),
                failure_stage=FAILURE_STAGE_CANDIDATES,
            )

        return replace(
            prepared, scope=scope, catalog=catalog, presentation=presentation
        )

    async def _question_vector(
        self,
        turn: GroupingTurn,
        resolved_query: str,
        search: HybridSearchCall,
    ) -> Tuple[QuestionVector, Optional[JudgeFailure]]:
        retrieval_query = search.retrieval_query
        if (
            retrieval_query is not None
            and retrieval_query == resolved_query
            and search.query_embedding
        ):
            embedding = tuple(float(value) for value in search.query_embedding)
            failure = self._dimension_failure(embedding)
            vector = QuestionVector(
                embedding=None if failure is not None else embedding,
                reused_retrieval_embedding=True,
                retrieval_query=retrieval_query,
            )
            return vector, failure

        # checkpoint 쓰기와 commit 실패는 fail-closed 로 올린다.
        call = await self._log_store.start_model_call(
            purpose=ModelCallPurpose.QUERY_EMBEDDING.value,
            provider=OPENAI_EMBEDDING_PROVIDER,
            model_name=OPENAI_EMBEDDING_MODEL,
            rag_run_id=turn.rag_run_id,
            classification_run_id=turn.classification_run_id,
        )
        model_call_id = call.id
        await self._session.commit()

        started = self._clock()
        attempt = 0
        while True:
            try:
                response = await asyncio.to_thread(
                    self._embedder.embed_many_with_usage,
                    [resolved_query],
                    sdk_max_retries=0,
                    timeout=QUERY_EMBEDDING_TIMEOUT_SECONDS,
                )
                break
            except Exception as error:  # noqa: BLE001 - 판별 실패로 기록한다
                if attempt + 1 < QUERY_EMBEDDING_MAX_ATTEMPTS and is_transient_openai_error(
                    error
                ):
                    await asyncio.sleep(
                        QUERY_EMBEDDING_INITIAL_RETRY_DELAY_SECONDS * (2**attempt)
                    )
                    attempt += 1
                    continue
                logger.warning(
                    "판별용 질문 임베딩에 실패했습니다: rag_run_id=%s",
                    turn.rag_run_id,
                    exc_info=error,
                )
                message = _safe_embedding_error_message(error)
                trace = ModelCallTrace(
                    provider=OPENAI_EMBEDDING_PROVIDER,
                    model_name=OPENAI_EMBEDDING_MODEL,
                    succeeded=False,
                    latency_ms=_elapsed_ms(started, self._clock()),
                    retry_count=attempt,
                    error_message=message,
                )
                kind = (
                    JudgeFailureKind.API_ERROR
                    if getattr(error, "status_code", None) is not None
                    or is_transient_openai_error(error)
                    else JudgeFailureKind.INTERNAL_ERROR
                )
                return (
                    QuestionVector(
                        embedding=None,
                        reused_retrieval_embedding=False,
                        retrieval_query=retrieval_query,
                        model_call_id=model_call_id,
                        trace=trace,
                    ),
                    JudgeFailure(kind=kind, safe_message=message),
                )

        embedding = tuple(float(value) for value in response.embeddings[0])
        trace = ModelCallTrace(
            provider=OPENAI_EMBEDDING_PROVIDER,
            model_name=OPENAI_EMBEDDING_MODEL,
            succeeded=True,
            latency_ms=_elapsed_ms(started, self._clock()),
            retry_count=attempt + response.retry_count,
            input_tokens=response.input_tokens,
        )
        failure = self._dimension_failure(embedding)
        return (
            QuestionVector(
                embedding=None if failure is not None else embedding,
                reused_retrieval_embedding=False,
                retrieval_query=retrieval_query,
                model_call_id=model_call_id,
                trace=trace,
            ),
            failure,
        )

    @staticmethod
    def _dimension_failure(embedding: Sequence[float]) -> Optional[JudgeFailure]:
        if len(embedding) == EMBEDDING_DIMENSIONS:
            return None
        return JudgeFailure(
            kind=JudgeFailureKind.INTERNAL_ERROR,
            safe_message=(
                f"질문 벡터 차원이 {EMBEDDING_DIMENSIONS}이 아닙니다: {len(embedding)}"
            ),
        )

    # ------------------------------------------------------------------
    # 3. 판별 호출
    # ------------------------------------------------------------------

    async def judge(
        self,
        prepared: PreparedJudgment,
        *,
        before_checkpoint: Optional[BeforeJudgeCheckpoint] = None,
    ) -> JudgedTurn:
        """판별을 한 번 호출하고 R7 로 정규화한다.

        외부 호출 직전 checkpoint 한 트랜잭션에서 (1) 대기 중인 재임베딩 model_call 마감,
        (2) before_checkpoint(호출자 쓰기, 예: 검색 임베딩 호출 마감과 검색 후보 기록),
        (3) QUESTION_CLASSIFICATION model_call 시작(rag_run_id + classification_run_id)을 하고
        commit 한다. checkpoint 실패는 올리고(fail-closed), 판별 호출 실패는 JudgedTurn 의
        실패 판별로 담는다(fail-open).
        """

        if not prepared.ready or prepared.presentation is None:
            raise ValueError("준비에 실패한 턴은 판별하지 않습니다.")
        turn = prepared.turn
        state: Dict[str, Any] = {
            "model_call_id": None,
            "embedding_finished": False,
            "checkpoint_error": False,
        }

        async def checkpoint(
            provider: str, model_name: str, prompt_version: Optional[str]
        ) -> None:
            try:
                vector = prepared.vector
                if vector.model_call_id is not None and vector.trace is not None:
                    await self._finish_model_call(vector.model_call_id, vector.trace)
                if before_checkpoint is not None:
                    await before_checkpoint()
                call = await self._log_store.start_model_call(
                    purpose=ModelCallPurpose.QUESTION_CLASSIFICATION.value,
                    provider=provider,
                    model_name=model_name,
                    rag_run_id=turn.rag_run_id,
                    prompt_version=prompt_version,
                    classification_run_id=turn.classification_run_id,
                )
                model_call_id = call.id
                await self._session.commit()
            except BaseException:
                state["checkpoint_error"] = True
                raise
            state["model_call_id"] = model_call_id
            state["embedding_finished"] = (
                vector.model_call_id is not None and vector.trace is not None
            )

        started = self._clock()
        try:
            call = await self._judge_client.judge(
                prepared.presentation.payload, before_model_call=checkpoint
            )
        except Exception as error:  # noqa: BLE001 - checkpoint 오류만 올린다
            if state["checkpoint_error"]:
                raise
            logger.warning(
                "판별 호출 중 예상하지 못한 오류가 발생했습니다: rag_run_id=%s",
                turn.rag_run_id,
                exc_info=error,
            )
            failure = JudgeFailure(
                kind=JudgeFailureKind.INTERNAL_ERROR,
                safe_message=f"판별 호출 중 오류: {type(error).__name__}",
            )
            call = JudgeCall(
                trace=ModelCallTrace(
                    provider=self._judge_client.provider,
                    model_name=prepared.model,
                    succeeded=False,
                    latency_ms=_elapsed_ms(started, self._clock()),
                    prompt_version=prepared.prompt_version,
                    error_message=failure.safe_message,
                ),
                failure=failure,
            )

        try:
            judgment = build_turn_judgment(call, prepared.presentation)
        except Exception as error:  # noqa: BLE001 - 정규화 결함도 판별 실패다
            logger.exception(
                "판별 응답을 정규화하지 못했습니다: rag_run_id=%s", turn.rag_run_id
            )
            judgment = failed_turn_judgment(
                JudgeFailure(
                    kind=JudgeFailureKind.INTERNAL_ERROR,
                    safe_message=f"판별 응답 정규화 오류: {type(error).__name__}",
                )
            )
        if judgment.failed:
            logger.info(
                "판별 실패로 기록합니다: rag_run_id=%s, kind=%s",
                turn.rag_run_id,
                None if judgment.failure is None else judgment.failure.kind.value,
            )
        return JudgedTurn(
            call=call,
            judgment=judgment,
            model_call_id=state["model_call_id"],
            embedding_call_finished=state["embedding_finished"],
        )

    # ------------------------------------------------------------------
    # 4. 기록
    # ------------------------------------------------------------------

    async def record_judgment_and_gate(
        self,
        prepared: PreparedJudgment,
        judged: JudgedTurn,
        *,
        semantic_cache_enabled: bool,
    ) -> RecordedJudgment:
        """판별 결과를 게이트에 넣고 판별 행과 캐시 시도를 쓴다. commit 하지 않는다.

        한 트랜잭션 순서: 턴 잠금 → 재임베딩·판별 model_call 마감 → question_embeddings →
        (CONNECT 면) 게이트 입력 재조회 → 게이트(R16·R17) → judgment_input(gate 포함) →
        question_classifications → question_cache_attempts.

        SERVED 면 served_citations 와 served_answer_markdown 으로 호출자가 같은 트랜잭션에서
        complete_rag_run 을 부르고 commit 한다. 그 외는 호출자가 commit 하고 생성으로 간다.
        """

        if not prepared.ready:
            raise ValueError("준비에 실패한 턴은 record_preparation_failure 로 기록합니다.")
        await self._write_common(prepared, judged)
        outcome = await self._evaluate_gate(prepared, judged.judgment, semantic_cache_enabled)
        latency_ms = _elapsed_ms(prepared.started, self._clock())
        canonical = (
            None if outcome.inputs is None else outcome.inputs.canonical_answer
        )
        canonical_answer_id = (
            None if canonical is None else canonical.canonical_answer_id
        )
        gate_block = gate_judgment_input(
            outcome.gate,
            citation_resolutions=outcome.resolutions,
            canonical_answer_id=canonical_answer_id,
        )
        if outcome.data_error is not None:
            gate_block["dataError"] = outcome.data_error
        recorded = await self._insert_rows(
            prepared,
            judged,
            outcome.gate,
            gate_block,
            latency_ms=latency_ms,
            resolutions=outcome.resolutions,
            canonical_answer_id=canonical_answer_id,
        )
        if not recorded.served:
            return recorded
        assert canonical is not None
        return replace(
            recorded,
            served_citations=served_citation_logs(outcome.gate),
            served_answer_markdown=canonical.content_markdown,
        )

    async def record_preparation_failure(
        self,
        prepared: PreparedJudgment,
    ) -> RecordedJudgment:
        """후보를 만들지 못한 턴: UNCLASSIFIED + NO_DOCUMENT/NONE 행과 FAILED 시도. commit 안 함.

        재임베딩 model_call 을 마감하고, 벡터가 있으면 question_embeddings 도 쓴다.
        """

        if prepared.failure is None:
            raise ValueError("준비에 성공한 턴은 judge 뒤 record_judgment_and_gate 로 기록합니다.")
        await self._write_common(prepared, None)
        judgment = failed_turn_judgment(prepared.failure)
        gate = evaluate_cache_gate(
            judgment,
            subproblem=None,
            canonical_answer=None,
            citation_resolutions=(),
            semantic_cache_enabled=False,
        )
        return await self._insert_rows(
            prepared,
            None,
            gate,
            gate_judgment_input(gate),
            latency_ms=_elapsed_ms(prepared.started, self._clock()),
            judgment=judgment,
        )

    async def record_serve_failure(
        self,
        prepared: PreparedJudgment,
        judged: JudgedTurn,
        recorded: RecordedJudgment,
    ) -> RecordedJudgment:
        """SERVED 트랜잭션(판별 행·시도·턴 완료) 쓰기나 commit 이 실패한 뒤 다시 쓴다(결정 7).

        세션을 rollback 해 실패한 트랜잭션을 버리고, 새 트랜잭션에서 같은 판별(SUBPROBLEM 귀속)과
        REJECTED[CANONICAL_SERVE_FAILED] 게이트·시도를 쓴다. commit 하지 않는다. 호출자는
        commit 뒤 생성 경로로 진행한다. 이 쓰기마저 실패하면 예외를 올린다(fail-closed).
        """

        if not recorded.served:
            raise ValueError("SERVED 기록만 서빙 실패로 다시 씁니다.")
        await self._session.rollback()
        await self._write_common(prepared, judged)
        gate = GateResult(
            outcome=CacheAttemptOutcome.REJECTED,
            rejection_reasons=(REJECT_CANONICAL_SERVE_FAILED,),
        )
        gate_block = gate_judgment_input(
            gate,
            citation_resolutions=recorded.citation_resolutions,
            canonical_answer_id=recorded.canonical_answer_id,
        )
        extra = dict(prepared.exact_match_metadata or {})
        extra["serveFailure"] = {
            "gateOutcome": CacheAttemptOutcome.SERVED.value,
            "cacheAttemptOutcome": CacheAttemptOutcome.REJECTED.value,
        }
        return await self._insert_rows(
            prepared,
            judged,
            gate,
            gate_block,
            latency_ms=recorded.latency_ms,
            resolutions=recorded.citation_resolutions,
            canonical_answer_id=recorded.canonical_answer_id,
            extra=extra,
        )

    async def finalize_attribution(
        self,
        recorded: RecordedJudgment,
        citations: Sequence[CitationLog],
    ) -> bool:
        """턴 끝 인용 귀속(CITATION). 결정 8 순서를 호출자가 지킨다.

        COMPLETED 로 끝날 턴에서 complete_rag_run 과 같은 트랜잭션, 그보다 먼저(턴이 아직
        PROCESSING) 부른다. CONNECT 행과 인용 없는 턴은 쓰지 않고 False 다. commit 하지 않는다.
        """

        if recorded.judgment.decision == ClassificationDecision.CONNECT or not citations:
            return False
        return await self._store.finalize_citation_attribution(
            recorded.classification_id, citations
        )

    # ------------------------------------------------------------------
    # 내부 쓰기
    # ------------------------------------------------------------------

    async def _write_common(
        self,
        prepared: PreparedJudgment,
        judged: Optional[JudgedTurn],
    ) -> None:
        turn = prepared.turn
        await self._log_store.lock_processing_run(turn.rag_run_id)
        vector = prepared.vector
        embedding_finished = judged is not None and judged.embedding_call_finished
        if (
            vector.model_call_id is not None
            and vector.trace is not None
            and not embedding_finished
        ):
            await self._finish_model_call(vector.model_call_id, vector.trace)
        if judged is not None and judged.model_call_id is not None:
            await self._finish_model_call(judged.model_call_id, judged.call.trace)
        if vector.embedding is not None and prepared.scope is not None:
            await self._store.insert_question_embedding(
                turn.rag_run_id,
                embedding=vector.embedding,
                embedding_config_id=prepared.scope.embedding_config_id,
            )

    async def _evaluate_gate(
        self,
        prepared: PreparedJudgment,
        judgment: TurnJudgment,
        semantic_cache_enabled: bool,
    ) -> _GateOutcome:
        if (
            judgment.failed
            or judgment.decision != ClassificationDecision.CONNECT
            or judgment.subproblem is None
        ):
            return _GateOutcome(
                gate=evaluate_cache_gate(
                    judgment,
                    subproblem=None,
                    canonical_answer=None,
                    citation_resolutions=(),
                    semantic_cache_enabled=semantic_cache_enabled,
                )
            )

        assert prepared.scope is not None
        try:
            inputs = await self._catalog_reader.load_gate_inputs(
                judgment.subproblem.subproblem_id, prepared.scope
            )
        except CatalogDataError as error:
            logger.warning(
                "게이트 입력 데이터 오류로 서빙하지 않습니다: rag_run_id=%s",
                prepared.turn.rag_run_id,
                exc_info=error,
            )
            return _GateOutcome(
                gate=GateResult(
                    outcome=CacheAttemptOutcome.REJECTED,
                    rejection_reasons=(REJECT_CANONICAL_DATA_INVALID,),
                ),
                data_error=_safe_candidate_error_message(error),
            )
        resolutions = resolve_citations(inputs.citations, inputs.contexts_by_source_id)
        gate = evaluate_cache_gate(
            judgment,
            subproblem=inputs.subproblem,
            canonical_answer=inputs.canonical_answer,
            citation_resolutions=resolutions,
            semantic_cache_enabled=semantic_cache_enabled,
        )
        return _GateOutcome(gate=gate, resolutions=resolutions, inputs=inputs)

    async def _insert_rows(
        self,
        prepared: PreparedJudgment,
        judged: Optional[JudgedTurn],
        gate: GateResult,
        gate_block: Dict[str, Any],
        *,
        latency_ms: int,
        resolutions: Sequence[CitationResolution] = (),
        canonical_answer_id: Optional[uuid.UUID] = None,
        judgment: Optional[TurnJudgment] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> RecordedJudgment:
        turn = prepared.turn
        if judgment is None:
            if judged is None:
                raise ValueError("판별 결과가 필요합니다.")
            judgment = judged.judgment
        judgment_input = build_judgment_input(prepared, judged, gate_block, extra=extra)
        classification_id = await self._store.insert_classification(
            turn.rag_run_id,
            run_id=turn.classification_run_id,
            judgment=judgment,
            judgment_input=judgment_input,
        )
        cache_attempt_id = await self._store.insert_cache_attempt(
            turn.rag_run_id,
            classification_id=classification_id,
            gate=gate,
            latency_ms=latency_ms,
        )
        return RecordedJudgment(
            classification_id=classification_id,
            cache_attempt_id=cache_attempt_id,
            judgment=judgment,
            gate=gate,
            latency_ms=latency_ms,
            citation_resolutions=tuple(resolutions),
            canonical_answer_id=canonical_answer_id,
        )

    async def _finish_model_call(
        self, model_call_id: int, trace: ModelCallTrace
    ) -> None:
        await self._log_store.finish_model_call(
            model_call_id,
            status=(
                ExecutionStatus.SUCCESS if trace.succeeded else ExecutionStatus.FAILED
            ),
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            cached_input_tokens=trace.cached_input_tokens,
            reasoning_tokens=trace.reasoning_tokens,
            latency_ms=trace.latency_ms,
            retry_count=trace.retry_count,
            error_message=trace.error_message,
        )
