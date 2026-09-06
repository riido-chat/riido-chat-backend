"""실행 중인 Chat API의 Multi-turn 응답과 DB 로그를 함께 평가한다."""

import argparse
import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import httpx
from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database.models import (
    AnswerStatus,
    ContentNode,
    DocumentChunk,
    DocumentSource,
    DocumentVersion,
    IndexDocument,
    IndexVersion,
    RagRun,
)
from app.database.session import dispose_engine, get_session_factory
from app.chat.dependencies import build_chat_service
from app.chat.log_store import RagLogStore
from app.chat.schema import ChatErrorResponse
from app.chat.query_rewrite import (
    OPENAI_QUERY_REWRITE_MODEL,
    MAX_QUERY_REWRITE_TURNS,
    QUERY_REWRITE_MAX_OUTPUT_TOKENS,
    QUERY_REWRITE_PROMPT_V3,
    QUERY_REWRITE_PROMPT_VERSION,
    QueryRewriteCandidateTurn,
    QueryRewriteDecision,
    QueryRewriteService,
    QueryRewriteTurnStatus,
)
from app.answering.generator import (
    ANSWER_PROMPT_V17,
    ANSWER_PROMPT_VERSION,
    ANSWER_REPAIR_PROMPT_V17,
    ANSWER_REPAIR_PROMPT_VERSION,
    GENERATION_PROMPT_VERSION,
    MAX_SOURCE_PLANNING_REGENERATIONS,
    OPENAI_GENERATION_MODEL,
    OpenAIGenerator,
    SOURCE_PLANNING_PROMPT_VERSION,
    SOURCE_PLANNING_REPAIR_PROMPT_VERSION,
    SOURCE_PLANNING_REPAIR_PROMPT_V10,
    SOURCE_PLANNING_PROMPT_V10,
)
from app.answering.models import (
    FinalAnswerStatus,
    FinalGenerationResult,
    FinalWithheldReason,
    GenerationStageTrace,
)
from app.answering.service import GenerationService, WITHHELD_RESPONSES
from app.core.model_trace import ModelCallTrace
from app.core.config import get_settings
from app.retrieval.corpus_state import CorpusState
from app.retrieval.embedding import OPENAI_EMBEDDING_MODEL, OpenAIEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import HybridRetrievalResult, HybridSearchCall
from app.retrieval.search_reader import SearchReader
from app.retrieval.vector_retriever import VectorRetriever


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluation/chat_multiturn_cases.json"
DEFAULT_CS_CASES_PATH = PROJECT_ROOT / "evaluation/chat_cs_cases.json"
DEFAULT_ANSWER_CONSISTENCY_CASES_PATH = (
    PROJECT_ROOT / "evaluation/answer_consistency_cases.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation/chat_multiturn_acceptance_results.json"
)


def detect_repository_revision() -> Optional[str]:
    configured = os.getenv("GITHUB_SHA")
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def detect_worktree_dirty() -> Optional[bool]:
    """평가기 작업 트리에 커밋되지 않은 변경이 있는지 확인한다."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat API와 DB 로그의 Multi-turn 수용 기준을 평가합니다."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--execution-mode",
        choices=("api", "in-process"),
        default="api",
        help="API 서버를 호출하거나 현재 작업 트리 코드를 프로세스 안에서 실행합니다.",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="각 case를 새 conversation에서 반복할 횟수입니다.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="지정한 case만 실행합니다. 여러 번 사용할 수 있습니다.",
    )
    parser.add_argument(
        "--recheck",
        action="store_true",
        help="모델을 다시 호출하지 않고 저장된 결과를 현재 기대값으로 재판정합니다.",
    )
    parser.add_argument(
        "--repository-revision",
        default=detect_repository_revision(),
        help="평가기 commit SHA. 생략하면 현재 HEAD를 사용합니다.",
    )
    parser.add_argument(
        "--target-repository-revision",
        help="실행 대상 서버의 commit SHA. 확인할 수 없으면 생략합니다.",
    )
    parser.add_argument(
        "--recheck-output",
        type=Path,
        help="재판정 결과 경로. 생략하면 원본 옆에 별도 파일을 만듭니다.",
    )
    return parser


def unique_output_path(
    requested_path: Path,
    *,
    label: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Path:
    """기존 평가 파일을 덮어쓰지 않는 출력 경로를 만든다."""

    if not requested_path.exists() and label is None:
        return requested_path

    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    suffix_label = f"-{label}" if label else ""
    candidate = requested_path.with_name(
        f"{requested_path.stem}{suffix_label}-{timestamp}{requested_path.suffix}"
    )
    sequence = 2
    while candidate.exists():
        candidate = requested_path.with_name(
            f"{requested_path.stem}{suffix_label}-{timestamp}-{sequence}"
            f"{requested_path.suffix}"
        )
        sequence += 1
    return candidate


def load_cases(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("version"), str):
        raise ValueError("평가 세트 version이 필요합니다.")
    if not isinstance(payload.get("cases"), list) or not payload["cases"]:
        raise ValueError("평가할 cases가 필요합니다.")
    return payload


def select_cases(
    payload: Dict[str, Any],
    case_ids: Optional[Sequence[str]],
) -> Dict[str, Any]:
    if not case_ids:
        return payload
    requested = list(dict.fromkeys(case_ids))
    case_by_id = {case["id"]: case for case in payload["cases"]}
    unknown = [case_id for case_id in requested if case_id not in case_by_id]
    if unknown:
        raise ValueError(f"알 수 없는 case id입니다: {unknown}")
    return {
        "version": payload["version"],
        "cases": [case_by_id[case_id] for case_id in requested],
    }


def selected_turn_nos(context_snapshot: Optional[dict]) -> List[int]:
    if context_snapshot is None:
        return []
    turns = context_snapshot.get("selectedTurns")
    if not isinstance(turns, list):
        return []
    return [
        turn["turnNo"]
        for turn in turns
        if isinstance(turn, dict) and isinstance(turn.get("turnNo"), int)
    ]


def response_answer_markdown(response_body: Dict[str, Any]) -> str:
    answer = response_body.get("answer")
    if not isinstance(answer, dict):
        return ""
    answer_markdown = answer.get("answerMarkdown")
    return answer_markdown if isinstance(answer_markdown, str) else ""


def answer_lead_paragraph(answer_markdown: str) -> str:
    return next(
        (
            paragraph.strip()
            for paragraph in answer_markdown.split("\n\n")
            if paragraph.strip()
        ),
        "",
    )


def answer_lead_sentence(answer_markdown: str) -> str:
    lead_paragraph = answer_lead_paragraph(answer_markdown)
    sentence_ends = [
        lead_paragraph.find(delimiter)
        for delimiter in (".", "!", "?")
        if delimiter in lead_paragraph
    ]
    if not sentence_ends:
        return lead_paragraph
    first_end = min(sentence_ends)
    return lead_paragraph[: first_end + 1].strip()


def response_citation_titles(response_body: Dict[str, Any]) -> List[str]:
    citations = response_body.get("citations")
    if not isinstance(citations, list):
        return []
    return [
        citation["documentTitle"]
        for citation in citations
        if isinstance(citation, dict)
        and isinstance(citation.get("documentTitle"), str)
    ]


def evaluate_turn(
    expected: Dict[str, Any],
    response_status_code: int,
    response_body: Dict[str, Any],
    db_snapshot: Optional[Dict[str, Any]],
) -> List[str]:
    """한 턴의 API 상태와 DB 문맥 기록이 기대값에 맞는지 판정한다."""

    failures = []
    if response_status_code != 200:
        failures.append(f"HTTP 200이 아님: {response_status_code}")

    expected_status = expected.get("expectedStatus")
    if response_body.get("status") != expected_status:
        failures.append(
            f"status 불일치: expected={expected_status}, "
            f"actual={response_body.get('status')}"
        )

    expected_reason = expected.get("expectedWithheldReason")
    if expected_reason is not None:
        withheld = response_body.get("withheld") or {}
        if withheld.get("reasonCode") != expected_reason:
            failures.append(
                f"withheld reason 불일치: expected={expected_reason}, "
                f"actual={withheld.get('reasonCode')}"
            )

    answer_markdown = response_answer_markdown(response_body)
    lead_paragraph = answer_lead_paragraph(answer_markdown)
    lead_sentence = answer_lead_sentence(answer_markdown)
    for concept_group in expected.get(
        "expectedDefinitionSentenceConceptGroups",
        [],
    ):
        if not any(keyword in lead_sentence for keyword in concept_group):
            failures.append(
                "첫 문장에 독립적인 용어 정의 개념이 없음: "
                f"alternatives={concept_group!r}"
            )

    for concept_group in expected.get("expectedLeadConceptGroups", []):
        if not any(keyword in lead_paragraph for keyword in concept_group):
            failures.append(
                "첫 문단에 용어 정의 개념이 없음: "
                f"alternatives={concept_group!r}"
            )

    for concept_group in expected.get("expectedAnswerConceptGroups", []):
        if not any(keyword in answer_markdown for keyword in concept_group):
            failures.append(
                "답변에 기대 개념이 없음: "
                f"alternatives={concept_group!r}"
            )

    citation_titles = response_citation_titles(response_body)
    minimum_citation_count = expected.get("minimumCitationCount")
    if (
        minimum_citation_count is not None
        and len(citation_titles) < minimum_citation_count
    ):
        failures.append(
            "Citation 개수 부족: "
            f"minimum={minimum_citation_count}, actual={len(citation_titles)}"
        )

    expected_citation_titles = expected.get(
        "expectedCitationDocumentTitlesAny",
        [],
    )
    if expected_citation_titles and not any(
        title in citation_titles for title in expected_citation_titles
    ):
        failures.append(
            "기대 Citation 문서가 없음: "
            f"expected_any={expected_citation_titles!r}, "
            f"actual={citation_titles!r}"
        )

    if db_snapshot is None:
        failures.append("ragRunId에 대응하는 DB 로그가 없음")
        return failures

    if db_snapshot["status"] != expected_status:
        failures.append(
            f"DB status 불일치: expected={expected_status}, "
            f"actual={db_snapshot['status']}"
        )

    expected_strategy = expected.get("expectedContextStrategy")
    if (
        expected_strategy is not None
        and db_snapshot["contextStrategy"] != expected_strategy
    ):
        failures.append(
            f"contextStrategy 불일치: expected={expected_strategy}, "
            f"actual={db_snapshot['contextStrategy']}"
        )

    expected_turn_nos = expected.get("expectedSelectedTurnNos")
    if (
        expected_turn_nos is not None
        and db_snapshot["selectedTurnNos"] != expected_turn_nos
    ):
        failures.append(
            f"selectedTurnNos 불일치: expected={expected_turn_nos}, "
            f"actual={db_snapshot['selectedTurnNos']}"
        )

    resolved_query = db_snapshot.get("resolvedQuery") or ""
    for keyword in expected.get("resolvedQueryKeywords", []):
        if keyword not in resolved_query:
            failures.append(
                f"resolvedQuery에 키워드가 없음: {keyword!r} "
                f"(actual={resolved_query!r})"
            )

    return failures


def summarize_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """반복 회차 전체와 case별 통과 횟수를 함께 집계한다."""

    case_results = [
        result
        for run in runs
        for result in run["results"]
    ]
    per_case: Dict[str, Dict[str, Any]] = {}
    per_mode: Dict[str, Dict[str, Any]] = {}
    for result in case_results:
        case_summary = per_case.setdefault(
            result["id"],
            {
                "id": result["id"],
                "executionCount": 0,
                "passedCount": 0,
                "failedCount": 0,
            },
        )
        case_summary["executionCount"] += 1
        if result["passed"]:
            case_summary["passedCount"] += 1
        else:
            case_summary["failedCount"] += 1

        mode = result.get("mode", "UNSPECIFIED")
        mode_summary = per_mode.setdefault(
            mode,
            {
                "mode": mode,
                "executionCount": 0,
                "passedCount": 0,
                "failedCount": 0,
            },
        )
        mode_summary["executionCount"] += 1
        if result["passed"]:
            mode_summary["passedCount"] += 1
        else:
            mode_summary["failedCount"] += 1

    passed_count = sum(result["passed"] for result in case_results)
    return {
        "repetitionCount": len(runs),
        "uniqueCaseCount": len(per_case),
        "totalCaseExecutionCount": len(case_results),
        "passedCaseExecutionCount": passed_count,
        "failedCaseExecutionCount": len(case_results) - passed_count,
        "perCase": list(per_case.values()),
        "perMode": list(per_mode.values()),
    }


def _recheck_run_results(
    cases_payload: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    case_by_id = {case["id"]: case for case in cases_payload["cases"]}
    rechecked_results = []
    for saved_result in results:
        case_id = saved_result["id"]
        if case_id not in case_by_id:
            raise ValueError(f"평가 세트에 case가 없습니다: {case_id}")
        case = case_by_id[case_id]
        result = copy.deepcopy(saved_result)
        if len(result["turns"]) != len(case["turns"]):
            raise ValueError(f"turn 수가 일치하지 않습니다: {case_id}")

        for expected, turn in zip(case["turns"], result["turns"]):
            if turn.get("question") != expected["question"]:
                raise ValueError(
                    f"저장된 질문이 평가 세트와 다릅니다: {case_id} "
                    f"turn={turn.get('turnNo')}"
                )
            failures = evaluate_turn(
                expected,
                turn["httpStatus"],
                turn["response"],
                turn.get("db"),
            )
            turn["failures"] = failures
            turn["passed"] = not failures

        result["description"] = case.get("description")
        result["passed"] = all(turn["passed"] for turn in result["turns"])
        rechecked_results.append(result)
    return rechecked_results


def recheck_saved_results(
    cases_payload: Dict[str, Any],
    saved_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """저장된 API·DB snapshot을 현재 평가 세트로 다시 판정한다."""

    payload = copy.deepcopy(saved_payload)
    payload["recheckedAt"] = datetime.now(timezone.utc).isoformat()
    payload["casesVersion"] = cases_payload["version"]
    if "runs" in saved_payload:
        runs = []
        for saved_run in saved_payload["runs"]:
            run = copy.deepcopy(saved_run)
            run["results"] = _recheck_run_results(
                cases_payload,
                saved_run["results"],
            )
            runs.append(run)
        payload["runs"] = runs
        payload["summary"] = summarize_runs(runs)
    else:
        rechecked_results = _recheck_run_results(
            cases_payload,
            saved_payload["results"],
        )
        passed_count = sum(result["passed"] for result in rechecked_results)
        payload["summary"] = {
            "totalCaseCount": len(rechecked_results),
            "passedCaseCount": passed_count,
            "failedCaseCount": len(rechecked_results) - passed_count,
        }
        payload["results"] = rechecked_results
    return payload


def generation_stage_trace_snapshot(
    trace: GenerationStageTrace,
) -> Dict[str, Any]:
    """메모리 전용 Generation trace를 JSON 저장 구조로 바꾼다."""

    return {
        "planningOutput": (
            None
            if trace.source_plan is None
            else trace.source_plan.model_dump(mode="json")
        ),
        "initialPlanningOutput": (
            None
            if trace.initial_source_plan is None
            else trace.initial_source_plan.model_dump(mode="json")
        ),
        "selectedSources": [
            {
                "sourceId": source.source_id,
                "chunkId": source.chunk.chunk_id,
                "documentVersionId": source.chunk.document_version_id,
                "documentId": source.chunk.document_id,
                "sectionId": source.chunk.section_id,
                "documentTitle": source.chunk.document_title,
                "sectionPath": list(source.chunk.section_path),
                "sourceUrl": source.chunk.source_url,
                "content": source.chunk.content,
            }
            for source in trace.selected_sources
        ],
        "preValidationResult": (
            None
            if trace.pre_validation_result is None
            else trace.pre_validation_result.model_dump(mode="json")
        ),
        "validationError": trace.validation_error,
        "validationErrors": list(trace.validation_errors),
        "planningAttemptCount": trace.planning_attempt_count,
        "planningRegenerationCount": trace.planning_regeneration_count,
        "planningRegenerationModelCall": (
            None
            if trace.planning_regeneration_model_call is None
            else _model_trace_snapshot(
                "SOURCE_PLAN_REGENERATION",
                trace.planning_regeneration_model_call,
            )
        ),
        "planningRegenerationResult": (
            None
            if trace.planning_regeneration_result is None
            else trace.planning_regeneration_result.model_dump(mode="json")
        ),
        "answerAttemptCount": trace.answer_attempt_count,
        "validationRegenerationCount": trace.validation_regeneration_count,
        "validationRegenerationModelCall": (
            None
            if trace.validation_regeneration_model_call is None
            else _model_trace_snapshot(
                "ANSWER_VALIDATION_REGENERATION",
                trace.validation_regeneration_model_call,
            )
        ),
        "validationRegenerationResult": (
            None
            if trace.validation_regeneration_result is None
            else trace.validation_regeneration_result.model_dump(mode="json")
        ),
    }


async def load_db_snapshot(
    rag_run_id: str,
    generation_stage_trace: Optional[GenerationStageTrace] = None,
) -> Optional[Dict[str, Any]]:
    async with get_session_factory()() as session:
        store = RagLogStore(session)
        parsed_rag_run_id = uuid.UUID(rag_run_id)
        detail = await store.get_rag_run_detail(parsed_rag_run_id)
        if detail is None:
            return None
        candidate_turns = await _load_query_rewrite_candidates(
            session,
            detail.run,
        )
        top_5 = await _load_retrieval_top_5(session, detail.retrieval_results)
        model_calls = [_to_model_call_snapshot(call) for call in detail.model_calls]
        generation_snapshot = (
            None
            if generation_stage_trace is None
            else generation_stage_trace_snapshot(generation_stage_trace)
        )
        return {
            "status": detail.run.status.value,
            "indexVersionId": detail.run.index_version_id,
            "turnNo": detail.run.turn_no,
            "userQuery": detail.run.user_query,
            "resolvedQuery": detail.run.resolved_query,
            "contextStrategy": detail.run.context_strategy.value,
            "contextTurnCount": detail.run.context_turn_count,
            "selectedTurnNos": selected_turn_nos(detail.run.context_snapshot),
            "withheldReasonCode": detail.run.withheld_reason_code,
            "errorCode": detail.run.error_code,
            "citationValidated": detail.run.citation_validated,
            "totalLatencyMs": detail.run.total_latency_ms,
            "modelCalls": model_calls,
            "retrievalResultCount": len(detail.retrieval_results),
            "citationCount": len(detail.citations),
            "citationDocumentTitles": [
                citation.document_title_snapshot for citation in detail.citations
            ],
            "stageTrace": {
                "queryResolution": {
                    "originalQuestion": detail.run.user_query,
                    "candidateTurns": [
                        candidate
                        for candidate in candidate_turns
                    ],
                    "selectedTurns": (
                        []
                        if detail.run.context_snapshot is None
                        else detail.run.context_snapshot.get("selectedTurns", [])
                    ),
                    "contextStrategy": detail.run.context_strategy.value,
                    "resolvedQuery": detail.run.resolved_query,
                },
                "retrieval": {
                    "top5": top_5,
                },
                "generation": {
                    "modelCalls": [
                        call
                        for call in model_calls
                        if call["purpose"] == "ANSWER_GENERATION"
                    ],
                    "planningOutput": {
                        "availability": (
                            "AVAILABLE"
                            if generation_snapshot is not None
                            else "UNAVAILABLE"
                        ),
                        "value": (
                            None
                            if generation_snapshot is None
                            else generation_snapshot["planningOutput"]
                        ),
                        "reason": (
                            "현재 Generation 로그에 Planning 출력이 저장되지 않음"
                            if generation_snapshot is None
                            else None
                        ),
                    },
                    "selectedSources": {
                        "availability": (
                            "AVAILABLE"
                            if generation_snapshot is not None
                            else "UNAVAILABLE"
                        ),
                        "value": (
                            None
                            if generation_snapshot is None
                            else generation_snapshot["selectedSources"]
                        ),
                        "reason": (
                            "현재 로그의 selectedAsEvidence는 Planning 선택이 아닌 Top-5 포함 여부임"
                            if generation_snapshot is None
                            else None
                        ),
                    },
                    "preValidationAnswer": {
                        "availability": (
                            "AVAILABLE"
                            if generation_snapshot is not None
                            else "UNAVAILABLE"
                        ),
                        "value": (
                            None
                            if generation_snapshot is None
                            else generation_snapshot["preValidationResult"]
                        ),
                        "reason": (
                            "현재 Generation 로그에 검증 전 답변이 저장되지 않음"
                            if generation_snapshot is None
                            else None
                        ),
                    },
                    "validationError": (
                        None
                        if generation_snapshot is None
                        else generation_snapshot["validationError"]
                    ),
                    "validationErrors": (
                        []
                        if generation_snapshot is None
                        else generation_snapshot["validationErrors"]
                    ),
                    "planningAttemptCount": (
                        0
                        if generation_snapshot is None
                        else generation_snapshot["planningAttemptCount"]
                    ),
                    "planningRegenerationCount": (
                        0
                        if generation_snapshot is None
                        else generation_snapshot["planningRegenerationCount"]
                    ),
                    "planningRegenerationModelCall": (
                        None
                        if generation_snapshot is None
                        else generation_snapshot[
                            "planningRegenerationModelCall"
                        ]
                    ),
                    "initialPlanningOutput": (
                        None
                        if generation_snapshot is None
                        else generation_snapshot["initialPlanningOutput"]
                    ),
                    "planningRegenerationOutput": (
                        None
                        if generation_snapshot is None
                        else generation_snapshot[
                            "planningRegenerationResult"
                        ]
                    ),
                    "answerAttemptCount": (
                        0
                        if generation_snapshot is None
                        else generation_snapshot["answerAttemptCount"]
                    ),
                    "validationRegenerationCount": (
                        0
                        if generation_snapshot is None
                        else generation_snapshot["validationRegenerationCount"]
                    ),
                    "validationRegenerationModelCall": (
                        None
                        if generation_snapshot is None
                        else generation_snapshot[
                            "validationRegenerationModelCall"
                        ]
                    ),
                    "validationRegenerationAnswer": (
                        None
                        if generation_snapshot is None
                        else generation_snapshot[
                            "validationRegenerationResult"
                        ]
                    ),
                    "finalAnswer": detail.run.answer_content,
                    "citationValidated": detail.run.citation_validated,
                },
                "terminal": _terminal_stage_snapshot(detail),
            },
        }


async def _load_query_rewrite_candidates(
    session: Any,
    current_run: RagRun,
) -> List[Dict[str, Any]]:
    """완료된 평가 턴을 기준으로 당시 Query Rewrite 후보를 복원한다."""

    recent_runs = (
        (
            await session.execute(
                select(RagRun)
                .where(
                    RagRun.conversation_id == current_run.conversation_id,
                    RagRun.turn_no < current_run.turn_no,
                    RagRun.status.in_(
                        (AnswerStatus.COMPLETED, AnswerStatus.WITHHELD)
                    ),
                )
                .order_by(RagRun.turn_no.desc())
                .limit(MAX_QUERY_REWRITE_TURNS)
            )
        )
        .scalars()
        .all()
    )
    return [
        {
            "ragRunId": str(run.id),
            "turnNo": run.turn_no,
            "status": run.status.value,
            "userQuery": run.user_query,
            "answerContent": (
                run.answer_content
                if run.status == AnswerStatus.COMPLETED
                else None
            ),
            "withheldReasonCode": (
                run.withheld_reason_code
                if run.status == AnswerStatus.WITHHELD
                else None
            ),
        }
        for run in reversed(recent_runs)
    ]


def _to_model_call_snapshot(call: Any) -> Dict[str, Any]:
    """DB ModelCall을 요청/응답 모델 구분이 드러나는 평가 snapshot으로 바꾼다."""

    return {
        "purpose": call.purpose.value,
        "status": call.status.value,
        "provider": call.provider,
        "requestedModel": call.model_name,
        "responseModel": None,
        "responseModelAvailability": "UNAVAILABLE",
        "promptVersion": call.prompt_version,
        "inputTokens": call.input_tokens,
        "outputTokens": call.output_tokens,
        "latencyMs": call.latency_ms,
        "retryCount": call.retry_count,
        "errorMessage": call.error_message,
    }


def _terminal_stage_snapshot(detail: Any) -> Dict[str, Any]:
    reason = detail.run.withheld_reason_code or detail.run.error_code
    failed_model_calls = [
        call for call in detail.model_calls if call.status.value == "FAILED"
    ]
    if failed_model_calls:
        failure_stage = failed_model_calls[-1].purpose.value
    elif detail.run.withheld_reason_code == "UNVERIFIABLE_ANSWER":
        failure_stage = "CITATION_VALIDATION"
    elif (
        detail.run.withheld_reason_code == "AMBIGUOUS_QUESTION"
        and not detail.retrieval_results
    ):
        failure_stage = "QUERY_REWRITE"
    elif detail.run.withheld_reason_code is not None:
        failure_stage = "GENERATION_DECISION"
    elif detail.run.error_code is not None:
        failure_stage = "UNKNOWN"
    else:
        failure_stage = None
    return {
        "status": detail.run.status.value,
        "failureStage": failure_stage,
        "reason": reason,
    }


async def _load_retrieval_top_5(
    session: Any,
    retrieval_rows: Sequence[Any],
) -> List[Dict[str, Any]]:
    """검색 로그의 융합 Top-5를 실제 Chunk 내용과 결합한다."""

    fused_rows_by_chunk: Dict[int, List[Any]] = {}
    for row in retrieval_rows:
        if row.fused_rank is not None and row.selected_as_evidence:
            fused_rows_by_chunk.setdefault(row.chunk_id, []).append(row)
    if not fused_rows_by_chunk:
        return []

    statement = (
        select(DocumentChunk, ContentNode, DocumentVersion, DocumentSource)
        .join(ContentNode, ContentNode.id == DocumentChunk.id)
        .join(
            DocumentVersion,
            DocumentVersion.id == ContentNode.document_version_id,
        )
        .join(
            DocumentSource,
            DocumentSource.id == DocumentVersion.document_source_id,
        )
        .where(DocumentChunk.id.in_(fused_rows_by_chunk))
    )
    chunk_rows = (await session.execute(statement)).all()
    chunk_snapshot_by_id = {}
    for document_chunk, node, version, source in chunk_rows:
        metadata = node.metadata_ or {}
        chunk_snapshot_by_id[document_chunk.id] = {
            "chunkId": document_chunk.id,
            "documentVersionId": version.id,
            "documentId": metadata.get("document_id"),
            "sectionId": metadata.get("section_id"),
            "documentTitle": source.title,
            "sectionPath": metadata.get("section_path"),
            "sourceUrl": source.canonical_uri,
            "contentHash": node.content_hash,
            "content": node.normalized_content,
        }

    snapshots = []
    for chunk_id, rows in fused_rows_by_chunk.items():
        first = rows[0]
        snapshot = chunk_snapshot_by_id.get(chunk_id, {"chunkId": chunk_id})
        snapshot.update(
            {
                "fusedRank": first.fused_rank,
                "fusedScore": (
                    None if first.fused_score is None else float(first.fused_score)
                ),
                "retrievers": [
                    {
                        "type": row.retriever_type.value,
                        "rank": row.retriever_rank,
                        "rawScore": (
                            None if row.raw_score is None else float(row.raw_score)
                        ),
                        "latencyMs": row.latency_ms,
                    }
                    for row in rows
                ],
            }
        )
        snapshots.append(snapshot)
    return sorted(snapshots, key=lambda item: item["fusedRank"])


async def load_index_metadata(index_version_id: Optional[int]) -> Optional[dict]:
    if index_version_id is None:
        return None
    async with get_session_factory()() as session:
        index = await session.get(IndexVersion, index_version_id)
        if index is None:
            return None
        document_rows = (
            await session.execute(
                select(
                    IndexDocument.document_version_id,
                    DocumentVersion.normalized_content_hash,
                )
                .join(
                    DocumentVersion,
                    DocumentVersion.id == IndexDocument.document_version_id,
                )
                .where(IndexDocument.index_version_id == index_version_id)
                .order_by(IndexDocument.document_version_id)
            )
        ).all()
        fingerprint_input = "\n".join(
            f"{document_version_id}:{content_hash}"
            for document_version_id, content_hash in document_rows
        )
        return {
            "id": index.id,
            "version": index.version,
            "status": index.status.value,
            "documentCount": len(document_rows),
            "documentContentFingerprint": text_sha256(fingerprint_input),
            "createdAt": index.created_at.isoformat(),
            "activatedAt": (
                None if index.activated_at is None else index.activated_at.isoformat()
            ),
        }


TurnExecutor = Callable[
    [str, Optional[str]],
    Awaitable[Tuple[int, Dict[str, Any], Optional[GenerationStageTrace]]],
]


def _model_trace_snapshot(
    purpose: str,
    trace: Optional[ModelCallTrace],
) -> Optional[Dict[str, Any]]:
    if trace is None:
        return None
    return {
        "purpose": purpose,
        "status": "SUCCESS" if trace.succeeded else "FAILED",
        "provider": trace.provider,
        "requestedModel": trace.model_name,
        "responseModel": None,
        "responseModelAvailability": "UNAVAILABLE",
        "promptVersion": trace.prompt_version,
        "inputTokens": trace.input_tokens,
        "outputTokens": trace.output_tokens,
        "latencyMs": trace.latency_ms,
        "retryCount": trace.retry_count,
        "errorMessage": trace.error_message,
    }


def _hybrid_top_5_snapshot(
    results: Sequence[HybridRetrievalResult],
) -> List[Dict[str, Any]]:
    return [
        {
            "fusedRank": result.final_rank,
            "fusedScore": result.rrf_score,
            "bm25Rank": result.bm25_rank,
            "vectorRank": result.vector_rank,
            "chunkId": result.chunk.chunk_id,
            "documentVersionId": result.chunk.document_version_id,
            "documentId": result.chunk.document_id,
            "sectionId": result.chunk.section_id,
            "documentTitle": result.chunk.document_title,
            "sectionPath": list(result.chunk.section_path),
            "sourceUrl": result.chunk.source_url,
            "contentSnapshotHash": text_sha256(result.chunk.content),
            "content": result.chunk.content,
        }
        for result in results
    ]


def _generation_snapshot(
    result: Optional[FinalGenerationResult],
) -> Dict[str, Any]:
    stage_trace = None if result is None else result.stage_trace
    trace_snapshot = (
        None
        if stage_trace is None
        else generation_stage_trace_snapshot(stage_trace)
    )
    return {
        "modelCalls": [],
        "planningOutput": {
            "availability": "AVAILABLE" if trace_snapshot else "UNAVAILABLE",
            "value": None if trace_snapshot is None else trace_snapshot["planningOutput"],
        },
        "selectedSources": {
            "availability": "AVAILABLE" if trace_snapshot else "UNAVAILABLE",
            "value": None if trace_snapshot is None else trace_snapshot["selectedSources"],
        },
        "preValidationAnswer": {
            "availability": "AVAILABLE" if trace_snapshot else "UNAVAILABLE",
            "value": (
                None
                if trace_snapshot is None
                else trace_snapshot["preValidationResult"]
            ),
        },
        "validationError": (
            None if trace_snapshot is None else trace_snapshot["validationError"]
        ),
        "validationErrors": (
            [] if trace_snapshot is None else trace_snapshot["validationErrors"]
        ),
        "planningAttemptCount": (
            0 if trace_snapshot is None else trace_snapshot["planningAttemptCount"]
        ),
        "planningRegenerationCount": (
            0
            if trace_snapshot is None
            else trace_snapshot["planningRegenerationCount"]
        ),
        "planningRegenerationModelCall": (
            None
            if trace_snapshot is None
            else trace_snapshot["planningRegenerationModelCall"]
        ),
        "initialPlanningOutput": (
            None
            if trace_snapshot is None
            else trace_snapshot["initialPlanningOutput"]
        ),
        "planningRegenerationOutput": (
            None
            if trace_snapshot is None
            else trace_snapshot["planningRegenerationResult"]
        ),
        "answerAttemptCount": (
            0 if trace_snapshot is None else trace_snapshot["answerAttemptCount"]
        ),
        "validationRegenerationCount": (
            0
            if trace_snapshot is None
            else trace_snapshot["validationRegenerationCount"]
        ),
        "validationRegenerationModelCall": (
            None
            if trace_snapshot is None
            else trace_snapshot["validationRegenerationModelCall"]
        ),
        "validationRegenerationAnswer": (
            None
            if trace_snapshot is None
            else trace_snapshot["validationRegenerationResult"]
        ),
        "finalAnswer": None if result is None else result.answer_markdown,
        "citationValidated": (
            None
            if result is None
            else result.status == FinalAnswerStatus.COMPLETED
        ),
    }


def _final_generation_response(
    result: FinalGenerationResult,
) -> Tuple[int, Dict[str, Any]]:
    conversation_id = str(uuid.uuid4())
    rag_run_id = str(uuid.uuid4())
    if result.status == FinalAnswerStatus.COMPLETED:
        citations = []
        for citation in result.citations:
            section_path = list(citation.section_path)
            if section_path and section_path[0] == citation.document_title:
                section_path = section_path[1:]
            citations.append(
                {
                    "citationNumber": citation.citation_number,
                    "documentTitle": citation.document_title,
                    "sectionPath": section_path,
                    "sourceUrl": citation.source_url,
                    "sourceKind": citation.source_kind.value,
                }
            )
        return 200, {
            "status": "COMPLETED",
            "conversationId": conversation_id,
            "ragRunId": rag_run_id,
            "answer": {"answerMarkdown": result.answer_markdown},
            "citations": citations,
        }
    if result.status == FinalAnswerStatus.WITHHELD:
        return 200, {
            "status": "WITHHELD",
            "conversationId": conversation_id,
            "ragRunId": rag_run_id,
            "answer": None,
            "withheld": {
                "reasonCode": result.withheld_reason.value,
                "message": WITHHELD_RESPONSES[result.withheld_reason],
            },
            "citations": [],
        }
    return 500, {
        "status": "ERROR",
        "conversationId": conversation_id,
        "ragRunId": rag_run_id,
        "answer": None,
        "error": {
            "code": result.error_code,
            "message": "평가 실행 중 모델 또는 검증 오류가 발생했습니다.",
            "retryable": False,
        },
        "citations": [],
    }


def _fixed_context_candidates(
    case: Dict[str, Any],
) -> Tuple[QueryRewriteCandidateTurn, ...]:
    fixed_context = case.get("fixedContext")
    if not isinstance(fixed_context, dict):
        raise ValueError(f"fixedContext가 필요합니다: {case['id']}")
    raw_candidates = fixed_context.get("candidateTurns")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"fixedContext.candidateTurns가 필요합니다: {case['id']}")

    candidates = []
    for raw in raw_candidates:
        turn_no = raw["turnNo"]
        candidates.append(
            QueryRewriteCandidateTurn(
                ragRunId=uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"riido-evaluation:{case['id']}:{turn_no}",
                ),
                turnNo=turn_no,
                status=QueryRewriteTurnStatus(raw["status"]),
                userQuery=raw["userQuery"],
                answerContent=raw.get("answerContent"),
                withheldReasonCode=raw.get("withheldReasonCode"),
            )
        )
    return tuple(candidates)


class InProcessEvaluationRuntime:
    """현재 작업 트리의 ChatService를 평가 프로세스 안에서 실행한다."""

    def __init__(
        self,
        corpus_state: CorpusState,
        embedder: OpenAIEmbedder,
        generation_service: GenerationService,
        query_rewrite_service: QueryRewriteService,
    ) -> None:
        self._corpus_state = corpus_state
        self._embedder = embedder
        self._generation_service = generation_service
        self._query_rewrite_service = query_rewrite_service

    @classmethod
    async def create(cls) -> "InProcessEvaluationRuntime":
        async with get_session_factory()() as session:
            chunks = await SearchReader(session).load_active_chunks()
        corpus_state = CorpusState(get_settings().corpus_dir)
        corpus_state.replace(chunks)
        return cls(
            corpus_state=corpus_state,
            embedder=OpenAIEmbedder(),
            generation_service=GenerationService(OpenAIGenerator()),
            query_rewrite_service=QueryRewriteService(),
        )

    async def execute_turn(
        self,
        question: str,
        conversation_id: Optional[str],
    ) -> Tuple[int, Dict[str, Any], Optional[GenerationStageTrace]]:
        observed_trace: Optional[GenerationStageTrace] = None

        def capture_trace(
            _rag_run_id: uuid.UUID,
            trace: GenerationStageTrace,
        ) -> None:
            nonlocal observed_trace
            observed_trace = trace

        parsed_conversation_id = (
            None if conversation_id is None else uuid.UUID(conversation_id)
        )
        async with get_session_factory()() as session:
            service = build_chat_service(
                session=session,
                corpus_state=self._corpus_state,
                embedder=self._embedder,
                generation_service=self._generation_service,
                query_rewrite_service=self._query_rewrite_service,
            )
            response = await service.answer_question(
                question,
                parsed_conversation_id,
                on_generation_stage_trace=capture_trace,
            )

        status_code = 500 if isinstance(response, ChatErrorResponse) else 200
        return (
            status_code,
            response.model_dump(mode="json", by_alias=True),
            observed_trace,
        )

    async def run_fixed_context_case(
        self,
        case: Dict[str, Any],
    ) -> Dict[str, Any]:
        """DB에 fixture 턴을 만들지 않고 고정 문맥의 현재 질문만 평가한다."""

        if len(case["turns"]) != 1:
            raise ValueError(
                f"고정 문맥 case의 평가 turn은 하나여야 합니다: {case['id']}"
            )
        expected = case["turns"][0]
        question = expected["question"]
        candidates = _fixed_context_candidates(case)
        model_calls = []
        query_call = await self._query_rewrite_service.rewrite(
            question,
            candidates,
        )
        query_model_call = _model_trace_snapshot(
            "QUERY_REWRITE",
            query_call.trace,
        )
        if query_model_call is not None:
            model_calls.append(query_model_call)

        resolution = query_call.resolution
        search = HybridSearchCall()
        generation_result: Optional[FinalGenerationResult] = None
        if query_call.error is not None:
            status_code = 500
            body = {
                "status": "ERROR",
                "answer": None,
                "error": {"code": query_call.error_code},
                "citations": [],
            }
            failure_stage = "QUERY_REWRITE"
            terminal_reason = query_call.error_code
        elif resolution is None:
            raise RuntimeError("Query Rewrite 결과에 resolution이 없습니다.")
        elif not resolution.should_retrieve:
            status_code = 200
            body = {
                "status": "WITHHELD",
                "answer": None,
                "withheld": {
                    "reasonCode": "AMBIGUOUS_QUESTION",
                    "message": WITHHELD_RESPONSES[
                        FinalWithheldReason.AMBIGUOUS_QUESTION
                    ],
                },
                "citations": [],
            }
            failure_stage = "QUERY_REWRITE"
            terminal_reason = "AMBIGUOUS_QUESTION"
        else:
            async with get_session_factory()() as session:
                retriever = HybridRetriever(
                    bm25_retriever=self._corpus_state.get_retriever(),
                    vector_retriever=VectorRetriever(
                        embedder=self._embedder,
                        store=SearchReader(session),
                    ),
                )
                search = await retriever.search_with_trace(
                    resolution.resolved_query
                )
            embedding_model_call = _model_trace_snapshot(
                "QUERY_EMBEDDING",
                search.embedding_call,
            )
            if embedding_model_call is not None:
                model_calls.append(embedding_model_call)

            if search.error is not None:
                status_code = 500
                body = {
                    "status": "ERROR",
                    "answer": None,
                    "error": {"code": "RETRIEVAL_ERROR"},
                    "citations": [],
                }
                failure_stage = "RETRIEVAL"
                terminal_reason = type(search.error).__name__
            else:
                generation_result = await self._generation_service.generate_answer(
                    resolution.resolved_query,
                    search.fused_results,
                )
                generation_model_call = _model_trace_snapshot(
                    "ANSWER_GENERATION",
                    generation_result.model_call,
                )
                if generation_model_call is not None:
                    model_calls.append(generation_model_call)
                status_code, body = _final_generation_response(generation_result)
                terminal_reason = (
                    generation_result.withheld_reason.value
                    if generation_result.withheld_reason is not None
                    else generation_result.error_code
                )
                if generation_result.status == FinalAnswerStatus.COMPLETED:
                    failure_stage = None
                elif (
                    generation_result.withheld_reason
                    == FinalWithheldReason.UNVERIFIABLE_ANSWER
                ):
                    failure_stage = "CITATION_VALIDATION"
                elif generation_result.status == FinalAnswerStatus.ERROR:
                    failure_stage = "ANSWER_GENERATION"
                else:
                    failure_stage = "GENERATION_DECISION"

        selected_turns = (
            []
            if resolution is None
            else [
                turn.model_dump(mode="json", by_alias=True)
                for turn in resolution.selected_turns
            ]
        )
        resolved_query = None if resolution is None else resolution.resolved_query
        context_strategy = (
            "NEW_TOPIC"
            if resolution is not None
            and resolution.decision == QueryRewriteDecision.NEW_TOPIC
            else "FOLLOW_UP_WINDOW"
        )
        generation_snapshot = _generation_snapshot(generation_result)
        generation_snapshot["modelCalls"] = [
            call for call in model_calls if call["purpose"] == "ANSWER_GENERATION"
        ]
        stage_snapshot = {
            "status": body["status"],
            "indexVersionId": self._corpus_state.index_version_id,
            "turnNo": max(turn.turn_no for turn in candidates) + 1,
            "userQuery": question,
            "resolvedQuery": resolved_query,
            "contextStrategy": context_strategy,
            "contextTurnCount": len(selected_turns),
            "selectedTurnNos": [turn["turnNo"] for turn in selected_turns],
            "withheldReasonCode": (
                body.get("withheld", {}).get("reasonCode")
            ),
            "errorCode": body.get("error", {}).get("code"),
            "citationValidated": generation_snapshot["citationValidated"],
            "modelCalls": model_calls,
            "retrievalResultCount": len(search.fused_results),
            "citationCount": len(body["citations"]),
            "citationDocumentTitles": [
                citation["documentTitle"] for citation in body["citations"]
            ],
            "stageTrace": {
                "queryResolution": {
                    "originalQuestion": question,
                    "candidateTurns": [
                        turn.model_dump(mode="json", by_alias=True)
                        for turn in candidates
                    ],
                    "selectedTurns": selected_turns,
                    "contextStrategy": context_strategy,
                    "resolvedQuery": resolved_query,
                },
                "retrieval": {
                    "top5": _hybrid_top_5_snapshot(search.fused_results),
                },
                "generation": generation_snapshot,
                "terminal": {
                    "status": body["status"],
                    "failureStage": failure_stage,
                    "reason": terminal_reason,
                },
            },
            "snapshotSource": "IN_PROCESS_FIXED_CONTEXT",
        }
        failures = evaluate_turn(expected, status_code, body, stage_snapshot)
        return {
            "id": case["id"],
            "description": case.get("description"),
            "mode": case.get("mode", "FIXED_CONTEXT"),
            "passed": not failures,
            "conversationId": None,
            "fixedContext": case["fixedContext"],
            "turns": [
                {
                    "turnNo": 1,
                    "question": question,
                    "passed": not failures,
                    "failures": failures,
                    "httpStatus": status_code,
                    "response": body,
                    "db": stage_snapshot,
                }
            ],
        }


async def run_case_with_executor(
    execute_turn: TurnExecutor,
    case: Dict[str, Any],
) -> Dict[str, Any]:
    conversation_id = None
    turn_results = []

    for turn_no, expected in enumerate(case["turns"], start=1):
        try:
            status_code, body, generation_trace = await execute_turn(
                expected["question"],
                conversation_id,
            )
        except Exception as error:
            turn_results.append(
                {
                    "turnNo": turn_no,
                    "question": expected["question"],
                    "passed": False,
                    "failures": [f"API 실행 실패: {type(error).__name__}: {error}"],
                }
            )
            break

        returned_conversation_id = body.get("conversationId")
        if isinstance(returned_conversation_id, str):
            conversation_id = returned_conversation_id
        rag_run_id = body.get("ragRunId")
        db_snapshot = (
            await load_db_snapshot(rag_run_id, generation_trace)
            if isinstance(rag_run_id, str)
            else None
        )
        failures = evaluate_turn(
            expected,
            status_code,
            body,
            db_snapshot,
        )
        turn_results.append(
            {
                "turnNo": turn_no,
                "question": expected["question"],
                "passed": not failures,
                "failures": failures,
                "httpStatus": status_code,
                "response": body,
                "db": db_snapshot,
            }
        )

    return {
        "id": case["id"],
        "description": case.get("description"),
        "mode": case.get("mode", "CONTINUOUS_CONVERSATION"),
        "passed": (
            len(turn_results) == len(case["turns"])
            and all(turn["passed"] for turn in turn_results)
        ),
        "conversationId": conversation_id,
        "turns": turn_results,
    }


async def run_case(
    client: httpx.AsyncClient,
    endpoint: str,
    case: Dict[str, Any],
) -> Dict[str, Any]:
    async def execute_turn(
        question: str,
        conversation_id: Optional[str],
    ) -> Tuple[int, Dict[str, Any], Optional[GenerationStageTrace]]:
        payload = {"question": question}
        if conversation_id is not None:
            payload["conversationId"] = conversation_id
        response = await client.post(endpoint, json=payload)
        return response.status_code, response.json(), None

    return await run_case_with_executor(execute_turn, case)


async def _run_repeated_cases(
    cases: Sequence[Dict[str, Any]],
    repeat: int,
    case_runner: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    runs = []
    for repeat_no in range(1, repeat + 1):
        results = []
        run_executed_at = datetime.now(timezone.utc).isoformat()
        for case in cases:
            print(
                f"[repeat {repeat_no}/{repeat}] "
                f"[{case['id']}] {case.get('description', '')}",
                flush=True,
            )
            result = await case_runner(case)
            result["repeatNo"] = repeat_no
            results.append(result)
        runs.append(
            {
                "repeatNo": repeat_no,
                "executedAt": run_executed_at,
                "results": results,
            }
        )
    return runs


async def run_evaluation(
    *,
    execution_mode: str,
    base_url: str,
    timeout: float,
    cases_path: Path,
    output_path: Path,
    repository_revision: Optional[str],
    target_repository_revision: Optional[str],
    repeat: int,
    case_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    cases_payload = select_cases(load_cases(cases_path), case_ids)
    if execution_mode == "api":
        endpoint = f"{base_url.rstrip('/')}/api/chat"
        async with httpx.AsyncClient(
            timeout=timeout,
            headers={"Accept": "application/json"},
        ) as client:

            async def api_case_runner(case: Dict[str, Any]) -> Dict[str, Any]:
                if "fixedContext" in case:
                    raise ValueError(
                        "fixedContext case는 --execution-mode in-process가 필요합니다."
                    )
                return await run_case(client, endpoint, case)

            runs = await _run_repeated_cases(
                cases_payload["cases"],
                repeat,
                api_case_runner,
            )
    else:
        runtime = await InProcessEvaluationRuntime.create()

        async def in_process_case_runner(
            case: Dict[str, Any],
        ) -> Dict[str, Any]:
            if "fixedContext" in case:
                return await runtime.run_fixed_context_case(case)
            return await run_case_with_executor(runtime.execute_turn, case)

        runs = await _run_repeated_cases(
            cases_payload["cases"],
            repeat,
            in_process_case_runner,
        )

    first_index_id = next(
        (
            (turn.get("db") or {}).get("indexVersionId")
            for run in runs
            for case in run["results"]
            for turn in case["turns"]
            if turn.get("db") is not None
        ),
        None,
    )
    observed_model_calls = _collect_observed_model_calls(runs)
    payload = {
        "schemaVersion": "v2",
        "executedAt": datetime.now(timezone.utc).isoformat(),
        "casesVersion": cases_payload["version"],
        "evaluationMode": "CASE_DEFINED_WITH_NEW_CONVERSATION_PER_REPEAT",
        "executionMode": execution_mode,
        "evaluator": {
            "repositoryRevision": repository_revision,
            "workingTreeDirty": detect_worktree_dirty(),
            "importedModelContract": {
                "queryRewrite": OPENAI_QUERY_REWRITE_MODEL,
                "embedding": OPENAI_EMBEDDING_MODEL,
                "generation": OPENAI_GENERATION_MODEL,
            },
            "prompts": {
                "queryRewrite": {
                    "version": QUERY_REWRITE_PROMPT_VERSION,
                    "sha256": text_sha256(QUERY_REWRITE_PROMPT_V3),
                },
                "sourcePlanning": {
                    "version": SOURCE_PLANNING_PROMPT_VERSION,
                    "sha256": text_sha256(SOURCE_PLANNING_PROMPT_V10),
                },
                "sourcePlanningRegeneration": {
                    "version": SOURCE_PLANNING_REPAIR_PROMPT_VERSION,
                    "sha256": text_sha256(SOURCE_PLANNING_REPAIR_PROMPT_V10),
                },
                "answer": {
                    "version": ANSWER_PROMPT_VERSION,
                    "sha256": text_sha256(ANSWER_PROMPT_V17),
                },
                "answerValidationRegeneration": {
                    "version": ANSWER_REPAIR_PROMPT_VERSION,
                    "sha256": text_sha256(ANSWER_REPAIR_PROMPT_V17),
                },
            },
            "callSettings": {
                "queryRewrite": {
                    "maxOutputTokens": QUERY_REWRITE_MAX_OUTPUT_TOKENS,
                    "temperature": "OMITTED",
                    "reasoning": "OMITTED",
                },
                "sourcePlanning": {
                    "temperature": "OMITTED",
                    "reasoning": "OMITTED",
                },
                "sourcePlanRegeneration": {
                    "maxCount": MAX_SOURCE_PLANNING_REGENERATIONS,
                    "temperature": "OMITTED",
                    "reasoning": "OMITTED",
                },
                "answer": {
                    "temperature": "OMITTED",
                    "reasoning": "OMITTED",
                },
                "answerValidationRegeneration": {
                    "maxCount": 1,
                    "temperature": "OMITTED",
                    "reasoning": "OMITTED",
                },
            },
        },
        "executionTarget": {
            "apiBaseUrl": base_url if execution_mode == "api" else None,
            "repositoryRevision": (
                target_repository_revision
                if execution_mode == "api"
                else repository_revision
            ),
            "repositoryRevisionSource": (
                "IN_PROCESS_EVALUATOR"
                if execution_mode == "in-process"
                else ("CLI" if target_repository_revision else "UNKNOWN")
            ),
            "observedModelCalls": observed_model_calls,
        },
        "databaseSnapshot": {
            "source": "CONFIGURED_DATABASE",
            "indexVersion": await load_index_metadata(first_index_id),
        },
        "summary": summarize_runs(runs),
        "runs": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _collect_observed_model_calls(
    runs: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """실행 대상 DB에서 실제 관측한 모델 호출 식별자를 중복 없이 모은다."""

    observed = []
    identities = set()
    for run in runs:
        for result in run["results"]:
            for turn in result["turns"]:
                db_snapshot = turn.get("db") or {}
                for call in db_snapshot.get("modelCalls", []):
                    identity = (
                        call["purpose"],
                        call["provider"],
                        call["requestedModel"],
                        call["responseModel"],
                        call["promptVersion"],
                    )
                    if identity in identities:
                        continue
                    identities.add(identity)
                    observed.append(
                        {
                            "purpose": call["purpose"],
                            "provider": call["provider"],
                            "requestedModel": call["requestedModel"],
                            "responseModel": call["responseModel"],
                            "responseModelAvailability": call[
                                "responseModelAvailability"
                            ],
                            "promptVersion": call["promptVersion"],
                        }
                    )
    return observed


async def run_evaluation_and_dispose(**kwargs) -> Dict[str, Any]:
    """평가와 SQLAlchemy engine 정리를 같은 event loop에서 수행한다."""

    try:
        return await run_evaluation(**kwargs)
    finally:
        await dispose_engine()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout은 0보다 커야 합니다.")
    if args.repeat <= 0:
        raise SystemExit("--repeat은 1 이상이어야 합니다.")
    if args.recheck:
        recheck_output = unique_output_path(
            args.recheck_output or args.output,
            label="recheck",
        )
        payload = recheck_saved_results(
            select_cases(load_cases(args.cases), args.case_ids),
            json.loads(args.output.read_text(encoding="utf-8")),
        )
        recheck_output.parent.mkdir(parents=True, exist_ok=True)
        recheck_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        saved_output = recheck_output
    else:
        output_path = unique_output_path(args.output)
        payload = asyncio.run(
            run_evaluation_and_dispose(
                execution_mode=args.execution_mode,
                base_url=args.base_url,
                timeout=args.timeout,
                cases_path=args.cases,
                output_path=output_path,
                repository_revision=args.repository_revision,
                target_repository_revision=args.target_repository_revision,
                repeat=args.repeat,
                case_ids=args.case_ids,
            )
        )
        saved_output = output_path

    summary = payload["summary"]
    if "totalCaseExecutionCount" in summary:
        passed_count = summary["passedCaseExecutionCount"]
        total_count = summary["totalCaseExecutionCount"]
        failed_count = summary["failedCaseExecutionCount"]
    else:
        passed_count = summary["passedCaseCount"]
        total_count = summary["totalCaseCount"]
        failed_count = summary["failedCaseCount"]
    print(f"결과: {passed_count}/{total_count} 통과", flush=True)
    print(f"저장: {saved_output}", flush=True)
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
