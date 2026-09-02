"""실행 중인 Chat API의 Multi-turn 응답과 DB 로그를 함께 평가한다."""

import argparse
import asyncio
import copy
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database.models import IndexVersion
from app.database.session import dispose_engine, get_session_factory
from app.rag.log_store import RagLogStore
from app.rag.query_rewrite import (
    OPENAI_QUERY_REWRITE_MODEL,
    QUERY_REWRITE_PROMPT_VERSION,
)
from generation.generator import GENERATION_PROMPT_VERSION, OPENAI_GENERATION_MODEL
from retrieval.embedding import OPENAI_EMBEDDING_MODEL


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluation/chat_multiturn_cases.json"
DEFAULT_CS_CASES_PATH = PROJECT_ROOT / "evaluation/chat_cs_cases.json"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat API와 DB 로그의 Multi-turn 수용 기준을 평가합니다."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
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
        help="결과에 기록할 commit SHA. 생략하면 현재 HEAD를 사용합니다.",
    )
    return parser


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


def recheck_saved_results(
    cases_payload: Dict[str, Any],
    saved_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """저장된 API·DB snapshot을 현재 평가 세트로 다시 판정한다."""

    result_by_id = {result["id"]: result for result in saved_payload["results"]}
    rechecked_results = []
    for case in cases_payload["cases"]:
        if case["id"] not in result_by_id:
            raise ValueError(f"저장된 결과에 case가 없습니다: {case['id']}")
        result = copy.deepcopy(result_by_id[case["id"]])
        if len(result["turns"]) != len(case["turns"]):
            raise ValueError(f"turn 수가 일치하지 않습니다: {case['id']}")

        for expected, turn in zip(case["turns"], result["turns"]):
            if turn.get("question") != expected["question"]:
                raise ValueError(
                    f"저장된 질문이 평가 세트와 다릅니다: {case['id']} "
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

    payload = copy.deepcopy(saved_payload)
    passed_count = sum(result["passed"] for result in rechecked_results)
    payload["recheckedAt"] = datetime.now(timezone.utc).isoformat()
    payload["casesVersion"] = cases_payload["version"]
    payload["summary"] = {
        "totalCaseCount": len(rechecked_results),
        "passedCaseCount": passed_count,
        "failedCaseCount": len(rechecked_results) - passed_count,
    }
    payload["results"] = rechecked_results
    return payload


async def load_db_snapshot(rag_run_id: str) -> Optional[Dict[str, Any]]:
    async with get_session_factory()() as session:
        detail = await RagLogStore(session).get_rag_run_detail(uuid.UUID(rag_run_id))
        if detail is None:
            return None
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
            "modelCalls": [
                {
                    "purpose": call.purpose.value,
                    "status": call.status.value,
                    "model": call.model_name,
                    "promptVersion": call.prompt_version,
                    "retryCount": call.retry_count,
                }
                for call in detail.model_calls
            ],
            "retrievalResultCount": len(detail.retrieval_results),
            "citationCount": len(detail.citations),
            "citationDocumentTitles": [
                citation.document_title_snapshot for citation in detail.citations
            ],
        }


async def load_index_metadata(index_version_id: Optional[int]) -> Optional[dict]:
    if index_version_id is None:
        return None
    async with get_session_factory()() as session:
        index = await session.get(IndexVersion, index_version_id)
        if index is None:
            return None
        return {
            "id": index.id,
            "version": index.version,
            "status": index.status.value,
            "createdAt": index.created_at.isoformat(),
            "activatedAt": (
                None if index.activated_at is None else index.activated_at.isoformat()
            ),
        }


async def run_case(
    client: httpx.AsyncClient,
    endpoint: str,
    case: Dict[str, Any],
) -> Dict[str, Any]:
    conversation_id = None
    turn_results = []

    for turn_no, expected in enumerate(case["turns"], start=1):
        payload = {"question": expected["question"]}
        if conversation_id is not None:
            payload["conversationId"] = conversation_id

        try:
            response = await client.post(endpoint, json=payload)
            body = response.json()
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
            await load_db_snapshot(rag_run_id)
            if isinstance(rag_run_id, str)
            else None
        )
        failures = evaluate_turn(
            expected,
            response.status_code,
            body,
            db_snapshot,
        )
        turn_results.append(
            {
                "turnNo": turn_no,
                "question": expected["question"],
                "passed": not failures,
                "failures": failures,
                "httpStatus": response.status_code,
                "response": body,
                "db": db_snapshot,
            }
        )

    return {
        "id": case["id"],
        "description": case.get("description"),
        "passed": (
            len(turn_results) == len(case["turns"])
            and all(turn["passed"] for turn in turn_results)
        ),
        "conversationId": conversation_id,
        "turns": turn_results,
    }


async def run_evaluation(
    *,
    base_url: str,
    timeout: float,
    cases_path: Path,
    output_path: Path,
    repository_revision: Optional[str],
    case_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    cases_payload = select_cases(load_cases(cases_path), case_ids)
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    async with httpx.AsyncClient(
        timeout=timeout,
        headers={"Accept": "application/json"},
    ) as client:
        results = []
        for case in cases_payload["cases"]:
            print(f"[{case['id']}] {case.get('description', '')}", flush=True)
            results.append(await run_case(client, endpoint, case))

    first_index_id = next(
        (
            turn.get("db", {}).get("indexVersionId")
            for case in results
            for turn in case["turns"]
            if turn.get("db") is not None
        ),
        None,
    )
    passed_count = sum(case["passed"] for case in results)
    payload = {
        "executedAt": datetime.now(timezone.utc).isoformat(),
        "repositoryRevision": repository_revision,
        "casesVersion": cases_payload["version"],
        "apiBaseUrl": base_url,
        "models": {
            "queryRewrite": OPENAI_QUERY_REWRITE_MODEL,
            "embedding": OPENAI_EMBEDDING_MODEL,
            "generation": OPENAI_GENERATION_MODEL,
        },
        "prompts": {
            "queryRewrite": QUERY_REWRITE_PROMPT_VERSION,
            "generation": GENERATION_PROMPT_VERSION,
        },
        "indexVersion": await load_index_metadata(first_index_id),
        "summary": {
            "totalCaseCount": len(results),
            "passedCaseCount": passed_count,
            "failedCaseCount": len(results) - passed_count,
        },
        "results": results,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


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
    if args.recheck:
        payload = recheck_saved_results(
            select_cases(load_cases(args.cases), args.case_ids),
            json.loads(args.output.read_text(encoding="utf-8")),
        )
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        payload = asyncio.run(
            run_evaluation_and_dispose(
                base_url=args.base_url,
                timeout=args.timeout,
                cases_path=args.cases,
                output_path=args.output,
                repository_revision=args.repository_revision,
                case_ids=args.case_ids,
            )
        )

    summary = payload["summary"]
    print(
        f"결과: {summary['passedCaseCount']}/{summary['totalCaseCount']} 통과",
        flush=True,
    )
    print(f"저장: {args.output}", flush=True)
    return 0 if summary["failedCaseCount"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
