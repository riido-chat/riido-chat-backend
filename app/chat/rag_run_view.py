"""저장된 턴 실행 결과를 결과 조회(polling) 응답으로 옮긴다.

스트림이 끊긴 클라이언트의 회수 경로이므로 파이프라인을 다시 실행하지 않고
이미 마감된 기록만 읽어 외부 표현으로 변환한다. 검색 후보, 모델 호출, 내부
error_code, 피드백은 어느 분기에서도 응답에 싣지 않는다.
"""

import logging
import uuid
from typing import Optional, Sequence, Tuple

from pydantic import ValidationError

from app.chat.schema import (
    ChatAnswer,
    ChatCitation,
    ChatCompletedResponse,
    ChatResponseStatus,
    ChatWithheld,
    ChatWithheldReasonCode,
    ChatWithheldResponse,
)
from app.chat.rag_run_schema import (
    RagRunProcessingResponse,
    RagRunProcessingStatus,
    RagRunResponse,
)
from app.database.models import AnswerCitation, AnswerStatus, RagRun

# 동기 응답과 같은 규칙을 쓰려고 원본 헬퍼를 그대로 참조한다. 복제하면 한쪽만
# 고쳐질 위험이 있어 비공개 심볼이라도 재사용한다. public alias 정리는 별도 건이다.
from app.chat.service import (
    _internal_error_response,
    _to_response_section_path,
    _to_response_source_url,
)
from app.answering.service import WITHHELD_RESPONSES
from app.chat.log_store import RagLogStore, RagRunDetail
from app.answering.models import (
    Citation,
    CitationSourceKind,
    FinalWithheldReason,
)


logger = logging.getLogger(__name__)

# answer_citations.node_path_snapshot이 이 구분자로 이어붙인 단일 문자열이라
# 같은 구분자로 되돌린다. 섹션 제목 자체에 " > "가 들어가면 분해가 어긋나는데,
# 스냅샷이 평탄화 저장이라 복원 단계에서는 막을 수 없는 한계다.
SECTION_PATH_SEPARATOR = " > "


class RagRunResultNotFoundError(LookupError):
    """조회할 수 없는 ragRunId로 결과를 요청했을 때 발생한다."""


async def get_rag_run_response(
    log_store: RagLogStore,
    rag_run_id: uuid.UUID,
) -> RagRunResponse:
    """저장된 턴을 조회해 상태에 맞는 응답으로 변환한다."""

    detail = await log_store.get_rag_run_detail(rag_run_id)
    if detail is None:
        raise RagRunResultNotFoundError(f"존재하지 않는 턴입니다: {rag_run_id}")

    return to_rag_run_response(detail)


def to_rag_run_response(detail: RagRunDetail) -> RagRunResponse:
    """조회 결과를 외부 응답 DTO로 변환한다."""

    run = detail.run

    if run.status == AnswerStatus.PROCESSING:
        return RagRunProcessingResponse(
            status=RagRunProcessingStatus.PROCESSING,
            conversation_id=run.conversation_id,
            rag_run_id=run.id,
        )

    if run.status == AnswerStatus.COMPLETED:
        return _completed_response(run, detail.citations)

    if run.status == AnswerStatus.WITHHELD:
        return _withheld_response(run)

    if run.status == AnswerStatus.ERROR:
        return _internal_error_response(run.conversation_id, run.id)

    logger.warning(
        "외부 표현이 없는 답변 상태입니다: rag_run_id=%s, status=%s",
        run.id,
        run.status,
    )
    return _internal_error_response(run.conversation_id, run.id)


def _completed_response(
    run: RagRun,
    citation_rows: Sequence[AnswerCitation],
) -> RagRunResponse:
    if not run.answer_content:
        logger.warning(
            "COMPLETED인데 답변 본문이 없어 오류로 응답합니다: rag_run_id=%s",
            run.id,
        )
        return _internal_error_response(run.conversation_id, run.id)

    try:
        return ChatCompletedResponse(
            status=ChatResponseStatus.COMPLETED,
            conversation_id=run.conversation_id,
            rag_run_id=run.id,
            answer=ChatAnswer(answer_markdown=run.answer_content),
            citations=[
                _to_chat_citation(row, run.id) for row in citation_rows
            ],
        )
    except ValidationError:
        # 인용 개수·번호가 응답 계약을 벗어난 기록이다. 500 대신 오류 응답으로 내린다.
        logger.warning(
            "COMPLETED 응답 계약을 만족하지 못했습니다: rag_run_id=%s, citations=%d",
            run.id,
            len(citation_rows),
        )
        return _internal_error_response(run.conversation_id, run.id)


def _withheld_response(run: RagRun) -> RagRunResponse:
    try:
        reason = FinalWithheldReason(run.withheld_reason_code)
        message = WITHHELD_RESPONSES[reason]
        reason_code = ChatWithheldReasonCode(reason.value)
    except (ValueError, KeyError):
        logger.warning(
            "외부 표현이 없는 보류 사유입니다: rag_run_id=%s, reason_code=%s",
            run.id,
            run.withheld_reason_code,
        )
        return _internal_error_response(run.conversation_id, run.id)

    return ChatWithheldResponse(
        status=ChatResponseStatus.WITHHELD,
        conversation_id=run.conversation_id,
        rag_run_id=run.id,
        answer=None,
        withheld=ChatWithheld(reason_code=reason_code, message=message),
        citations=[],
    )


def _to_chat_citation(
    row: AnswerCitation,
    rag_run_id: uuid.UUID,
) -> ChatCitation:
    document_title = row.document_title_snapshot
    source_url = row.source_uri_snapshot
    if document_title is None or source_url is None:
        # 인용 행을 빼면 citationNumber 연속성이 깨지므로 빈 문자열로 채우고 남긴다.
        logger.warning(
            "인용 스냅샷이 비어 있어 빈 문자열로 대체합니다: "
            "rag_run_id=%s, citation_order=%s",
            rag_run_id,
            row.citation_order,
        )

    # 문서 제목 접두 제거 규칙을 그대로 쓰려고 스냅샷에서 Citation을 복원한다.
    citation = Citation(
        citation_number=row.citation_order,
        document_title=document_title or "",
        section_path=_split_section_path(row.node_path_snapshot),
        source_url=source_url or "",
        source_kind=CitationSourceKind.from_canonical_uri(source_url or ""),
    )
    return ChatCitation(
        citation_number=citation.citation_number,
        document_title=citation.document_title,
        section_path=_to_response_section_path(citation),
        source_url=_to_response_source_url(citation),
        source_kind=citation.source_kind,
    )


def _split_section_path(node_path: Optional[str]) -> Tuple[str, ...]:
    if not node_path:
        return ()
    return tuple(node_path.split(SECTION_PATH_SEPARATOR))
