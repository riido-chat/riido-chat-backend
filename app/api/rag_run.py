"""RagRun 결과 조회(polling) endpoint를 제공한다."""

import uuid

from fastapi import APIRouter, Depends, status

from app.api.rag_run_schema import (
    RagRunErrorCode,
    RagRunErrorResponse,
    RagRunResponse,
)
from app.rag.dependencies import get_rag_log_store
from app.rag.log_store import RagLogStore
from app.rag.rag_run_view import get_rag_run_response


router = APIRouter(tags=["rag-run"])

RAG_RUN_PATH = "/api/chat/{rag_run_id}"
RAG_RUN_NOT_FOUND_MESSAGE = "존재하지 않는 답변입니다."

RAG_RUN_ERROR_RESPONSES = {
    status.HTTP_404_NOT_FOUND: {
        "model": RagRunErrorResponse,
        "description": "존재하지 않는 ragRunId",
    },
}


def rag_run_result_not_found_response() -> RagRunErrorResponse:
    """존재하지 않는 ragRunId로 결과를 조회했을 때의 응답을 만든다."""

    return RagRunErrorResponse(
        code=RagRunErrorCode.NOT_FOUND,
        message=RAG_RUN_NOT_FOUND_MESSAGE,
    )


@router.get(
    RAG_RUN_PATH,
    response_model=RagRunResponse,
    status_code=status.HTTP_200_OK,
    responses=RAG_RUN_ERROR_RESPONSES,
    summary="답변 결과 조회",
)
async def get_rag_run(
    rag_run_id: uuid.UUID,
    log_store: RagLogStore = Depends(get_rag_log_store),
) -> RagRunResponse:
    """저장된 턴 결과를 반환한다.

    조회만 하므로 commit하지 않고, 마감된 결과는 몇 번을 호출해도 같다.
    """

    return await get_rag_run_response(log_store, rag_run_id)
