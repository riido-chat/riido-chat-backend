"""RetrievalChunk의 embedding 입력과 OpenAI embedding 생성을 관리한다."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from openai import OpenAI

from app.core.config import get_settings
from retrieval.models import RetrievalChunk


OPENAI_EMBEDDING_PROVIDER = "openai"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_EMBEDDING_DIMENSIONS = 1536


@dataclass(frozen=True)
class EmbeddingResponse:
    """embedding 결과와 model_calls에 남길 사용량."""

    embeddings: List[List[float]]
    input_tokens: Optional[int] = None


def build_embedding_text(chunk: RetrievalChunk) -> str:
    """Chunk의 의미 정보만 newline으로 결합한다."""

    parts = [chunk.document_title, *chunk.section_path[1:], chunk.content]
    return "\n".join(part for part in parts if part)


class OpenAIEmbedder:
    """OpenAI API를 호출해 text를 고정 차원의 vector로 변환한다."""

    def __init__(self, client: Optional[OpenAI] = None) -> None:
        if client is None:
            api_key = get_settings().openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")
            client = OpenAI(api_key=api_key)

        self._client = client

    def embed(self, text: str) -> List[float]:
        """확정된 모델과 차원으로 text embedding을 생성한다."""

        return self.embed_many([text])[0]

    def embed_many(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:
        """여러 text의 embedding을 한 번의 요청으로 생성한다."""

        return self.embed_many_with_usage(texts).embeddings

    def embed_many_with_usage(
        self,
        texts: Sequence[str],
    ) -> EmbeddingResponse:
        """embedding과 함께 호출 사용량을 반환한다 (model_calls 기록용)."""

        input_texts = list(texts)
        if not input_texts:
            raise ValueError("Embedding할 text가 하나 이상이어야 합니다.")

        response = self._client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=input_texts,
            dimensions=OPENAI_EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )

        if len(response.data) != len(input_texts):
            raise RuntimeError(
                "OpenAI embedding 응답 개수가 입력과 일치하지 않습니다: "
                f"입력 {len(input_texts)}개, 응답 {len(response.data)}개"
            )

        ordered_data = sorted(response.data, key=lambda item: item.index)
        response_indexes = [item.index for item in ordered_data]
        if response_indexes != list(range(len(input_texts))):
            raise RuntimeError(
                "OpenAI embedding 응답 index가 입력 순서와 일치하지 않습니다."
            )

        embeddings = [list(item.embedding) for item in ordered_data]
        for embedding in embeddings:
            if len(embedding) != OPENAI_EMBEDDING_DIMENSIONS:
                raise ValueError(
                    "OpenAI embedding은 "
                    f"{OPENAI_EMBEDDING_DIMENSIONS}차원이어야 합니다."
                )

        usage = getattr(response, "usage", None)
        return EmbeddingResponse(
            embeddings=embeddings,
            input_tokens=getattr(usage, "prompt_tokens", None),
        )
