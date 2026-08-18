"""RetrievalChunk의 embedding 입력과 OpenAI embedding 생성을 관리한다."""

from typing import List, Optional

from openai import OpenAI

from app.core.config import get_settings
from retrieval.models import RetrievalChunk


OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_EMBEDDING_DIMENSIONS = 1536


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

        response = self._client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
            dimensions=OPENAI_EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )
        return list(response.data[0].embedding)
