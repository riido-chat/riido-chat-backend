import unittest

from pipeline.document.models import Chunk, NormalizedDocument
from retrieval.bm25_retriever import BM25Retriever
from retrieval.models import RetrievalChunk


class BM25RetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chunks = [
            cls._retrieval_chunk(
                index=0,
                document_title="캘린더 가이드",
                section_title="연동 설정",
                content="외부 일정 서비스에 연결합니다.",
            ),
            cls._retrieval_chunk(
                index=1,
                document_title="업무 관리",
                section_title="이슈 생성",
                content="새로운 업무 항목을 추가합니다.",
            ),
            cls._retrieval_chunk(
                index=2,
                document_title="회의 가이드",
                section_title="회의록 작성",
                content="회의 내용을 문서로 기록합니다.",
            ),
        ]
        cls.retriever = BM25Retriever(cls.chunks)

    def test_creates_index_once_and_reuses_it(self) -> None:
        index = self.retriever._index

        self.retriever.search("캘린더")
        self.retriever.search("회의")

        self.assertIs(index, self.retriever._index)
        self.assertEqual(len(self.chunks), self.retriever._index.corpus_size)

    def test_searches_document_title(self) -> None:
        results = self.retriever.search("캘린더")

        self.assertEqual("document-id-0:0", results[0].chunk.chunk_id)

    def test_searches_section_path(self) -> None:
        results = self.retriever.search("이슈 생성")

        self.assertEqual("document-id-1:0", results[0].chunk.chunk_id)

    def test_searches_content(self) -> None:
        results = self.retriever.search("기록")

        self.assertEqual("document-id-2:0", results[0].chunk.chunk_id)

    def test_builds_search_text_without_duplicate_document_title(self) -> None:
        chunk = self.chunks[0]

        search_text = self.retriever._create_search_text(chunk)

        self.assertEqual(1, search_text.count(chunk.document_title))
        self.assertIn(chunk.section_path[1], search_text)
        self.assertIn(chunk.content, search_text)


    def test_returns_top_ten_including_zero_scores(self) -> None:
        chunks = [
            self._retrieval_chunk(
                index=index,
                document_title=f"문서 {index}",
                section_title=f"섹션 {index}",
                content=f"본문 {index}",
            )
            for index in range(12)
        ]
        retriever = BM25Retriever(chunks)

        results = retriever.search("검색어없음")

        self.assertEqual(10, len(results))
        self.assertTrue(all(result.score == 0.0 for result in results))

    def test_returns_score_rank_and_metadata(self) -> None:
        results = self.retriever.search("캘린더")

        first = results[0]
        self.assertIsInstance(first.score, float)
        self.assertEqual(1, first.rank)
        self.assertEqual(self.chunks[0], first.chunk)
        self.assertEqual("https://docs.riido.io/document-0.md", first.chunk.source_url)
        self.assertEqual("test", first.chunk.category)

    def test_returns_empty_list_for_empty_query(self) -> None:
        self.assertEqual([], self.retriever.search(""))
        self.assertEqual([], self.retriever.search("   "))

    def test_uses_corpus_order_for_equal_scores(self) -> None:
        results = self.retriever.search("검색어없음")

        self.assertEqual(
            [chunk.chunk_id for chunk in self.chunks],
            [result.chunk.chunk_id for result in results],
        )
        self.assertEqual([1, 2, 3], [result.rank for result in results])

    def test_builds_retrieval_chunk_from_document_and_chunk(self) -> None:
        document = NormalizedDocument(
            document_id="document-id",
            title="문서 제목",
            source_url="https://docs.riido.io/test.md",
            category="test",
            content="문서 본문",
        )
        chunk = Chunk(
            chunk_id="chunk-id",
            document_id="document-id",
            section_id="section-id",
            section_path=("문서 제목", "섹션 제목"),
            content="Chunk 본문",
            sequence=0,
        )

        retrieval_chunk = RetrievalChunk.from_document_chunk(document, chunk)

        self.assertEqual("chunk-id", retrieval_chunk.chunk_id)
        self.assertEqual("document-id", retrieval_chunk.document_id)
        self.assertEqual("section-id", retrieval_chunk.section_id)
        self.assertEqual("문서 제목", retrieval_chunk.document_title)
        self.assertEqual(("문서 제목", "섹션 제목"), retrieval_chunk.section_path)
        self.assertEqual("https://docs.riido.io/test.md", retrieval_chunk.source_url)
        self.assertEqual("test", retrieval_chunk.category)
        self.assertEqual("Chunk 본문", retrieval_chunk.content)

    @staticmethod
    def _retrieval_chunk(
        index: int,
        document_title: str,
        section_title: str,
        content: str,
    ) -> RetrievalChunk:
        return RetrievalChunk(
            chunk_id=f"document-id-{index}:0",
            document_id=f"document-id-{index}",
            section_id=f"document-id-{index}:0",
            document_title=document_title,
            section_path=(document_title, section_title),
            source_url=f"https://docs.riido.io/document-{index}.md",
            category="test",
            content=content,
        )


if __name__ == "__main__":
    unittest.main()
