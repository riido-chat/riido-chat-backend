import unittest

from app.document.models import NormalizedDocument
from app.document.section_parser import parse_sections
from app.question_grouping.document_candidates import (
    attach_outlines,
    build_document_outline,
    document_retrieval_settings,
    headings_from_markdown,
    headings_from_sections,
    parent_path_from_document_key,
    rank_documents,
    rank_documents_from_search,
)
from app.question_grouping.models import RankedDocument
from app.retrieval.hybrid_retriever import RRF_RANK_CONSTANT
from app.retrieval.models import HybridSearchCall, RetrievalChunk, RetrievalResult


def _result(chunk_id: int, version_id: int, rank: int) -> RetrievalResult:
    return RetrievalResult(
        chunk=RetrievalChunk(
            document_id=f"doc-{version_id}",
            section_id=f"doc-{version_id}:{chunk_id}",
            document_title=f"문서 {version_id}",
            section_path=(f"문서 {version_id}", f"절 {chunk_id}"),
            source_url=f"https://docs.riido.io/doc-{version_id}",
            category=None,
            content="본문",
            chunk_id=chunk_id,
            document_version_id=version_id,
            index_version_id=1,
        ),
        score=1.0,
        rank=rank,
    )


def _rrf(*ranks: int) -> float:
    return sum(1.0 / (RRF_RANK_CONSTANT + rank) for rank in ranks)


class RankDocumentsTest(unittest.TestCase):
    def test_scores_documents_by_best_fused_chunk(self) -> None:
        bm25 = [_result(1, 10, 1), _result(2, 20, 2), _result(3, 10, 3)]
        vector = [_result(3, 10, 1), _result(4, 30, 2)]

        ranked = rank_documents(bm25, vector)

        self.assertEqual([10, 20, 30], [item.document_version_id for item in ranked])
        self.assertAlmostEqual(_rrf(3, 1), ranked[0].score)
        self.assertEqual(3, ranked[0].best_chunk_id)
        self.assertAlmostEqual(_rrf(2), ranked[1].score)
        self.assertAlmostEqual(_rrf(2), ranked[2].score)
        self.assertEqual([1, 2, 3], [item.retrieval_rank for item in ranked])

    def test_ties_follow_first_chunk_position_by_chunk_id(self) -> None:
        # 같은 점수의 청크는 chunk_id 순으로 서고, 문서는 그 첫 위치로 순서를 정한다.
        bm25 = [_result(9, 90, 1)]
        vector = [_result(5, 50, 1)]

        ranked = rank_documents(bm25, vector)

        self.assertEqual([50, 90], [item.document_version_id for item in ranked])
        self.assertAlmostEqual(ranked[0].score, ranked[1].score)

    def test_uses_full_union_not_fused_top_five(self) -> None:
        bm25 = [_result(index, 100 + index, index) for index in range(1, 11)]
        vector = [_result(20 + index, 200 + index, index) for index in range(1, 11)]

        ranked = rank_documents(bm25, vector, top_k=20)

        self.assertEqual(20, len(ranked))
        self.assertEqual(5, len(rank_documents(bm25, vector)))

    def test_keeps_first_rank_for_duplicated_chunk(self) -> None:
        bm25 = [_result(1, 10, 1), _result(1, 10, 4)]

        ranked = rank_documents(bm25, [])

        self.assertAlmostEqual(_rrf(1), ranked[0].score)

    def test_rejects_chunks_without_database_ids_or_bad_top_k(self) -> None:
        missing = RetrievalResult(
            chunk=RetrievalChunk(
                document_id="doc",
                section_id="doc:1",
                document_title="문서",
                section_path=("문서",),
                source_url="https://docs.riido.io/doc",
                category=None,
                content="본문",
            ),
            score=1.0,
            rank=1,
        )

        with self.assertRaisesRegex(ValueError, "DB 식별자"):
            rank_documents([missing], [])
        with self.assertRaisesRegex(ValueError, "top_k"):
            rank_documents([], [], top_k=0)
        self.assertEqual((), rank_documents([], []))

    def test_rejects_same_chunk_with_different_versions(self) -> None:
        with self.assertRaisesRegex(ValueError, "문서 판"):
            rank_documents([_result(1, 10, 1)], [_result(1, 11, 1)])

    def test_ranks_from_hybrid_search_call(self) -> None:
        search = HybridSearchCall(
            bm25_results=(_result(1, 10, 1),),
            vector_results=(_result(2, 20, 1),),
        )

        ranked = rank_documents_from_search(search)

        self.assertEqual([10, 20], [item.document_version_id for item in ranked])

    def test_retrieval_settings_match_hybrid_search(self) -> None:
        self.assertEqual(
            {"dense": 10, "bm25": 10, "rrfK": 60},
            document_retrieval_settings(),
        )


MARKDOWN = """# 알림 설정

도입 문단

## 알림 끄기

본문

### 모바일 ###

```
## 코드 안의 제목
```

#### 깊은 제목

## 방해 금지 시간

### 요일별 설정
"""


class OutlineTest(unittest.TestCase):
    def test_parent_path_is_document_key_prefix(self) -> None:
        self.assertEqual("workspaces", parent_path_from_document_key("workspaces/plans"))
        self.assertEqual("a/b", parent_path_from_document_key("a/b/c"))
        self.assertEqual("", parent_path_from_document_key("introduction"))

    def test_headings_from_markdown_match_poc_rules(self) -> None:
        self.assertEqual(
            ("알림 끄기", "알림 끄기 > 모바일", "방해 금지 시간", "방해 금지 시간 > 요일별 설정"),
            headings_from_markdown(MARKDOWN),
        )
        self.assertEqual(("고아 H3", "절"), headings_from_markdown("### 고아 H3\n## 절\n"))

    def test_headings_from_stored_sections_match_markdown(self) -> None:
        document = NormalizedDocument(
            document_id="notifications/settings",
            title="알림 설정",
            source_url="https://docs.riido.io/notifications/settings",
            category=None,
            content=MARKDOWN,
            raw_content_uri=None,
            raw_content_hash="raw",
            normalized_content_hash="normalized",
        )
        sections = [
            (section.section_path, section.body) for section in parse_sections(document)
        ]

        self.assertEqual(headings_from_markdown(MARKDOWN), headings_from_sections(sections))

    def test_build_outline_and_attach_to_ranking(self) -> None:
        outline = build_document_outline(
            document_source_id=3,
            document_version_id=30,
            document_key="notifications/settings",
            title=" 알림 설정 ",
            headings=["알림 끄기"],
        )
        ranked = (
            RankedDocument(document_version_id=30, score=0.03, retrieval_rank=1, best_chunk_id=7),
        )

        candidates = attach_outlines(ranked, {30: outline})

        self.assertEqual("알림 설정", outline.title)
        self.assertEqual("notifications", outline.parent_path)
        self.assertEqual(("알림 끄기",), outline.headings)
        self.assertEqual(outline, candidates[0].outline)
        self.assertEqual((0.03, 1), (candidates[0].score, candidates[0].retrieval_rank))
        with self.assertRaisesRegex(ValueError, "outline"):
            attach_outlines(ranked, {})
        with self.assertRaisesRegex(ValueError, "제목"):
            build_document_outline(
                document_source_id=3,
                document_version_id=30,
                document_key="k",
                title="  ",
                headings=[],
            )


if __name__ == "__main__":
    unittest.main()
