import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.retrieval.models import RetrievalChunk
from evaluation.run_embedding_model_comparison import (
    CANDIDATE_CONFIGS,
    CandidateConfig,
    ExperimentEmbedder,
    attach_experiment_ids,
    calculate_cost_usd,
    calculate_raw_vector_storage_bytes,
    cosine_similarity,
    rank_vector_results,
)


def chunk(section_id: str) -> RetrievalChunk:
    return RetrievalChunk(
        document_id=f"document-{section_id}",
        section_id=section_id,
        document_title=f"문서 {section_id}",
        section_path=(f"문서 {section_id}", f"섹션 {section_id}"),
        source_url=f"https://example.com/{section_id}",
        category="guide",
        content=f"본문 {section_id}",
    )


class EmbeddingModelComparisonTest(unittest.TestCase):
    def test_defines_agreed_candidate_matrix(self) -> None:
        self.assertEqual(
            [
                ("A", "text-embedding-3-large", 1536),
                ("B", "text-embedding-3-small", 1536),
                ("C", "text-embedding-3-large", 3072),
                ("D", "text-embedding-3-large", 768),
                ("E", "text-embedding-3-small", 768),
            ],
            [(item.id, item.model, item.dimensions) for item in CANDIDATE_CONFIGS],
        )

    def test_attaches_stable_ids_for_existing_rrf(self) -> None:
        chunks = attach_experiment_ids([chunk("a"), chunk("b")])

        self.assertEqual([1, 2], [item.chunk_id for item in chunks])
        self.assertEqual([1, 1], [item.index_version_id for item in chunks])
        self.assertEqual(["a", "b"], [item.section_id for item in chunks])

    def test_calculates_cosine_similarity(self) -> None:
        self.assertAlmostEqual(1.0, cosine_similarity([1.0, 0.0], [2.0, 0.0]))
        self.assertAlmostEqual(0.0, cosine_similarity([1.0, 0.0], [0.0, 1.0]))

    def test_rejects_invalid_cosine_inputs(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])

    def test_ranks_vector_results_with_stable_tie_break(self) -> None:
        chunks = attach_experiment_ids([chunk("a"), chunk("b"), chunk("c")])
        results = rank_vector_results(
            chunks,
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [1.0, 0.0],
            top_k=3,
        )

        self.assertEqual(["a", "c", "b"], [item.chunk.section_id for item in results])
        self.assertEqual([1, 2, 3], [item.rank for item in results])

    def test_calculates_cost_and_raw_storage(self) -> None:
        self.assertAlmostEqual(0.013, calculate_cost_usd(100_000, 0.13))
        self.assertEqual(
            142 * 1536 * 4,
            calculate_raw_vector_storage_bytes(142, 1536),
        )

    def test_rejects_invalid_cost_and_storage_inputs(self) -> None:
        with self.assertRaises(ValueError):
            calculate_cost_usd(-1, 0.13)
        with self.assertRaises(ValueError):
            calculate_raw_vector_storage_bytes(1, 0)

    def test_embedder_passes_model_and_dimensions_and_aggregates_usage(self) -> None:
        client = Mock()
        client.embeddings.create.side_effect = [
            SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])],
                usage=SimpleNamespace(prompt_tokens=3),
            ),
            SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[0.0, 1.0])],
                usage=SimpleNamespace(prompt_tokens=4),
            ),
        ]
        config = CandidateConfig("X", "embedding-test", 2, 0.1)

        result = ExperimentEmbedder(client).embed_many(
            ["첫 번째", "두 번째"],
            config,
            batch_size=1,
        )

        self.assertEqual(((1.0, 0.0), (0.0, 1.0)), result.embeddings)
        self.assertEqual(7, result.input_tokens)
        self.assertEqual(2, result.request_count)
        self.assertEqual(
            [
                unittest.mock.call(
                    model="embedding-test",
                    input=["첫 번째"],
                    dimensions=2,
                    encoding_format="float",
                ),
                unittest.mock.call(
                    model="embedding-test",
                    input=["두 번째"],
                    dimensions=2,
                    encoding_format="float",
                ),
            ],
            client.embeddings.create.call_args_list,
        )


if __name__ == "__main__":
    unittest.main()
