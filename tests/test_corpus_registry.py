import unittest
from unittest.mock import patch

from app.retrieval.corpus_state import CorpusRegistry
from app.indexing.index_builder import _apply
from app.retrieval.models import RetrievalChunk


def _chunk(index_version_id: int, chunk_id: int) -> RetrievalChunk:
    return RetrievalChunk(
        document_id=f"document-{chunk_id}",
        section_id=f"section-{chunk_id}",
        document_title=f"문서 {chunk_id}",
        section_path=(f"문서 {chunk_id}",),
        source_url=f"https://docs.example/{chunk_id}",
        category="guide",
        content=f"본문 {chunk_id}",
        chunk_id=chunk_id,
        document_version_id=chunk_id,
        index_version_id=index_version_id,
    )


class CorpusRegistryTest(unittest.TestCase):
    def test_active_snapshots_are_isolated_by_document_group(self) -> None:
        registry = CorpusRegistry()

        with patch("app.retrieval.corpus_state.BM25Retriever") as bm25:
            group_one_retriever = object()
            group_two_retriever = object()
            bm25.side_effect = [group_one_retriever, group_two_retriever]
            registry.replace(10, [_chunk(101, 1)])
            registry.replace(20, [_chunk(202, 2)])

        first, first_index = registry.get_search_snapshot(10)
        second, second_index = registry.get_search_snapshot(20)
        self.assertIs(first, group_one_retriever)
        self.assertIs(second, group_two_retriever)
        self.assertEqual(101, first_index)
        self.assertEqual(202, second_index)

    def test_replacing_group_keeps_old_snapshot_tuple_stable_for_inflight_turn(self) -> None:
        registry = CorpusRegistry()

        with patch("app.retrieval.corpus_state.BM25Retriever") as bm25:
            old_retriever = object()
            new_retriever = object()
            bm25.side_effect = [old_retriever, new_retriever]
            registry.replace(10, [_chunk(101, 1)])
            old_snapshot = registry.get_search_snapshot(10)
            registry.replace(10, [_chunk(102, 2)])

        self.assertIs(old_retriever, old_snapshot[0])
        self.assertEqual(101, old_snapshot[1])
        current_snapshot = registry.get_search_snapshot(10)
        self.assertIs(new_retriever, current_snapshot[0])
        self.assertEqual(102, current_snapshot[1])

    def test_reindex_publishes_target_group_only_after_commit(self) -> None:
        import asyncio

        async def run() -> None:
            registry = CorpusRegistry()
            old_chunk = _chunk(801, 1)
            new_chunk = _chunk(701, 2)
            events = []

            class Writer:
                async def apply_index(self, _index_run_id):
                    events.append("apply")

                async def finish_apply_run(self, _index_run_id):
                    events.append("finish")

            class Session:
                async def commit(self):
                    events.append("commit")

                async def rollback(self):
                    events.append("rollback")

            class Reader:
                async def load_active_chunks(self):
                    events.append("load")
                    return [new_chunk]

            with patch("app.retrieval.corpus_state.BM25Retriever") as bm25:
                bm25.side_effect = [object(), object()]
                registry.replace(8, [old_chunk])
                original_publish = registry.publish

                def publish(group_id, prepared):
                    events.append("publish")
                    return original_publish(group_id, prepared)

                with patch.object(registry, "publish", side_effect=publish):
                    with patch(
                        "app.indexing.index_builder.SearchReader",
                        return_value=Reader(),
                    ):
                        await _apply(
                            Session(),
                            Writer(),
                            registry,
                            12,
                            document_group_id=7,
                        )

            self.assertLess(events.index("commit"), events.index("publish"))
            self.assertEqual(801, registry.get_search_snapshot(8)[1])
            self.assertEqual(701, registry.get_search_snapshot(7)[1])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
