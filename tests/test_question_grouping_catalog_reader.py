import unittest
from unittest.mock import AsyncMock

from app.question_grouping.catalog_reader import (
    CatalogDataError,
    applicability_rules_from_json,
)
from app.question_grouping.models import DocumentOutline
from app.question_grouping.outline_reader import (
    DocumentOutlineCache,
    DocumentOutlineReader,
    _section_path,
)


def _outline(version_id: int) -> DocumentOutline:
    return DocumentOutline(
        document_source_id=1,
        document_version_id=version_id,
        document_key="docs/doc",
        title="문서",
        parent_path="docs",
        headings=(),
    )


class ApplicabilityRulesTest(unittest.TestCase):
    def test_converts_rules_object_to_list(self) -> None:
        self.assertEqual((), applicability_rules_from_json(None))
        self.assertEqual((), applicability_rules_from_json({}))
        self.assertEqual((), applicability_rules_from_json({"rules": None}))
        self.assertEqual(
            ("관리자만", "유료 플랜"),
            applicability_rules_from_json({"rules": [" 관리자만 ", "", "유료 플랜"]}),
        )

    def test_rejects_other_shapes(self) -> None:
        for value in (["규칙"], {"rules": "규칙"}, {"rules": [1]}, {"rules": [], "x": 1}):
            with self.subTest(value=value), self.assertRaises(CatalogDataError):
                applicability_rules_from_json(value)


class OutlineCacheTest(unittest.TestCase):
    def test_evicts_least_recently_used(self) -> None:
        cache = DocumentOutlineCache(max_entries=2)
        cache.put((1, 9), _outline(1))
        cache.put((2, 9), _outline(2))
        self.assertIsNotNone(cache.get((1, 9)))

        cache.put((3, 9), _outline(3))

        self.assertIn((1, 9), cache)
        self.assertNotIn((2, 9), cache)
        self.assertIn((3, 9), cache)
        self.assertIsNone(cache.get((1, 8)))
        with self.assertRaises(ValueError):
            DocumentOutlineCache(max_entries=0)

    def test_section_path_prefers_metadata(self) -> None:
        self.assertEqual(
            ("문서", "절"), _section_path("다른 > 경로", {"section_path": ["문서", "절"]})
        )
        self.assertEqual(("문서", "절"), _section_path("문서 > 절", None))
        self.assertEqual((), _section_path(None, {"section_path": "문서"}))


class OutlineReaderCacheHitTest(unittest.IsolatedAsyncioTestCase):
    async def test_cached_versions_do_not_query(self) -> None:
        cache = DocumentOutlineCache()
        cache.put((10, 3), _outline(10))
        session = AsyncMock()
        reader = DocumentOutlineReader(session, cache=cache)

        outlines = await reader.load_outlines([10, 10], chunking_config_id=3)

        self.assertEqual({10: _outline(10)}, outlines)
        session.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
