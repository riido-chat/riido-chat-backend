import unittest

from app.question_grouping.canonical_validation import (
    ERROR_CITATION_NOT_REFERENCED,
    ERROR_DUPLICATE_CITATION_ORDER,
    ERROR_MARKER_WITHOUT_CITATION,
    ERROR_NO_CITATION_MARKERS,
    ERROR_NO_CITATIONS,
    check_canonical_content,
    check_canonical_citations,
    extract_citation_numbers,
)


class ExtractCitationNumbersTest(unittest.TestCase):
    def test_extracts_bracket_numbers_as_a_set(self) -> None:
        self.assertEqual(
            frozenset({1, 2}),
            extract_citation_numbers("설정에서 끕니다 [1]. 모바일은 따로 끕니다 [2][1]."),
        )

    def test_ignores_code_regions(self) -> None:
        markdown = "\n".join(
            [
                "배열은 `items[3]` 로 읽습니다 [1].",
                "```python",
                "value = data[2]",
                "```",
                "~~~",
                "[4]",
                "~~~",
            ]
        )

        self.assertEqual(frozenset({1}), extract_citation_numbers(markdown))

    def test_ignores_non_citation_brackets(self) -> None:
        markdown = "[0] 과 [01] 과 [링크](https://x) 와 [1](https://x) 와 [ 2 ]"

        self.assertEqual(frozenset(), extract_citation_numbers(markdown))


class CheckCanonicalCitationsTest(unittest.TestCase):
    def test_matching_sets_pass(self) -> None:
        result = check_canonical_citations("A [1]. B [2]. A 다시 [1].", [2, 1])

        self.assertTrue(result.valid)
        self.assertEqual(frozenset({1, 2}), result.body_numbers)
        self.assertEqual(frozenset({1, 2}), result.citation_orders)

    def test_requires_at_least_one_citation(self) -> None:
        result = check_canonical_citations("근거 없는 본문", [])

        self.assertFalse(result.valid)
        self.assertEqual((ERROR_NO_CITATIONS, ERROR_NO_CITATION_MARKERS), result.errors)

    def test_marker_without_citation_row(self) -> None:
        result = check_canonical_citations("A [1]. B [3].", [1])

        self.assertEqual((ERROR_MARKER_WITHOUT_CITATION,), result.errors)

    def test_citation_row_not_referenced_in_body(self) -> None:
        result = check_canonical_citations("A [1].", [1, 2])

        self.assertEqual((ERROR_CITATION_NOT_REFERENCED,), result.errors)

    def test_marker_only_inside_code_counts_as_missing(self) -> None:
        result = check_canonical_citations("`[1]` 만 있다", [1])

        self.assertEqual(
            (ERROR_NO_CITATION_MARKERS, ERROR_CITATION_NOT_REFERENCED),
            result.errors,
        )

    def test_duplicate_citation_orders(self) -> None:
        result = check_canonical_citations("A [1].", [1, 1])

        self.assertEqual((ERROR_DUPLICATE_CITATION_ORDER,), result.errors)


class CheckCanonicalContentTest(unittest.TestCase):
    def test_rejects_bengali_letters_mixed_into_korean_body(self) -> None:
        reason = check_canonical_content("설정에서 알림을 끄고 তারপর 다시 저장합니다.")

        self.assertEqual(
            "정본 본문에 한국어 맥락의 벵골 문자가 포함되어 있습니다.", reason
        )

    def test_rejects_georgian_letters_mixed_into_korean_body(self) -> None:
        reason = check_canonical_content("설정에서 알림을 끄고 შემდეგ 다시 저장합니다.")

        self.assertEqual(
            "정본 본문에 한국어 맥락의 조지아 문자가 포함되어 있습니다.", reason
        )

    def test_ignores_suspect_letters_inside_code_regions(self) -> None:
        markdown = "설정에서 확인합니다. `তারপর`\n\n```text\nშემდეგ\n```"

        self.assertIsNone(check_canonical_content(markdown))

    def test_rejects_short_korean_context(self) -> None:
        self.assertEqual(
            "정본 본문에 한국어 맥락의 벵골 문자가 포함되어 있습니다.",
            check_canonical_content("확인 তারপর 저장 [1]."),
        )
        self.assertEqual(
            "정본 본문에 한국어 맥락의 조지아 문자가 포함되어 있습니다.",
            check_canonical_content("확인 შემდეგ 저장 [1]."),
        )

    def test_preserves_normal_korean_english_emoji_latin_and_math(self) -> None:
        markdown = "설정(Settings)을 엽니다 😊 café; 계산은 x ≤ 3, ∑x 입니다."

        self.assertIsNone(check_canonical_content(markdown))

    def test_existing_link_error_keeps_priority_over_script_error(self) -> None:
        markdown = "[설정](https://example.com)에서 확인하고 তারপর 저장합니다."

        self.assertEqual(
            "답변 본문에 링크 또는 HTML이 포함됐습니다.",
            check_canonical_content(markdown),
        )

    def test_preserves_body_written_only_in_another_language(self) -> None:
        self.assertIsNone(check_canonical_content("This is a normal English answer."))
        self.assertIsNone(check_canonical_content("বাংলায় লেখা একটি স্বাভাবিক উত্তর।"))
        self.assertIsNone(check_canonical_content("ეს არის ჩვეულებრივი პასუხი."))


if __name__ == "__main__":
    unittest.main()
