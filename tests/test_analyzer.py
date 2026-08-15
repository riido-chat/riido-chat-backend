import unittest

from retrieval.analyzer import KiwiAnalyzer


class KiwiAnalyzerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analyzer = KiwiAnalyzer()

    def test_keeps_only_allowed_korean_pos(self) -> None:
        tokens = self.analyzer.tokenize("문서를 읽고 화면이 예쁘다.")

        self.assertEqual(["문서", "읽", "화면", "예쁘"], tokens)

    def test_removes_particles_and_endings(self) -> None:
        tokens = self.analyzer.tokenize("사용자가 캘린더를 연결합니다.")

        self.assertEqual(["사용자", "캘린더", "연결"], tokens)

    def test_keeps_foreign_words_and_numbers(self) -> None:
        tokens = self.analyzer.tokenize("Slack 123 연동")

        self.assertEqual(["Slack", "123", "연동"], tokens)

    def test_returns_empty_list_for_empty_input(self) -> None:
        self.assertEqual([], self.analyzer.tokenize(""))
        self.assertEqual([], self.analyzer.tokenize("   "))

    def test_uses_same_policy_for_same_text(self) -> None:
        text = "캘린더에서 일정을 찾는다."

        document_tokens = self.analyzer.tokenize(text)
        query_tokens = self.analyzer.tokenize(text)

        self.assertEqual(document_tokens, query_tokens)


if __name__ == "__main__":
    unittest.main()
