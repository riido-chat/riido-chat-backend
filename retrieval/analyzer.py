"""Kiwi를 사용해 BM25 검색용 토큰을 생성한다."""

from typing import FrozenSet, List

from kiwipiepy import Kiwi


ALLOWED_POS: FrozenSet[str] = frozenset({"NNG", "NNP", "VV", "VA", "SL", "SN"})


class KiwiAnalyzer:
    """문서와 Query에 동일한 형태소 분석 정책을 적용한다."""

    def __init__(self) -> None:
        self._kiwi = Kiwi()

    def tokenize(self, text: str) -> List[str]:
        """허용된 품사의 형태소만 검색 토큰으로 반환한다."""

        return [
            token.form
            for token in self._kiwi.tokenize(text)
            if token.tag in ALLOWED_POS
        ]
