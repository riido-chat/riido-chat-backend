"""챗봇 전체 동작을 가리키는 시스템 버전.

MAJOR.MINOR.PATCH 규칙을 따른다.

- MAJOR: 모델 교체(임베딩 포함) 또는 파이프라인 구조 변경(단계 추가나 제거 등)
- MINOR: 프롬프트 판, 문턱값, 검색 파라미터, 정본과 데이터 변경, 재색인
- PATCH: 응답 동작이 바뀌지 않는 수정

버전을 올릴 때마다 evaluation/SYSTEM_VERSIONS.md 에 같은 버전의 행을 추가한다.
CI(scripts/check_system_version.py)가 동작 관련 파일 변경 시 버전 갱신과 기록 행을 확인한다.
"""


SYSTEM_VERSION = "1.1.0"
