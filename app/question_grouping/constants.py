"""질문 판별과 정본 캐시 게이트의 코드 상수.

판별 모델과 프롬프트 판은 프로필이 아니라 코드 상수다. 어떤 값으로 판별했는지는
classification_runs.model / prompt_version 이 남긴다.
"""


JUDGE_PROVIDER = "openai"
JUDGE_MODEL = "gpt-5.6-luna"
JUDGE_REASONING_EFFORT = "low"
JUDGE_PROMPT_VERSION = "question-grouping-v7-2"
JUDGE_OUTPUT_SCHEMA_NAME = "question_grouping_decision_v7_2"
# 추론 토큰도 출력 한도를 쓰므로 PoC 에서 빈 응답이 나지 않았던 값을 유지한다.
JUDGE_MAX_OUTPUT_TOKENS = 1024
# v7.2 측정 p95 는 약 7.4초다. 판별 지연 상한은 timeout × 시도 횟수다.
JUDGE_TIMEOUT_SECONDS = 20.0
# 첫 시도 1회 + 일시 오류일 때만 재시도 1회.
JUDGE_MAX_ATTEMPTS = 2
JUDGE_RETRY_DELAY_SECONDS = 0.5

SUBPROBLEM_CANDIDATE_TOP_K = 5
DOCUMENT_CANDIDATE_TOP_K = 5

# 세부 문제 포함 기준 임베딩 문장 구성(이름 + 포함 기준)의 판.
SUBPROBLEM_EMBEDDING_TEXT_VERSION = "name-inclusion-v1"

# 가이드 밖(NO_DOCUMENT) 문제 그룹은 문서 키가 없어 payload group.id 로 이 값을 쓴다.
# 문서 키는 GitBook 경로("폴더/문서")라 슬래시 없는 대문자 값과 겹치지 않는다.
NO_DOCUMENT_GROUP_KEY = "NO_DOCUMENT"
NO_DOCUMENT_GROUP_NAME = "가이드 밖 질문"

# (document_version_id, chunking_config_id) 별 문서 outline LRU 캐시 크기.
DOCUMENT_OUTLINE_CACHE_SIZE = 512
# v2: 다른 표에 정본이 있는 사용량, 모델, 프롬프트 판, 추론 설정, 색인 판, 문서 그룹, 분류 실행 칸을 뺐다.
JUDGMENT_INPUT_SCHEMA_VERSION = "v2"

# payload 식별자와 섞기 스트림
DOCUMENT_ID_PREFIX = "D"
HEADING_SEPARATOR = " > "
SUBPROBLEM_SHUFFLE_STREAM = "subproblems"
DOCUMENT_SHUFFLE_STREAM = "documents"

# 판별 응답 무효 사유(R7). 이 셋 외의 형식 위반은 보정한다.
INVALID_EMPTY_OUTPUT = "EMPTY_OUTPUT"
INVALID_UNPARSEABLE_OUTPUT = "UNPARSEABLE_OUTPUT"
INVALID_UNKNOWN_SUBPROBLEM_KEY = "UNKNOWN_SUBPROBLEM_KEY"

# 캐시 게이트 거부 사유(question_cache_attempts.rejection_reasons, 50자 이하)
REJECT_CLASSIFICATION_NOT_CONNECTED = "CLASSIFICATION_NOT_CONNECTED"
REJECT_SUBPROBLEM_NOT_APPROVED = "SUBPROBLEM_NOT_APPROVED"
REJECT_CANONICAL_ANSWER_NOT_FOUND = "CANONICAL_ANSWER_NOT_FOUND"
REJECT_SUBPROBLEM_VERSION_MISMATCH = "SUBPROBLEM_VERSION_MISMATCH"
# 판별에 제시한 정본과 게이트가 다시 읽은 승인 정본이 다를 때.
REJECT_CANONICAL_ANSWER_CHANGED = "CANONICAL_ANSWER_CHANGED"
# 세부 문제가 속한 문서가 꺼져 있을 때(document_sources.enabled = false).
REJECT_SUBPROBLEM_DOCUMENT_DISABLED = "SUBPROBLEM_DOCUMENT_DISABLED"
REJECT_CANONICAL_CITATION_MISSING = "CANONICAL_CITATION_MISSING"
REJECT_CITED_DOCUMENT_NOT_INDEXED = "CITED_DOCUMENT_NOT_INDEXED"
REJECT_CITED_DOCUMENT_DISABLED = "CITED_DOCUMENT_DISABLED"
REJECT_CITED_SECTION_CHANGED = "CITED_SECTION_CHANGED"
REJECT_SUBPROBLEM_UNUSED = "SUBPROBLEM_UNUSED"
REJECT_SUBPROBLEM_STOPPED = "SUBPROBLEM_STOPPED"
REJECT_CANONICAL_SERVE_FAILED = "CANONICAL_SERVE_FAILED"
# 게이트 재조회에서 정본·인용 데이터가 시드 규칙과 맞지 않을 때(CatalogDataError). 판별은 유효하다.
REJECT_CANONICAL_DATA_INVALID = "CANONICAL_DATA_INVALID"

# 정확 일치 원천 분류를 재사용하지 않는 사유. 이때는 캐시 시도 행을 쓰지 않고 일치 없음처럼
# 일반 흐름으로 넘기므로, 같은 턴 판별 행의 judgment_input.exactQuestionFallthrough 에 남긴다.
# 원천 분류가 본 세부 문제 개정이 현재 개정과 다를 때.
EXACT_SOURCE_SUBPROBLEM_VERSION_STALE = "EXACT_SOURCE_SUBPROBLEM_VERSION_STALE"
# 원천 분류가 본 정본이 현재 승인 정본과 다를 때.
EXACT_SOURCE_CANONICAL_CHANGED = "EXACT_SOURCE_CANONICAL_CHANGED"
# 원천 분류가 본 정본을 알 수 없는데 현재 승인 정본에 적용 제외 규칙이 있을 때.
EXACT_SOURCE_CANONICAL_UNVERIFIED = "EXACT_SOURCE_CANONICAL_UNVERIFIED"
