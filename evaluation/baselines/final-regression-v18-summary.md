# Generation v18 최종 회귀 평가

- 실행일: 2026-09-06
- 기준 HEAD: `4ec0c8a00423b51d0b624319caa314983d033301` + dirty working tree
- Query Rewrite: v3
- Source Planning: v10
- Answer: v17
- 논리 Generation: v18
- 모델: `gpt-5.4-mini`
- ACTIVE index: 1938 (`idx-20260905T092921909660Z-453fb7fa`)
- corpus: 문서 39개, Chunk 142개

## 결과

| 평가 | 결과 | 해석 |
| --- | ---: | --- |
| 로컬 전체 테스트 | 482 통과, 16 skip | 테스트 실패 없음 |
| 동일 질문 반복 10개 × 5회 | 40/50 | AC01·AC06만 각 0/5, 나머지 8개는 40/40 |
| Generation E2E 30문항 | 실행 30/30, Top-5 적중 30/30 | 실행·검색 회귀 없음. 내용 수용 점수는 아님 |
| CS 25개 | 22/25 | 실제 생성 오류 2건, 과거 기대 범위와 새 정의 정책 불일치 1건 |
| Multi-turn 18개 | 14/18 | 생성 오류 파급 1건, 보류 사유 2건, Rewrite 대상 오류 1건 |

## 실패 분류

### 1. Source Planning 구조 검증이 HTTP 500으로 종료됨

- CS07 `팀 대시보드는 왜 필요한 거야?`
- CS10 `구글 캘린더 연동은 왜 필요한 거야?`
- MT09 첫 질문 `슬랙 연동은 어떤 기능을 제공해?`
- 공통 원인: 모델이 `DEFINITION` 또는 `FEATURE_SUMMARY`의 `SUMMARY` 계획에서 둘 이상의
  EvidenceRequirement/Source를 반환했고, `GenerationSourcePlan`의 한 Source 검증이
  `ValidationError`를 발생시켰다. 현재 Planning에는 이 구조 오류를 교정하는 재생성 경로가
  없어 HTTP 500으로 끝난다.
- MT09 두 번째 질문의 문맥 실패는 첫 질문이 ERROR라 후속 문맥으로 선택되지 못한 파급이다.

### 2. 보류 상태는 안정적이지만 보류 사유가 기대와 다름

- AC06 5/5, MT14, MT15의 급여 계산 질문은 모두 WITHHELD였다.
- 기대는 `OUT_OF_SCOPE`, 실제는 `INSUFFICIENT_EVIDENCE`였다.
- 사용자에게 근거 없는 답을 제공하지 않는 동작은 유지됐지만 내부 사유 구분은 수정 대상이다.

### 3. Query Rewrite가 후속 대상을 잘못 선택함

- MT18 마지막 질문은 최근 구글 캘린더 문맥 대신 오래된 슬랙 문맥을 선택했다.
- 기대 resolved query: 구글 캘린더 연동의 작업 마감일 동기화 여부.
- 실제 resolved query: 슬랙 연동의 작업 마감일 동기화 여부.
- 결과는 WITHHELD였으며 Generation 유형은 GENERAL이었다. Generation보다 Rewrite 담당 문제다.

### 4. 현재 제품 정책과 과거 키워드 평가의 불일치

- AC01은 5회 모두 디스코드를 팀의 소통 도구/공간으로 정의하고 뤼이도 업데이트 전달과
  논의 자동 기록을 설명했으며 같은 `디스코드 > 개요`를 인용했다.
- 자동 실패는 `공간`을 `도구/서비스`로 허용하지 않거나, 의미상 관계를 설명해도 문자
  `연동/연결`이 없으면 실패시키는 과거 키워드 기준 때문이다. 합의한 핵심 사실·근거 기준으로는
  내용과 Source가 안정적이지만 평가 기준 갱신이 필요하다.
- CS23 `대기 작업이 뭐야?`는 정의와 시작점 역할을 설명했지만, 과거 기준이 백로그 이동과
  중요도 검토까지 요구해 실패했다. 직접 정의 질문에는 상세 절차를 붙이지 않는 현재 정책과
  충돌하므로 제품 실패로 단정하지 않는다.

## 결론과 다음 순서

1. 한 Source 정책을 없애기보다 Planning 구조 위반을 사용자 500으로 내보내지 않는 최소
   교정 방식을 먼저 정한다.
2. `OUT_OF_SCOPE`과 `INSUFFICIENT_EVIDENCE`의 의미 경계를 Planning에 명확히 한다.
3. MT18의 최근 명시 대상 선택은 Query Rewrite의 별도 국소 수정으로 다룬다.
4. AC01·CS23의 과거 키워드 기대값은 현재 합의한 의미 기반·질문 범위 기준으로 검토한다.

## Generation v19 후속 수정 결과

- Source Planning의 Pydantic 구조 검증 실패에만 교정 호출을 최대 한 번 추가했다.
- 정상 Planning에는 추가 호출하지 않으며 의미·질문 범위·Top-5를 그대로 유지한다.
- CS07·CS10 각 5회 재평가: **10/10 통과**, HTTP 500 0건.
- MT09 5회 재평가: **5/5 통과**, HTTP 500 0건.
- CS10 1회와 MT09 1회에서 구조 교정이 실제로 작동해 정상 답변으로 복구됐다.
- 두 번째 계획도 구조 검증에 실패하면 더 반복하지 않고 오류로 종료하는 단위 테스트를
  포함했다.
- 최신 전체 테스트: **485개 통과, 16개 조건부 skip**.
- 결과:
  - `evaluation/baselines/source-plan-repair-v19-cs-targeted.json`
  - `evaluation/baselines/source-plan-repair-v19-mt09-targeted.json`

## 원본 artifact

- `evaluation/baselines/answer-consistency-v18-final.json`
- `evaluation/baselines/generation-e2e-v18-final.json`
- `evaluation/baselines/chat-cs-v18-final.json`
- `evaluation/baselines/chat-multiturn-v18-final.json`
