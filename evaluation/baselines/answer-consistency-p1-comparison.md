# P0/P1 답변 일관성 비교

실행일: 2026-09-05  
평가 세트: `evaluation/answer_consistency_cases.json` (`answer-consistency-v1`)  
실행 방식: in-process, 10개 케이스 × 5회

## 결과

| 구분 | 수정 전 | P1 수정 후 |
| --- | ---: | ---: |
| 통과 | 37/50 | 36/50 |
| NEW_CONVERSATION | 21/30 | 21/30 |
| CONTINUOUS_CONVERSATION | 6/10 | 5/10 |
| FIXED_CONTEXT | 10/10 | 10/10 |
| 검증 재생성 | 기록 전 | 0회 |
| API 오류 재시도 | 0회 | 0회 |
| 전체 ModelCall | 120회 | 120회 |
| 전체 input tokens | 235,882 | 235,728 |
| 전체 output tokens | 8,163 | 8,046 |
| ANSWER_GENERATION latency 합계 | 119,806ms | 112,177ms |

수정 전 결과는 `answer-consistency-before-p1.json`, 수정 후 결과는
`answer-consistency-after-p1.json`에 있다. 두 실행의 working tree와 Answer prompt가
서로 다르므로 1회 비교를 통계적 우열로 해석하지 않는다.

## 단계별 판정

- 각 케이스의 resolved query와 Retrieval Top-5는 한 실행의 5회 동안 모두 같았다.
- 수정 후 P1 검증 재생성은 0회였다. 이번 실호출 표본에서는 미선택·미존재 marker나
  본문 형식 검증 실패가 발생하지 않아 추가 호출 비용도 없었다.
- AC01은 Retrieval 이후 Answer의 정의 필수 개념 누락이 주된 실패였다.
- AC02와 AC07 첫 턴은 Planning이 4개 고유 Citation을 선택해 코드가
  `AMBIGUOUS_QUESTION`으로 보류하는 변동이 주된 실패였다.
- 수정 후 AC06 1회는 같은 고정 Retrieval 입력에서 Planning이 `OUT_OF_SCOPE` 대신
  `INSUFFICIENT_EVIDENCE`를 선택했다.
- 따라서 37→36 차이는 P1 검증 경로의 재실패가 아니라 Planning/Answer 변동이다.

## P1 회귀 보장

실호출 빈도와 별개로 결정적 단위 테스트에서 다음 경로를 고정했다.

- Answer에 전달하지 않은 SOURCE marker는 Top-5에 있어도 통과하지 않는다.
- 유효 marker와 미존재 marker가 섞이면 잘못된 marker만 삭제하지 않고 전체를 실패시킨다.
- marker 전무, 고유 Citation 4개 초과, 중복 Citation 병합을 검증한다.
- 형식·marker 실패 시 원 질문·선택 Source·Required Answer Coverage를 유지하고
  Answer 단계만 최대 1회 재생성한다.
- 재검증 실패는 `UNVERIFIABLE_ANSWER`, 재생성 API 실패는 기존 ERROR 계약을 따른다.
- 검증 재생성과 일시 API 오류 재시도의 횟수·token·latency trace를 구분한다.
