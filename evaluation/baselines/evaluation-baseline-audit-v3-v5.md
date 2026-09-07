# 잔여 평가 기준 및 기준선 감사

## 범위

- Answer Consistency v3 전체 10개 케이스
- CS v5 전체 25개 케이스
- Generation v18 저장 결과의 오프라인 재채점
- CS07·CS10 구조 오류 수정 후 표적 평가 기록 대조

모델과 Chat API는 다시 호출하지 않았다. 기존 결과의 응답·DB snapshot만 현재 평가 기준으로
재채점했다.

## 변경한 기준

| 케이스 | 변경 | 근거 |
| --- | --- | --- |
| AC06 | 기대 보류 사유를 `OUT_OF_SCOPE`에서 `INSUFFICIENT_EVIDENCE`로 변경 | 급여 계산은 뤼이도 기능에 관한 질문이므로 제품 범위 밖 일반 질문이 아니라 공식 문서 근거가 부족한 제품 질문이다. |
| CS01 | 정의 표현에 `공간`을 허용하고 `연동/연결` 문자열 강제를 제거 | 서비스 종류와 사용 목적, 뤼이도에서의 실제 관계·효과가 설명되면 특정 관계 단어 자체는 필수가 아니다. |
| CS13 | CS01과 같은 기준으로 슬랙 정의 조건을 조정 | 동일한 정의 답변 정책을 외부 서비스별로 일관되게 적용한다. |

나머지 AC·CS·Multi-turn 케이스의 상태, 보류 사유, 필수 개념 조건도 확인했다. 현재 정책과
충돌하거나 저장 결과에서 과도한 문자열 강제 문제가 확인되지 않아 변경하지 않았다.

## 재채점 결과

| 평가 | 결과 | 실행 오류 | 기준 불일치 |
| --- | ---: | ---: | ---: |
| Answer Consistency v3 | 50/50 | 0 | 0 |
| CS v5 | 23/25 | 2 | 0 |

CS v5의 실패 2건은 v18 실행 당시 HTTP 500이 저장된 CS07·CS10이다. 평가기는 이제 실패를
`executionErrorCaseExecutionCount`와 `criteriaMismatchCaseExecutionCount`로 분리하므로,
과거 실행 오류를 현재 평가 기준 불일치로 해석하지 않는다.

CS07·CS10은 이후 Source Plan 구조 교정이 적용된 v19 표적 평가에서 각각 5회, 총 10/10
통과했다. 따라서 v18 원본 기록은 삭제하거나 성공으로 바꾸지 않고 과거 실행 이력으로 보존한다.

## 결과 파일

- `answer-consistency-v18-rechecked-v3-recheck-20260907T162038Z.json`
- `chat-cs-v18-rechecked-v5-recheck-20260907T162038Z.json`
- 구조 오류 수정 후 확인: `source-plan-repair-v19-cs-targeted.json`
