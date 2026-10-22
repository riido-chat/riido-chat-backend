# P2/P3 답변 일관성 실험 비교

> 이 문서의 v9/v11은 원인 가설을 확인하기 위한 프롬프트 개입 실험이다. v11의 50/50을
> 제품 해결책 채택이나 일반적인 답변 일관성 보장으로 해석하지 않는다.

실행일: 2026-09-06  
평가 세트: `evaluation/answer_consistency_cases.json` (`answer-consistency-v1`)  
실행 방식: in-process, 10개 케이스 × 5회

## 결과

| 구분 | P1 기준 | P2 temperature=0 | P3 v9 실험 | P3 v11 실험 |
| --- | ---: | ---: | ---: | ---: |
| 통과 | 36/50 | 35/50 | 46/50 | **50/50** |
| NEW_CONVERSATION | 21/30 | 20/30 | 26/30 | **30/30** |
| CONTINUOUS_CONVERSATION | 5/10 | 5/10 | 10/10 | **10/10** |
| FIXED_CONTEXT | 10/10 | 10/10 | 10/10 | **10/10** |
| 전체 ModelCall | 120회 | 120회 | 120회 | 120회 |
| 전체 input tokens | 235,728 | 231,430 | 272,381 | 291,745 |
| 전체 output tokens | 8,046 | 7,956 | 9,305 | 9,971 |
| ANSWER_GENERATION latency 합계 | 112,177ms | 99,892ms | 121,016ms | 138,674ms |

결과 파일:

- P1 기준: `answer-consistency-after-p1.json`
- P2: `answer-consistency-temperature-zero.json`
- P3 중간 확인: `answer-consistency-v9-final.json`
- P3 v11 개입 실험: `answer-consistency-v11-final.json`

각 실행은 비결정적인 외부 모델을 한 번씩 5회 반복한 제한된 표본이다. 50/50을 모든 질문에서의
완전한 결정성으로 확대 해석하지 않고, 같은 평가 입력에서 수용 기준이 반복 충족됐다는 뜻으로
사용한다.

## P2 판정

- gpt-5.4-mini Responses Structured Output 호출에서 `temperature=0` 호환성을 실제 확인했다.
- Query Rewrite·Source Planning·Answer에 `temperature=0`을 적용한 결과는 35/50으로,
  P1의 36/50보다 개선되지 않았다.
- 실패는 AC01 5회, AC02 5회, AC07 5회에 집중됐다. 낮은 temperature가 필요한 설명 누락과
  과도한 Source 선택을 해결하지 못했으므로 제품 설정에는 채택하지 않았다.
- 최종 v11도 temperature와 reasoning을 명시하지 않는 기존 호출 설정을 유지한다.

## P3 개입 실험과 판정

- Source Planning v10은 질문을 SUMMARY와 MULTI_DETAIL로 구분하고, 질문이 요구하지 않은
  세부 근거를 확장하지 않으며 최소 충분 Source 집합을 선택하도록 강화했다.
- 문서에 기능이 언급되지 않았다는 사실만으로 미지원 결론을 만들지 않고, 직접적인 지원·미지원
  근거가 없으면 `INSUFFICIENT_EVIDENCE`로 보류하도록 했다.
- Answer v11은 용어의 분류와 사용 주체를 보존하고, 문서가 연동 관계를 설명할 때
  `연동/연결` 표현과 구체적인 효과를 함께 답하도록 했다.
- 고유 Citation 3개를 초과한 Source Plan만 동일 질문·Top-5로 최대 1회 교정한다. 교정 후에도
  초과하면 기존 `AMBIGUOUS_QUESTION` 보류 정책을 유지한다.
- v11 50회에서는 Source Plan 교정과 Answer 검증 재생성이 모두 0회였다. 따라서 50/50은
  추가 호출로 실패를 덮은 결과가 아니라 첫 Planning·Answer 결과가 수용 기준을 충족한 결과다.
  다만 평가 사례를 프롬프트에 직접 반영한 개입이라 제품 변경의 채택 근거로는 부족하다.

## 비용·관측 해석

- P3 최종의 ANSWER_GENERATION input tokens는 231,811로 P1의 176,629보다 55,182 많다.
  Planning과 Answer 프롬프트의 판단 기준을 구체화한 영향이며, DB의 ANSWER_GENERATION 한 건은
  두 단계를 합산해 기록한다.
- 한정 Source Plan 교정은 일반 요청에 항상 호출되지 않고, 3개 초과 계획에서만 실행된다.
  최종 표본에서는 발동하지 않아 ModelCall 수는 P1과 같은 120회였다.
- 평가 trace에는 최초 계획, 교정 계획, 교정 호출 횟수·token·latency를 분리해 기록하므로
  향후 발동 빈도와 추가 비용을 결과 파일에서 확인할 수 있다.

## 남은 한계

- 평가 세트는 대표 10개 사례이므로 실제 질의 분포를 모두 대표하지 않는다.
- 의미 단위의 사실성은 기대 키워드·상태·Citation 규칙으로 평가한다. 모든 생성 문장의 의미를
  Backend가 자동 검증하는 구조는 아니다.
- Source Plan 교정의 실제 성공률은 최종 표본에서 발동 횟수가 0이므로 운영 로그나 더 넓은
  스트레스 평가에서 별도로 관찰해야 한다.
