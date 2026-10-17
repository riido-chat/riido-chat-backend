# 캐시 경로 오프라인 평가

이 도구는 캐시 서비스를 실행하지 않는다. 동결된 cache-path 선택 manifest와 별도로
수집한 턴별 관측값(JSONL)을 대조해 P1/P3/P4/P6 경로별 집계 결과를 만든다. 이 방식으로
합성 입력을 검증할 수 있고, 나중에 승인된 실행에서 얻은 관측값도 같은 결과 형식으로
기록할 수 있다.

선택 manifest는 비공개 평가 등록부의 `scripts/build_cache_suite.py`가 만든다. 질문·답변
텍스트는 결과에 복사하지 않는다. 관측 JSONL에는 manifest의 `itemId`, `turnNo`와 다음
필드만 허용한다.

| 필드 | 의미 |
| --- | --- |
| `outcome` | `SERVED`, `SHADOW`, `GROUP_DISABLED`, `REJECTED`, `SKIPPED`, `FAILED` 중 실제 관측 결과 |
| `nextStage` | `PASS`, `FAIL`, `NOT_APPLICABLE`; PASS 비율의 분모는 PASS와 FAIL 합계 |
| `latencyMs` | 해당 턴의 캐시 경로 처리 시간(0 이상의 정수) |
| `modelCalls` | 관측된 모델 호출 횟수 |
| `embeddingCalls` | 관측된 임베딩 호출 횟수 |

모든 manifest 턴을 한 번씩 기록해야 한다. `expectedServed`가 false인 턴에서 `SERVED`가
관측되면 `ineligibleServed`에 집계한다. `context_only`는 `expectedServed` 기준이 없으므로
이 지표의 분모에 포함되지 않는다. 결과 파일에는 개별 문항 ID와 원문이 없고 집계값 및
입력 파일 SHA-256만 담긴다. 입력에 질문·답변 등 허용되지 않은 필드가 있으면 실행을
거부한다.

```sh
python -m evaluation.cache_path_eval.runner \
  --manifest /private/path/cache-path-manifest.json \
  --observations /private/path/cache-observations.jsonl \
  --out /private/path/cache-path-result.json
```

출력 파일은 이미 존재하면 덮어쓰지 않는다. 이 명령은 DB, 모델, 임베딩 서비스나 네트워크를
호출하지 않는다. 관측값을 만드는 실제 실행 절차와 시스템·프로필 식별자는 별도 승인 및
사전 등록이 필요하며 이 도구가 이를 대신하지 않는다.

## 운영 유스케이스와 실행 범위

애플리케이션의 공통 `AnswerQuestion` 유스케이스는 `FULL_ANSWER`와 `CACHE_DECISION` 종료점을
제공한다. 운영 채팅은 `FULL_ANSWER`를 사용한다. `CacheDecisionExecutionAdapter`는 평가에서
`CACHE_DECISION`을 호출해 질의 확정, 검색, 캐시 분류·게이트 결과까지만 실행하고 관측 JSONL에
넣을 텍스트 없는 행을 만든다. 캐시 결과를 운영 응답으로 처리하는 콜백과 답변 생성 단계에는
진입하지 않는다. 답변 생성은 건너뛰지만, 선택한 경로에 따라 검색 임베딩이나 캐시 판정 모델
호출은 발생할 수 있으므로 캐시 전용 평가가 항상 무료인 것은 아니다.

콜백 묶음은 선택된 서비스 설정판의 읽기 전용 자료와 비공개 평가 등록부에 연결해야 한다. 운영
`ChatService`의 콜백을 그대로 넘기면 캐시 분류·게이트 기록이 운영 DB에 저장될 수 있으므로
사용하지 않는다. 실행 어댑터는 호출자가 묶어 준 콜백의 읽기·쓰기 대상을 자동으로 바꾸지 않는다.
현재는 공통 유스케이스에 연결하는 어댑터 라이브러리까지 마련됐고, 특정 서비스 DB·프로필을
선택해 콜백을 조립하는 실행 설정은 아직 없다. 기존 `runner` CLI는 계속 오프라인 관측값 집계기이며,
모델이나 DB를 호출하지 않는다.
