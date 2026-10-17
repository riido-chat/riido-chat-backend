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
`CACHE_DECISION`을 호출한다. 흐름은 먼저 검색 전 정확 캐시를 확인하고, 미처리일 때 질의 확정,
검색, 검색 후 의미 캐시 판정으로 이어진다. 정확 캐시가 답을 제공하면 질의 재작성과 검색도
건너뛴다. 어느 경로든 답변 생성에는 진입하지 않는다.

검색 전 정확 캐시와 검색 후 의미 캐시는 서로 다른 포트다. 각 서비스 어댑터가 같은 평가 전용
읽기 데이터와 선택한 서비스 프로필을 사용해 두 포트를 조립한다. 운영 `ChatService`의 콜백을
평가에 넘기지 않는다. 그 콜백은 운영 로그·분류·캐시 시도 기록을 쓰기 때문이다. 평가 결과와
단계별 추적은 비공개 등록부에 쓰며, 골든 문항 원문은 결과에 복사하지 않는다. 답변 생성을
건너뛰어도 경로에 따라 임베딩이나 캐시 판정 모델 호출은 생길 수 있다.

공통 포트와 추적 계약은 마련됐다. 운영 채팅은 이제 검색 전 정확 캐시와 검색 후 의미 캐시를
공통 유스케이스에서 순서대로 실행한다. `execute_cache_path_run`은 서비스 profile adapter가
만든 포트 묶음을 문항별로 실행하고, `CachePathRunStore`는 버전·프로필·색인 설정·단계 trace와
집계를 `evaluation/registry/evals/cache-path/<run-id>.json`에 덮어쓰기 없이 저장한다. 결과에는
질문·답변·후보·문서 본문을 허용하지 않는다. 다만 특정 서비스의 실제 평가 전용 읽기 어댑터와
프로필을 선택하는 CLI 설정은 아직 없다. 기존 `runner` CLI는 계속 모델과 DB를 호출하지 않는
오프라인 집계기다.
