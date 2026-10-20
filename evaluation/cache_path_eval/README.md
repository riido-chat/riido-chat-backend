# 캐시 경로 오프라인 평가

이 디렉터리는 두 경로를 제공한다. `runner.py`는 동결된 cache-path 선택 manifest와 별도로
수집한 턴별 관측값(JSONL)을 대조해 P1/P3/P4/P6 경로별 집계를 만든다. `cli.py`는 등록된
서비스 profile과 실행 adapter를 주입받아 공통 평가 lifecycle에서 cache-only replay를 수행하고,
텍스트 없는 결과와 trace를 비공개 등록부에 기록한다.

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
호출하지 않는다. 실제 cache-only replay는 아래 서비스 profile 경계에서 실행한다.

## 운영 유스케이스와 실행 범위

애플리케이션의 공통 `AnswerQuestion` 유스케이스는 `FULL_ANSWER`와 `CACHE_DECISION` 종료점을
제공한다. 운영 채팅은 `FULL_ANSWER`를 사용한다. `CacheDecisionExecutionAdapter`는 평가에서
`CACHE_DECISION`을 호출한다. 흐름은 먼저 검색 전 정확 캐시를 확인하고, 미처리일 때 질의 확정,
검색, 검색 후 의미 캐시 판정으로 이어진다. 정확 캐시가 답을 제공하면 질의 재작성과 검색도
건너뛴다. 어느 경로든 답변 생성에는 진입하지 않는다.

검색 전 정확 캐시와 검색 후 의미 캐시는 서로 다른 포트다. Riido profile이 같은 평가 전용
읽기 데이터와 선택한 서비스 프로필을 사용해 두 포트를 조립한다. 운영 `ChatService`의 콜백을
평가에 넘기지 않는다. 그 콜백은 운영 로그·분류·캐시 시도 기록을 쓰기 때문이다. 평가 결과와
단계별 추적은 비공개 등록부에 쓰며, 골든 문항 원문은 결과에 복사하지 않는다. 답변 생성을
건너뛰어도 경로에 따라 임베딩이나 캐시 판정 모델 호출은 생길 수 있다.

`execute_cache_path_run`은 공통 `evaluation.core.run.run_evaluation` lifecycle을 사용한다. 전체
manifest를 먼저 검증하고, run ID를 create-only 상태 파일로 예약한 뒤 모든 replay context를
확인한다. 각 턴은 callback 준비 → `CACHE_DECISION` 실행 → 텍스트 없는 관측/trace checkpoint
순서로 처리한다. 실패 시 원시 예외 메시지는 버리고 실패 단계와 예외 형식만 ledger에 남긴다.
완료 JSON을 create-only로 쓴 다음 상태를 `SUCCEEDED`로 바꾼다. 결과 파일 쓰기 뒤 상태 전이가
실패하면 `FINALIZATION_INCOMPLETE`로 남아 결과 파일과 상태 ledger의 불일치를 드러낸다.

기본 경로는 `evals/cache-path/<run-id>.json`이며, 진행 상태는 `state/<run-id>.json`, 성공 턴
checkpoint는 `checkpoints/<run-id>/<item>-<turn>.json`, 실행 실패는
`failures/<run-id>.json`에 저장된다. 파일은 모두 create-only 또는 atomic replace 방식으로 쓴다.
실패한 실행은 성공 결과 JSON을 만들지 않으며 이미 완료한 턴의 checkpoint는 보존한다. 정상적인
취소는 실패 상태를 best-effort로 기록한 뒤 취소를 다시 전달한다. 프로세스 강제 종료처럼
정리 코드를 실행하지 못한 경우에는 상태가 `RUNNING` 또는 `FINALIZING`으로 남을 수 있다.
상태, checkpoint, 실패 ledger, 완료 결과의 필드 제한은 각각 `run-state.schema.json`,
`checkpoint.schema.json`, `failed-run.schema.json`, `run.schema.json`에 정의돼 있다. 실패 단계 trace는
`failure-trace.schema.json`을 따른다.

manifest 검증과 결과 저장은 같은 검증 함수를 사용한다. factory 시그니처는 유지하지만 전달되는
manifest에는 `group`, `goldenVersion`, 선택적인 `sourceCorpusVersion`, 입력 provenance 해시,
그리고 `itemId`/`turnNo`만 있는 turns가 포함된다. 각 replay-context 및 cache callback에는
`itemId`와 `turnNo`만 전달한다. `path`, `kind`,
`expectedServed`, `expectedJudgeDecision` 등 정답 라벨과 골든 경로 정보는 점수 계산을 맡은 평가
task 내부에만 남는다. CLI의 사전 중복 확인은 읽기 전용이며, 실제 run 소유권은 공통 lifecycle이
원자적으로 예약한다. 따라서 중복 동시 실행에서도 callback과 평가 실행은 한 번만 시작된다.
profile factory는 설정 객체만 조립하고 DB·네트워크·모델 작업을 시작하지 않아야 한다.

`CacheDecisionService`는 후보 준비, judge 호출 정규화, gate 판단, 정확 캐시 원천의 최신성
확인을 공유한다. 이는 운영 DB 쓰기가 없는 결정 core이며, 주입된 judge client는 외부 모델 호출을
할 수 있다. `ReadOnlyRiidoServiceData`는 선택한 profile/index/embedding 설정을 검증하고 각 턴마다
PostgreSQL repeatable-read, read-only snapshot을 연다. 각 snapshot은 턴 종료 때 rollback/close한다.
한 턴 안에서는 일관된 snapshot이지만 여러 턴이 하나의 DB 시점을 공유하지 않으므로 승인 상태가
턴 사이에 달라질 수 있다.

Riido 연결은 production exact-log reader와 question-vector helper를 재사용한다. 임베딩 재시도
정책은 helper에 공유되고, 운영 facade에는 `_question_vector` 모델 호출 checkpoint와 실패 매핑,
exact-question-log lookup이 남아 있다. Eval 연결은 운영 기록 writer를 사용하지 않는다. DB guarded
test는 read-only 트랜잭션이 쓰기를 거부하는지 검사하며 전체 live replay를 수행하지 않는다. 일반 자동
검증은 fake reader/model client를 사용하고, 실제 DB replay 또는 유료 model/embedding 실행과 비용
검증을 뜻하지 않는다.
관측은 논리적 model-call 횟수를 기록하고 retry 시도는 trace의 `retryCount`에 남기며 금액은
수집하지 않는다.

질문 분류 모델 호출에는 선택적으로 `requestEvidence`를 기록할 수 있다. 이 필드는 공통 입력,
정규화된 논리 요청, 지시문 해시와 입력 변환 버전 및 응답 모델 식별자만 담으며 원문 입력이나
요청 텍스트를 저장하지 않는다. 제공자가 응답한 모델 식별자를 기록하며, 별칭인 경우 고정 버전을
증명하지 않는다. 반환값이 없거나 안전한 식별자 형식이 아니면 `null`로 기록한다. 기존 checkpoint와
실행 기록에서는 이 필드가 생략될 수 있다.

`profiles.riido_chat.register_riido_chat_profile`은 `HELP_CHATBOT` 또는
`HELP_CHATBOT_TEST`를 명시해 factory를 등록한다. 호출자는 `ReadOnlyRiidoServiceData`에 session
factory를, profile에 embedder/rewrite/judge 의존성을 직접 주입해야 한다. profile factory 생성은
이 의존성을 실행하지 않으며 callback 실행은 run 예약 뒤에 시작한다. Bootstrap은 등록 후
`run_cache_path_cli(..., execution_adapter=create_riido_cache_decision_execution_adapter())`를
호출한다. CLI의 기본 `main()`은 DB/model 의존성을 만들지 않고 adapter가 주입되지 않으면 거부한다.
private registry의 `golden/manifest.json`은 `golden_commit`에 고정되고,
`golden/sources/cache-drafts.json`은 frozen manifest hash 및 선택 manifest의 `draftsSha256`로
검증되며 draft 진단 상태(`status: draft`)도 확인한다. Runtime은 고정된 source corpus에서 질문과
명시된 이전 턴 문맥을 비공개로 읽고 운영과 동일하게 가장 최근의 유효한 이전 턴 최대 5개만
rewrite 입력으로 제공한다. 미지원된 prior-turn 상태는 fail closed 한다. 공개 manifest와 결과에는
그 원문을 복사하지 않는다.
Factory에는 source corpus version, 입력 provenance 해시, `{itemId, turnNo}` 참조만 전달하고,
runtime judge 입력에서는 `path`, `kind`, expected labels를 제외한다.

## 프로필 선택 실행 경계

`profile_registry.py`는 service key를 factory에 연결하며, factory는 동기 또는 async일 수 있다.
`RiidoChatCachePathEvaluationProfile`은 읽기 전용 서비스 데이터 포트와 비공개 평가 provenance
포트를 주입받는 경계다. 운영 `ChatService` 콜백을 재사용하면 그룹 판정·캐시 시도·턴 로그를 운영
테이블에 쓸 수 있으므로 평가 어댑터로 등록해서는 안 된다.
