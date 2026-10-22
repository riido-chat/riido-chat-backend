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

## 고정 입력 평가 계획

`plan.py`의 `execute_cache_path_plan`은 하나의 고정 manifest와 N개의 cache-path run 설정을
`evaluation.core.plan.run_evaluation_plan`에 연결한다. 계획과 모든 child run ID를 사전 검증하고,
parent plan은 shared input 준비 전에 원자적으로 예약한다. 각 child run ID는 기존 단일 실행 lifecycle에
들어갈 때 따로 예약되며 plan 전체의 원자성은 보장하지 않는다. 중간 child 예약 경합은 안전하게 실패하고
기존 결과를 보존하지만, 그 전에 발생한 shared preparation 비용은 되돌릴 수 없다. shared input preparer는 label-free reference manifest를 한 번만 받고,
variant별 prepared profile factory에는 해당 variant/spec, 같은 immutable `PreparedCacheInputs`,
그리고 같은 reference manifest를 전달한다. factory는 런타임 질문과 judge 입력을 다시 읽거나 만들지 말고,
bundle의 `payload_for_turn`에서 준비된 값을 가져와야 한다. payload 내부는 canonical JSON snapshot이고
접근할 때마다 새 복사본을 돌려준다. expected labels는 scoring manifest에만 남는다.

각 variant는 기존 `execute_cache_path_run`과 `CacheDecisionExecutionAdapter`를 사용하므로 child run은
기존 checkpoint와 `evals/cache-path/<run-id>.json` 결과 형식을 유지한다. plan ledger는 child 결과
경로와 SHA-256만 연결하며 prepared input 값은 기록하지 않는다. 공유 준비 비용은 parent의
`preparationEvidence`에 기록하고 child별 모델·임베딩 횟수에 중복 합산하지 않는다. 준비가 실패해
prepared bundle을 반환하기 전에 끝나면 현재 port는 실패 형식만 기록할 수 있어 그 시도의 상세 비용은
ledger에 남지 않는다.

parent variant configuration에는 judge model/prompt version, `profileConfigSha256`, 그리고 복원 가능한
`profileConfig` snapshot이 기록된다. `profileConfig`는 threshold와 같이 실행에 영향을 주는 안전한 JSON
설정만 담는다. credential, prompt 본문, 임의 텍스트 또는 절대 경로 같은 data locator는 넣지 말고, 필요한
data source는 shared preparer 주입 경계에서 선택한다.

실제 service/DB snapshot, judge/embedding client와 provider bootstrap은 이 모듈 바깥에서 명시적으로
주입해야 한다. prepared profile factory는 query rewrite와 embedding을 다시 호출하지 않고 준비된 query와
후보를 사용해야 한다. DB gate view 역시 injected data 경계에서 읽는다. 입력 payload hash는 전달된
payload를 고정하지만 DB 전체 view가 여러 run에서 같은 시점임을 보장하지 않는다. 현재는 고정 입력
capable factory와 shared-input preparer의 경계만 제공하며 production Jev transport 또는 credentials
bootstrap을 제공하지 않는다. `compare_cache_path_plan`은 완료 plan의
child hash를 확인한 뒤 기존 summary counter를 읽기 전용으로 보여준다. 자동으로 우승 variant를 고르거나
통계적 유의성을 주장하지 않는다.

Riido 계획 실행은 `profiles.riido_chat.create_riido_chat_plan_binding`으로 shared-input preparer와
prepared profile factory를 함께 만든다. 준비기는 선택한 턴마다 정확 캐시를 먼저 확인하고, 미제공 경로의
질의 재작성·검색·질문 벡터·후보 presentation을 한 번 계산한다. presentation payload와 검색/후보 객체는
프로세스 메모리에서만 공유한다. parent `preparationEvidence`에는 재작성/임베딩 호출 수와 집계된 비용
측정값만 기록하고, 각 child 관측에는 해당 child가 실제 실행한 분류 호출만 기록한다. parent에서 확인한
exact decision도 메모리에서 공유하므로 exact hit는 payload가 없는 입력으로 고정되고 각 child의 기존
`CACHE_DECISION` exact 처리 경로에서 judge를 호출하지 않는다.

기본적으로 각 child의 judge model/prompt는 선택한 production profile revision과 일치해야 한다. 평가용으로
다른 judge를 비교할 때는 일반 `judge_client_factory(variant, spec)`를 binding에 주입하고
`evaluation_only_judge_override=True`로 만든 뒤 모든 child의 `profileConfig.evaluationOnlyJudgeOverride`를
`true`로 설정한다. factory는 plan preflight와 shared preparation 이후 variant마다 한 번 호출된다.
variant metadata의
`judgeModelVersion`/`judgePromptVersion`이 실제 요청 대상이며 등록된 `judge` prompt 버전을 불러 사용한다.
서비스 profile revision, index, embedding, cache 설정은 계속 선택 metadata와 검증한다.
`profileConfig.judgeParams`에는 `reasoning`과 `max_output_tokens`만 안전한 JSON 값으로 지정할 수 있다.
키가 없으면 선택된 production stage의 요청 설정을 유지하고, `{}`를 지정하면 빈 요청 설정을 사용하므로
provider별 단계 계약을 설정만으로 표현할 수 있다. 모델 ID, 등록 prompt 버전, 요청 설정은 plan preflight에서
검증되며 이 검증이 끝나기 전에는 공통 rewrite/embedding 준비와 variant client factory가 실행되지 않는다.
credential이 필요 없는 provider-client 선택 검증은 선택적으로 `judge_client_validator(variant, spec)`에
주입할 수 있다. 실제 client factory는 preflight와 shared preparation 뒤에 실행된다. 검증은 prompt 본문을
metadata나 결과에 복사하지 않는다.

## 프로필 선택 실행 경계

`profile_registry.py`는 service key를 factory에 연결하며, factory는 동기 또는 async일 수 있다.
`RiidoChatCachePathEvaluationProfile`은 읽기 전용 서비스 데이터 포트와 비공개 평가 provenance
포트를 주입받는 경계다. 운영 `ChatService` 콜백을 재사용하면 그룹 판정·캐시 시도·턴 로그를 운영
테이블에 쓸 수 있으므로 평가 어댑터로 등록해서는 안 된다.

## 캐시 그룹 모델 비교

`evaluation.judge_eval.cache_group`은 별도 비공개 골든 저장소의 `cache-path@2.0.0` 그룹을
사용해 캐시 판별 모델만 비교한다. P1 정확 적중 20건은 모델을 호출하지 않고, 판별 대상 P3/P4/P6
65건만 실행한다. 실행 전 그룹·골든 manifest와 운영 payload 해시를 검증하며, 결과와 원시 모델 출력,
호출·비용 ledger는 비공개 등록부에 저장한다. 질문을 결과에 복사하지 않는다.

```sh
python -m evaluation.judge_eval.cache_group prepare \
  --registry /private/path/riido-evals \
  --group golden/groups/cache-path/2.0.0.json

python -m evaluation.judge_eval.cache_group smoke \
  --registry /private/path/riido-evals \
  --group golden/groups/cache-path/2.0.0.json \
  --judge claude-haiku \
  --output /private/path/riido-evals/evals/cache-path/smoke/<run-id>.json

python -m evaluation.judge_eval.cache_group run \
  --registry /private/path/riido-evals \
  --group golden/groups/cache-path/2.0.0.json \
  --judge luna \
  --output /private/path/riido-evals/evals/cache-path/<run-id>.json
```

지원 모델 별칭은 `luna`(gpt-6-luna), `jev`(jev-latest), `claude-haiku`, `claude-sonnet`,
`claude-opus`, `claude-fable`이다. OpenAI와 TypeSafe는 개인 환경의 `OPENAI_API_KEY`,
`TYPESAFE_API_KEY`, Anthropic은 `ANTHROPIC_API_KEY`를 사용한다. Claude 요청은 호출당 한 번만
시도하고 출력 상한 1024토큰으로 예약 비용을 계산한다. 이전 512토큰 설정은 Haiku, Sonnet,
Opus 실제 실행에서 각각 2건, 36건, 51건의 응답을 잘라 비교 결과를 불완전하게 만들었다.
상한은 실제로 확인된 사용량을 반영해 1024토큰으로 올렸으며, 사용자가 승인한 총 $20 한도에서
각 모델 실행 전에 누적 사용·예약액을 포함해 비용을 확인한다. 네 모델을 모두 65건씩 실행하는
보수적 상한 추정은 입력 토큰에 15% 안전 여유를 둬 약 $15.25이며, 기존 사용·예약액을 더한
계획 상한은 약 $21.23다. 이 여유는 이전 실제 Claude 응답에서 입력 토큰 사용량이 문자 기반
추정치보다 약 8~10% 높았던 점을 반영한다. $20 예산에서는 모든 모델을 다시 실행할 수 없으므로
각 실행은 예산 검사에 따라 승인되며, Fable 실행 여부는 앞선 실제 비용을 반영한다.
Claude가 출력 상한에 도달하면 해당 결과는 오류로 기록하고 유효한 판정으로 취급하지 않는다.
