"""판별 모델 오프라인 측정 실행기 (#204 단계별 평가 실행기의 판별 단계, #205 luna 대 Jev 비교).

평가 등록부(evaluation/registry, riido-evals)의 동결 판별 입력으로 판별 모델을 부르고, 결과를
schemas/eval-result.schema.json 모양으로 남긴다. 앞 단계(재작성, 후보 검색)는 동결 입력을 그대로
쓰므로 모델 사이에 판별 입력이 바이트까지 같다.

    golden.py    골든셋 버전과 동결 해시 확인, 묶음 선택, 라벨과 동결 행 읽기, 보류 시험셋 관문
    inputs.py    동결 행으로 운영 판별 payload 재구성 (app.question_grouping.payload 재사용)
    clients.py   판별 모델 어댑터 (luna 는 운영 QuestionJudgeClient, Jev 는 2026-10-01 실행과 같은 요청)
    ledger.py    실행 기록부. 묶음, 모델, 골든셋 버전마다 실행 2회 상한과 보류 시험셋 사용 횟수
    scoring.py   labels.md 6절 지표, 신뢰 상한, 두 결과 짝 비교
    cli.py       python -m evaluation.judge_eval {run,score,compare}
"""
