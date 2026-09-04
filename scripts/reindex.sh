#!/usr/bin/env bash
# 앱 EC2에서 실행한다. 문서 수집부터 BM25 리로드까지 한 번에 수행한다.
set -euo pipefail

APP_DIR="/opt/riido"
DATA_DIR="${APP_DIR}/data"
STAGING_DIR="${APP_DIR}/data.staging"
PREVIOUS_DIR="${APP_DIR}/data.previous"
LOG_DIR="${APP_DIR}/logs"
BASE_URL="http://localhost:8000"

mkdir -p "$LOG_DIR" "$DATA_DIR"
LOG_FILE="${LOG_DIR}/reindex-$(date +%Y%m%d-%H%M%S).log"

# 상세 출력은 로그 파일에만 남긴다. SSM 출력은 24000자에서 잘리므로 요약만 남긴다.
log() { echo "[reindex] $*" | tee -a "$LOG_FILE"; }

# 라이브러리가 오류 메시지에 인증 헤더를 그대로 담는 경우가 있어 로그에서 가린다
mask_secrets() { sed -E 's/sk-[A-Za-z0-9_-]{8,}/sk-***REDACTED***/g'; }

# 실행 중인 앱과 같은 이미지를 써서 코드 버전을 일치시킨다
IMAGE_URI=$(docker inspect --format '{{.Config.Image}}' riido-chat-api 2>/dev/null || true)
if [ -z "$IMAGE_URI" ]; then
  echo "[reindex] 실행 중인 riido-chat-api 컨테이너를 찾을 수 없습니다" >&2
  exit 1
fi

# staging 디렉터리가 root 소유라 이미지 기본 사용자(riido)로는 쓸 수 없다.
# 앱 컨테이너는 그대로 비루트로 읽기 전용 마운트를 사용한다.
run_stage() {
  docker run --rm --user root --env-file "${APP_DIR}/.env" \
    -v "${STAGING_DIR}:/app/data" \
    "$IMAGE_URI" python -m "$1" 2>&1 | mask_secrets >> "$LOG_FILE"
}

log "이미지: ${IMAGE_URI}"
log "로그: ${LOG_FILE}"

# 기존 corpus를 건드리지 않도록 staging에서 작업한다
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

log "1/5 문서 목록 수집"
run_stage app.document.gitbook.list_urls

log "2/5 원문 수집 (39건, 1초 간격)"
run_stage app.document.gitbook.fetch_pages

log "3/5 정제"
run_stage app.document.clean

log "4/5 벡터 색인"
run_stage app.indexing.index_vector_corpus

# 앱 컨테이너의 볼륨 마운트는 시작 시점의 디렉터리를 붙들고 있다.
# 디렉터리를 통째로 교체하면 컨테이너가 옛 디렉터리를 계속 보게 되므로 내용만 바꾼다.
log "5/5 corpus 교체와 리로드"
rm -rf "$PREVIOUS_DIR"
cp -a "$DATA_DIR" "$PREVIOUS_DIR"
find "$DATA_DIR" -mindepth 1 -delete
cp -a "${STAGING_DIR}/." "${DATA_DIR}/"
rm -rf "$STAGING_DIR"

curl -fsS -X POST "${BASE_URL}/internal/corpus/reload" 2>&1 | mask_secrets >> "$LOG_FILE"
log "corpus 상태: $(curl -fsS "${BASE_URL}/internal/corpus")"
log "완료. 이전 corpus는 ${PREVIOUS_DIR}에 보관된다."
