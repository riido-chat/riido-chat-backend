#!/usr/bin/env bash
# 실행 중인 애플리케이션의 문서 그룹별 색인 도메인 경로를 호출한다.
# 이 스크립트는 문서를 수집하지 않는다. 이미 DB에 READY로 저장된 문서만
# /api/admin/document-groups/{id}/reindex 경로로 색인하고, 성공한 그룹의
# 메모리 corpus를 /internal/corpus/reload 로 다시 읽는다.
set -uo pipefail

APP_DIR="${APP_DIR:-/opt/riido}"
LOG_DIR="${APP_DIR}/logs"
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODE="${1:-ALL}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/riido-reindex.XXXXXX")"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/reindex-$(date +%Y%m%d-%H%M%S).log"
trap 'rm -rf "$TMP_DIR"' EXIT

# 상세 출력은 로그 파일에만 남긴다. SSM 출력은 24000자에서 잘리므로 요약만 남긴다.
log() { echo "[reindex] $*" | tee -a "$LOG_FILE"; }

json_code() {
  "${PYTHON_BIN:-python3}" -c '
import json, sys
try:
    value = json.load(sys.stdin).get("code")
except (json.JSONDecodeError, AttributeError):
    value = None
print(value or "UNKNOWN")
' 2>/dev/null || echo UNKNOWN
}

json_groups() {
  "${PYTHON_BIN:-python3}" -c '
import json, sys
payload = json.load(sys.stdin)
groups = payload.get("groups", [])
for group in sorted(groups, key=lambda item: (int(item["groupId"]), item.get("groupKey", ""))):
    group_id = int(group["groupId"])
    group_key = group.get("groupKey", "")
    print(f"{group_id}\t{group_key}")
' 2>/dev/null
}

if [[ "$MODE" != "ALL" ]]; then
  log "지원하지 않는 모드입니다: ${MODE}. ALL만 허용합니다."
  exit 2
fi

# 실행 중인 앱과 같은 이미지를 써서 코드 버전을 일치시킨다
DOCKER_BIN="${DOCKER_BIN:-docker}"
IMAGE_URI=$("$DOCKER_BIN" inspect --format '{{.Config.Image}}' riido-chat-api 2>/dev/null || true)
if [ -z "$IMAGE_URI" ]; then
  echo "[reindex] 실행 중인 riido-chat-api 컨테이너를 찾을 수 없습니다" >&2
  exit 1
fi

log "이미지: ${IMAGE_URI}"
log "모드: ALL"
log "로그: ${LOG_FILE}"

GROUPS_FILE="${TMP_DIR}/groups.json"
if ! curl -fsS --connect-timeout 10 --max-time 60 \
  "${BASE_URL}/api/admin/document-groups" >"$GROUPS_FILE"; then
  log "문서 그룹 목록을 조회하지 못했습니다."
  exit 1
fi
GROUP_LINES="${TMP_DIR}/groups.tsv"
if ! json_groups <"$GROUPS_FILE" >"$GROUP_LINES"; then
  log "문서 그룹 응답 형식이 올바르지 않습니다."
  exit 1
fi

total=0
success=0
failed=0
summary_file="${TMP_DIR}/summary.tsv"
: >"$summary_file"

while IFS=$'\t' read -r group_id group_key; do
  [[ -n "$group_id" ]] || continue
  total=$((total + 1))
  response_file="${TMP_DIR}/group-${group_id}.json"
  error_file="${TMP_DIR}/group-${group_id}.error"
  log "그룹 ${group_id} (${group_key:-unknown}) 색인 시작"

  # -f를 쓰지 않아 409 오류도 body의 code만 판별한다. 응답 전문은 로그에
  # 쓰지 않으므로 설정값이나 외부 API 오류가 SSM 출력으로 새지 않는다.
  : >"$response_file"
  http_status=$(curl -sS --connect-timeout 10 --max-time 3600 \
    -o "$response_file" -w '%{http_code}' -X POST \
    "${BASE_URL}/api/admin/document-groups/${group_id}/reindex" \
    2>"$error_file" || true)

  group_result="FAILED"
  if [[ "$http_status" == "200" ]]; then
    if curl -fsS --connect-timeout 10 --max-time 300 -X POST \
      "${BASE_URL}/internal/corpus/reload?documentGroupId=${group_id}" \
      >"${TMP_DIR}/reload-${group_id}.json" 2>"$error_file"; then
      group_result="SUCCESS"
      success=$((success + 1))
      log "그룹 ${group_id} 색인 및 corpus reload 완료"
    else
      log "그룹 ${group_id} reload 실패"
    fi
  else
    code=$(json_code <"$response_file")
    if [[ "$http_status" == "409" && "$code" == "REINDEX_NOT_REQUIRED" ]]; then
      # 이미 최신인 그룹도 현재 ACTIVE corpus를 다시 읽어 앱 상태를 맞춘다.
      if curl -fsS --connect-timeout 10 --max-time 300 -X POST \
        "${BASE_URL}/internal/corpus/reload?documentGroupId=${group_id}" \
        >"${TMP_DIR}/reload-${group_id}.json" 2>"$error_file"; then
        group_result="SUCCESS_NOOP"
        success=$((success + 1))
        log "그룹 ${group_id} 이미 최신 상태; corpus reload 완료"
      else
        log "그룹 ${group_id} 이미 최신이지만 reload 실패"
      fi
    else
      log "그룹 ${group_id} 색인 실패 (HTTP ${http_status}, code ${code})"
    fi
  fi

  if [[ "$group_result" == FAILED ]]; then
    failed=$((failed + 1))
  fi
  printf '%s\t%s\t%s\n' "$group_id" "${group_key:-unknown}" "$group_result" >>"$summary_file"
done <"$GROUP_LINES"

log "그룹별 결과"
while IFS=$'\t' read -r group_id group_key result; do
  [[ -n "$group_id" ]] || continue
  log "  ${group_id} (${group_key}): ${result}"
done <"$summary_file"
log "요약: 전체 ${total}, 성공 ${success}, 실패 ${failed}"

if (( failed > 0 )); then
  log "일부 그룹이 실패했습니다. 실패한 그룹의 기존 ACTIVE corpus는 보존됩니다."
  exit 1
fi
exit 0
