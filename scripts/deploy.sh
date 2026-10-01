#!/usr/bin/env bash
# 앱 EC2에서 실행한다. 배포할 이미지 URI를 인자로 받는다.
#
# 커밋 태그 이미지가 쌓여 디스크가 가득 차지 않도록 pull 전과 배포 성공 후에 앱 이미지를 정리한다.
# - 앱 이미지(APP_IMAGE_REPO): 컨테이너가 쓰는 이미지와 그 밖의 최신 KEEP_APP_IMAGES 개만 남긴다
# - 앱 이외의 이미지(caddy 등): dangling 이미지만 지운다
# - 볼륨과 네트워크는 건드리지 않는다
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/riido}"
COMPOSE_FILE="${APP_DIR}/docker-compose.api.prod.yml"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
# 앱 이미지 저장소 이름. ECR 주소를 뺀 마지막 경로와 비교한다
APP_IMAGE_REPO="${APP_IMAGE_REPO:-riido-chat-api}"
# 컨테이너가 쓰는 이미지 외에 롤백용으로 남길 최신 앱 이미지 수
KEEP_APP_IMAGES="${KEEP_APP_IMAGES:-2}"
# pull 전에 Docker 데이터 경로에 남아 있어야 하는 최소 여유 공간(MB)
MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-2048}"

log() { echo "[deploy] $*"; }
warn() { echo "[deploy] 경고: $*" >&2; }

# ECR 주소가 붙은 이름과 붙지 않은 이름을 모두 앱 저장소로 본다
is_app_repo() {
  [[ "$1" == "$APP_IMAGE_REPO" || "$1" == */"$APP_IMAGE_REPO" ]]
}

# 중지된 컨테이너를 포함해 컨테이너가 쓰는 이미지 ID를 한 줄에 하나씩 출력한다
used_image_ids() {
  local containers
  containers="$(docker container ls -aq --no-trunc)" || return 1
  if [[ -z "$containers" ]]; then
    return 0
  fi
  # shellcheck disable=SC2086 # 컨테이너 ID 목록을 인자로 나눠 넘긴다
  docker container inspect --format '{{.Image}}' $containers
}

# 이미지 목록($1)에서 이미지 ID($2)에 붙은 앱 태그를 출력한다
app_refs_of() {
  local id repo tag
  while IFS=$'\t' read -r id repo tag; do
    if [[ "$id" == "$2" && "$tag" != "<none>" ]] && is_app_repo "$repo"; then
      echo "${repo}:${tag}"
    fi
  done <<< "$1"
  return 0
}

# 앱 이미지 보존 규칙을 적용한다.
# 컨테이너가 쓰는 이미지는 모두 남기고, 나머지는 생성 시각이 최신인 KEEP_APP_IMAGES 개만 남긴다.
# 목록을 읽지 못하면 아무것도 지우지 않고, 개별 이미지 삭제 실패는 기록만 하고 넘어간다.
prune_app_images() {
  if ! [[ "$KEEP_APP_IMAGES" =~ ^[0-9]+$ ]]; then
    warn "KEEP_APP_IMAGES 값이 0 이상의 정수가 아니어서 앱 이미지 정리를 건너뜁니다: ${KEEP_APP_IMAGES}"
    return 0
  fi

  local listing used
  if ! listing="$(docker image ls --no-trunc --format '{{.ID}}\t{{.Repository}}\t{{.Tag}}')"; then
    warn "이미지 목록을 읽지 못해 앱 이미지 정리를 건너뜁니다"
    return 0
  fi
  # 사용 중인 이미지를 모르면 실행 중인 앱 이미지를 지울 수 있으므로 정리하지 않는다
  if ! used="$(used_image_ids)"; then
    warn "컨테이너 목록을 읽지 못해 앱 이미지 정리를 건너뜁니다"
    return 0
  fi

  local id repo tag app_ids=""
  while IFS=$'\t' read -r id repo tag; do
    if [[ -n "$id" ]] && is_app_repo "$repo"; then
      app_ids+="${id}"$'\n'
    fi
  done <<< "$listing"
  app_ids="$(printf '%s' "$app_ids" | sort -u)"
  if [[ -z "$app_ids" ]]; then
    log "정리할 앱 이미지 없음"
    return 0
  fi

  # 생성 시각(초 단위, UTC)이 최신인 순서로 정렬한다
  local inspected sorted="" created
  # shellcheck disable=SC2086 # 이미지 ID 목록을 인자로 나눠 넘긴다
  if ! inspected="$(docker image inspect --format '{{.Created}} {{.Id}}' $app_ids)"; then
    warn "이미지 생성 시각을 읽지 못해 앱 이미지 정리를 건너뜁니다"
    return 0
  fi
  while read -r created id; do
    if [[ -n "$id" ]]; then
      sorted+="${created:0:19} ${id}"$'\n'
    fi
  done <<< "$inspected"
  sorted="$(printf '%s' "$sorted" | sort -r)"

  local kept=0 removed=0 failed=0 short refs ref
  local -a targets
  while read -r created id; do
    [[ -n "$id" ]] || continue
    short="${id#sha256:}"
    short="${short:0:12}"
    if grep -Fxq -- "$id" <<< "$used"; then
      log "사용 중인 앱 이미지 유지: ${short}"
      continue
    fi
    if (( kept < KEEP_APP_IMAGES )); then
      kept=$((kept + 1))
      log "롤백용 앱 이미지 유지: ${short} (${created})"
      continue
    fi

    # 태그로 지워야 다른 저장소 태그가 함께 붙은 이미지를 강제로 지우지 않는다
    targets=()
    refs="$(app_refs_of "$listing" "$id")"
    while read -r ref; do
      if [[ -n "$ref" ]]; then
        targets+=("$ref")
      fi
    done <<< "$refs"
    if (( ${#targets[@]} == 0 )); then
      targets=("$id")
    fi

    if docker image rm "${targets[@]}" > /dev/null; then
      removed=$((removed + 1))
      log "앱 이미지 삭제: ${short} (${created})"
    else
      failed=$((failed + 1))
      warn "앱 이미지 삭제 실패, 계속 진행합니다: ${short}"
    fi
  done <<< "$sorted"

  log "앱 이미지 정리 결과: 삭제 ${removed}개, 실패 ${failed}개, 롤백용 유지 ${kept}개"
  return 0
}

# 태그가 없는(dangling) 이미지만 지운다. -a 를 쓰지 않으므로 caddy 같은 이미지는 남는다
prune_dangling_images() {
  if ! docker image prune -f > /dev/null; then
    warn "dangling 이미지 정리 실패, 계속 진행합니다"
  fi
  return 0
}

cleanup_images() {
  prune_dangling_images
  prune_app_images || warn "앱 이미지 정리 중 오류가 있었지만 계속 진행합니다"
}

# Docker 데이터 경로의 여유 공간이 기준보다 작으면 pull 전에 배포를 중단한다
check_free_space() {
  if ! [[ "$MIN_FREE_DISK_MB" =~ ^[0-9]+$ ]]; then
    echo "[deploy] MIN_FREE_DISK_MB 값이 0 이상의 정수가 아닙니다: ${MIN_FREE_DISK_MB}" >&2
    return 1
  fi

  local root avail_kb avail_mb
  root="$(docker info --format '{{.DockerRootDir}}' 2> /dev/null || true)"
  root="${root:-/var/lib/docker}"
  avail_kb="$(df -Pk "$root" 2> /dev/null | awk 'NR == 2 { print $4 }' || true)"
  if ! [[ "$avail_kb" =~ ^[0-9]+$ ]]; then
    echo "[deploy] 디스크 여유 공간을 확인하지 못해 배포를 중단합니다: ${root}" >&2
    return 1
  fi

  avail_mb=$((avail_kb / 1024))
  if (( avail_mb < MIN_FREE_DISK_MB )); then
    echo "[deploy] 디스크 여유 공간 부족으로 이미지 pull 전에 배포를 중단합니다:" \
      "${root} 남은 공간 ${avail_mb}MB, 필요 ${MIN_FREE_DISK_MB}MB" >&2
    return 1
  fi
  log "디스크 여유 공간 ${avail_mb}MB (${root}, 기준 ${MIN_FREE_DISK_MB}MB)"
}

main() {
  IMAGE_URI="${1:?배포할 이미지 URI를 인자로 전달하세요}"
  export IMAGE_URI
  cd "$APP_DIR"

  # 쌓인 이미지 때문에 pull 이 디스크 부족으로 실패하지 않도록 먼저 정리한다
  log "pull 전 이미지 정리"
  cleanup_images
  check_free_space

  log "ECR 로그인"
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "${IMAGE_URI%%/*}"

  log "이미지 pull: ${IMAGE_URI}"
  docker compose -f "$COMPOSE_FILE" pull

  # 마이그레이션이 실패하면 실행 중인 앱을 그대로 두고 배포를 중단한다
  log "마이그레이션 적용"
  docker run --rm --env-file "${APP_DIR}/.env" "$IMAGE_URI" alembic upgrade head

  log "컨테이너 기동"
  docker compose -f "$COMPOSE_FILE" up -d

  log "헬스체크"
  for _ in $(seq 1 15); do
    if curl -fsS http://localhost:8000/health > /dev/null 2>&1 \
      && curl -fsS http://localhost:8000/health/db > /dev/null 2>&1; then
      log "완료"
      # 새 이미지가 실행 중이므로 이전 이미지 중 최신 KEEP_APP_IMAGES 개만 롤백용으로 남는다
      log "배포 후 이미지 정리"
      cleanup_images
      exit 0
    fi
    sleep 2
  done

  echo "[deploy] 헬스체크 실패" >&2
  docker compose -f "$COMPOSE_FILE" logs --tail 50 >&2
  exit 1
}

# 테스트에서 source 로 함수만 불러올 수 있도록 직접 실행할 때만 main 을 호출한다
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
