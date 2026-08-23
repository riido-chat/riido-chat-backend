#!/usr/bin/env bash
# 앱 EC2에서 실행한다. 배포할 이미지 URI를 인자로 받는다.
set -euo pipefail

IMAGE_URI="${1:?배포할 이미지 URI를 인자로 전달하세요}"
APP_DIR="/opt/riido"
COMPOSE_FILE="${APP_DIR}/docker-compose.app.yml"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"

cd "$APP_DIR"
export IMAGE_URI

echo "[deploy] ECR 로그인"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${IMAGE_URI%%/*}"

echo "[deploy] 이미지 pull: ${IMAGE_URI}"
docker compose -f "$COMPOSE_FILE" pull

# 마이그레이션이 실패하면 실행 중인 앱을 그대로 두고 배포를 중단한다
echo "[deploy] 마이그레이션 적용"
docker run --rm --env-file "${APP_DIR}/.env" "$IMAGE_URI" alembic upgrade head

echo "[deploy] 컨테이너 기동"
docker compose -f "$COMPOSE_FILE" up -d

echo "[deploy] 헬스체크"
for _ in $(seq 1 15); do
  if curl -fsS http://localhost:8000/health > /dev/null 2>&1 \
    && curl -fsS http://localhost:8000/health/db > /dev/null 2>&1; then
    echo "[deploy] 완료"
    docker image prune -f > /dev/null
    exit 0
  fi
  sleep 2
done

echo "[deploy] 헬스체크 실패" >&2
docker compose -f "$COMPOSE_FILE" logs --tail 50 >&2
exit 1
