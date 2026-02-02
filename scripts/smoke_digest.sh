#!/usr/bin/env bash
set -euo pipefail

NOTIFY_DIR="services/orion-notify"
DIGEST_DIR="services/orion-notify-digest"

for dir in "${NOTIFY_DIR}" "${DIGEST_DIR}"; do
  if [[ ! -f "${dir}/.env" ]]; then
    echo "Missing ${dir}/.env. Copy ${dir}/.env_example to ${dir}/.env and edit settings." >&2
    exit 1
  fi
  if ! grep -q '^POSTGRES_URI=' "${dir}/.env"; then
    echo "POSTGRES_URI=sqlite:////data/notify.db" >> "${dir}/.env"
  fi
  if ! grep -q '^NOTIFY_SERVICE_URL=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "NOTIFY_SERVICE_URL=http://orion-notify:7140" >> "${dir}/.env"
  fi
  if ! grep -q '^DIGEST_RUN_ON_START=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "DIGEST_RUN_ON_START=false" >> "${dir}/.env"
  fi
  if ! grep -q '^DIGEST_ENABLED=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "DIGEST_ENABLED=true" >> "${dir}/.env"
  fi
  if ! grep -q '^DIGEST_WINDOW_HOURS=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "DIGEST_WINDOW_HOURS=24" >> "${dir}/.env"
  fi
  if ! grep -q '^DIGEST_TIME_LOCAL=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "DIGEST_TIME_LOCAL=07:30" >> "${dir}/.env"
  fi
  if ! grep -q '^DIGEST_RECIPIENT_GROUP=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "DIGEST_RECIPIENT_GROUP=juniper_primary" >> "${dir}/.env"
  fi
  if ! grep -q '^NOTIFY_API_TOKEN=' "${dir}/.env"; then
    echo "NOTIFY_API_TOKEN=" >> "${dir}/.env"
  fi
done

if [[ -z "${SKIP_COMPOSE:-}" ]]; then
  docker compose -f "${NOTIFY_DIR}/docker-compose.yml" up -d --build
  docker compose -f "${DIGEST_DIR}/docker-compose.yml" up -d --build
  sleep 2
fi

NOTIFY_PORT=$(grep -E '^PORT=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2)
NOTIFY_PORT=${NOTIFY_PORT:-7140}
API_TOKEN=$(grep -E '^API_TOKEN=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2-)
AUTH_HEADER=()
if [[ -n "${API_TOKEN}" ]]; then
  AUTH_HEADER=(-H "X-Orion-Notify-Token: ${API_TOKEN}")
fi

seed_payload() {
  local kind=$1
  local severity=$2
  curl -s -X POST "http://localhost:${NOTIFY_PORT}/notify" \
    -H "Content-Type: application/json" \
    "${AUTH_HEADER[@]}" \
    -d "{\"source_service\":\"smoke-test\",\"event_kind\":\"${kind}\",\"severity\":\"${severity}\",\"title\":\"Smoke ${kind}\",\"body_text\":\"Smoke body\",\"recipient_group\":\"juniper_primary\"}"
}

echo "Seeding notifications..."
seed_payload "smoke.one" "warning" | cat
echo
seed_payload "smoke.two" "error" | cat
echo

DIGEST_CONTAINER=$(docker compose -f "${DIGEST_DIR}/docker-compose.yml" ps -q notify-digest)
if [[ -z "${DIGEST_CONTAINER}" ]]; then
  echo "Digest container not running" >&2
  exit 1
fi

echo "Running digest on demand..."
docker exec -t "${DIGEST_CONTAINER}" python -m app.run_digest --window-hours 1

echo "Latest digest notification:"
curl -s "http://localhost:${NOTIFY_PORT}/notifications?limit=5" "${AUTH_HEADER[@]}" \
  | jq -r '.[] | select(.event_kind=="orion.digest.daily") | "\(.created_at) | \(.title) | \(.status)"' \
  | head -n 1

echo "Latest digest attempt:"
LATEST_DIGEST_ID=$(curl -s "http://localhost:${NOTIFY_PORT}/notifications?limit=5" "${AUTH_HEADER[@]}" \
  | jq -r '.[] | select(.event_kind=="orion.digest.daily") | .notification_id' \
  | head -n 1)

if [[ -n "${LATEST_DIGEST_ID}" ]]; then
  curl -s "http://localhost:${NOTIFY_PORT}/notifications/${LATEST_DIGEST_ID}/attempts" "${AUTH_HEADER[@]}" \
    | jq -r '.[] | "\(.attempted_at) | \(.status) | \(.channel)"' \
    | head -n 1
else
  echo "No digest notification found."
fi
