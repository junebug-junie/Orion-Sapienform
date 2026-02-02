#!/usr/bin/env bash
set -euo pipefail

BUS_DIR="services/orion-bus"
NOTIFY_DIR="services/orion-notify"
HUB_DIR="services/orion-hub"

for dir in "${NOTIFY_DIR}" "${HUB_DIR}"; do
  if [[ ! -f "${dir}/.env" ]]; then
    echo "Missing ${dir}/.env. Copy ${dir}/.env_example to ${dir}/.env and edit settings." >&2
    exit 1
  fi
  if ! grep -q '^ORION_BUS_URL=' "${dir}/.env"; then
    echo "ORION_BUS_URL=redis://localhost:6379/0" >> "${dir}/.env"
  fi
  if ! grep -q '^ORION_BUS_ENABLED=' "${dir}/.env"; then
    echo "ORION_BUS_ENABLED=true" >> "${dir}/.env"
  fi
  if ! grep -q '^ORION_BUS_ENFORCE_CATALOG=' "${dir}/.env"; then
    echo "ORION_BUS_ENFORCE_CATALOG=true" >> "${dir}/.env"
  fi
done

PROJECT=${PROJECT:-orion-smoke}
TELEMETRY_ROOT=${TELEMETRY_ROOT:-/tmp/orion-telemetry}
mkdir -p "${TELEMETRY_ROOT}/${PROJECT}/bus/data"

if [[ -z "${SKIP_COMPOSE:-}" ]]; then
  PROJECT="${PROJECT}" TELEMETRY_ROOT="${TELEMETRY_ROOT}" \
    docker compose -f "${BUS_DIR}/docker-compose.yml" up -d

  docker compose -f "${NOTIFY_DIR}/docker-compose.yml" up -d --build

  docker compose --env-file "${HUB_DIR}/.env" -f "${HUB_DIR}/docker-compose.yml" up -d --build
fi

wait_for_health() {
  local url=$1
  local name=$2
  local retries=20
  local delay=1
  for _ in $(seq 1 ${retries}); do
    if curl -fsS "${url}" >/dev/null; then
      echo "PASS: ${name} healthy"
      return 0
    fi
    sleep ${delay}
    delay=$((delay + 1))
  done
  echo "FAIL: ${name} health check failed (${url})" >&2
  return 1
}

wait_for_health "http://localhost:7140/health" "notify"
wait_for_health "http://localhost:8080/health" "hub"

NOTIFY_PORT=$(grep -E '^PORT=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2)
NOTIFY_PORT=${NOTIFY_PORT:-7140}
API_TOKEN=$(grep -E '^API_TOKEN=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2-)
AUTH_HEADER=()
if [[ -n "${API_TOKEN}" ]]; then
  AUTH_HEADER=(-H "X-Orion-Notify-Token: ${API_TOKEN}")
fi

PRESENCE=$(curl -s "http://localhost:8080/api/presence" | jq -r '.active')
if [[ "${PRESENCE}" != "true" ]]; then
  echo "WARN: Hub presence inactive; continuing with poll fallback."
fi

SESSION_ID=$(python - <<'PY'
import uuid
print(uuid.uuid4())
PY
)

MESSAGE_PAYLOAD=$(cat <<JSON
{
  "source_service": "smoke-test",
  "session_id": "${SESSION_ID}",
  "preview_text": "Smoke chat message for hub delivery",
  "full_text": "Smoke chat message full text",
  "severity": "info",
  "require_read_receipt": true,
  "tags": ["chat", "message", "smoke"]
}
JSON
)

MESSAGE_RESP=$(curl -s -X POST "http://localhost:${NOTIFY_PORT}/chat/message" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${MESSAGE_PAYLOAD}")

MESSAGE_ID=$(echo "${MESSAGE_RESP}" | jq -r '.message_id')
NOTIFICATION_ID=$(echo "${MESSAGE_RESP}" | jq -r '.notification_id')

if [[ -z "${MESSAGE_ID}" || "${MESSAGE_ID}" == "null" ]]; then
  echo "FAIL: chat message did not return message_id" >&2
  echo "Response: ${MESSAGE_RESP}" >&2
  exit 1
fi

FOUND=""
LAST_NOTIFICATIONS=""
for _ in {1..20}; do
  LAST_NOTIFICATIONS=$(curl -s "http://localhost:8080/api/notifications?limit=50")
  FOUND=$(echo "${LAST_NOTIFICATIONS}" | jq -r --arg mid "${MESSAGE_ID}" --arg sid "${SESSION_ID}" '.[] | select(.message_id==$mid or .session_id==$sid) | .message_id' | head -n 1)
  if [[ -n "${FOUND}" && "${FOUND}" != "null" ]]; then
    break
  fi
  sleep 1
 done

if [[ -z "${FOUND}" || "${FOUND}" == "null" ]]; then
  echo "WARN: hub notification not found; verifying notify persistence instead."
  NOTIFY_FOUND=$(curl -s "http://localhost:${NOTIFY_PORT}/chat/messages?status=unread&limit=50&session_id=${SESSION_ID}" \
    "${AUTH_HEADER[@]}" | jq -r --arg mid "${MESSAGE_ID}" '.[] | select(.message_id==$mid) | .message_id' | head -n 1)
  if [[ -z "${NOTIFY_FOUND}" || "${NOTIFY_FOUND}" == "null" ]]; then
    echo "FAIL: message missing from hub and notify persistence" >&2
    echo "Last hub notifications: ${LAST_NOTIFICATIONS}" >&2
    exit 1
  fi
else
  EVENT_KIND=$(echo "${LAST_NOTIFICATIONS}" | jq -r --arg mid "${MESSAGE_ID}" '.[] | select(.message_id==$mid) | .event_kind' | head -n 1)
  SEVERITY=$(echo "${LAST_NOTIFICATIONS}" | jq -r --arg mid "${MESSAGE_ID}" '.[] | select(.message_id==$mid) | .severity' | head -n 1)
  SESSION_FOUND=$(echo "${LAST_NOTIFICATIONS}" | jq -r --arg mid "${MESSAGE_ID}" '.[] | select(.message_id==$mid) | .session_id' | head -n 1)
  if [[ "${EVENT_KIND}" != "orion.chat.message" ]]; then
    echo "FAIL: event_kind mismatch (${EVENT_KIND})" >&2
    exit 1
  fi
  if [[ -z "${SEVERITY}" || "${SEVERITY}" == "null" ]]; then
    echo "FAIL: severity missing in hub notification" >&2
    exit 1
  fi
  if [[ "${SESSION_FOUND}" != "${SESSION_ID}" ]]; then
    echo "FAIL: session_id mismatch (${SESSION_FOUND})" >&2
    exit 1
  fi
  echo "PASS: hub notification received"
fi

RECEIPT_PAYLOAD=$(cat <<JSON
{
  "message_id": "${MESSAGE_ID}",
  "session_id": "${SESSION_ID}",
  "receipt_type": "opened"
}
JSON
)

curl -s -X POST "http://localhost:${NOTIFY_PORT}/chat/message/${MESSAGE_ID}/receipt" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${RECEIPT_PAYLOAD}" \
  | cat

echo ""

OPENED_AT=""
for _ in {1..10}; do
  OPENED_AT=$(curl -s "http://localhost:${NOTIFY_PORT}/chat/messages?status=seen&limit=50&session_id=${SESSION_ID}" \
    "${AUTH_HEADER[@]}" | jq -r --arg mid "${MESSAGE_ID}" '.[] | select(.message_id==$mid) | .opened_at' | head -n 1)
  if [[ -n "${OPENED_AT}" && "${OPENED_AT}" != "null" ]]; then
    break
  fi
  sleep 1
 done

if [[ -z "${OPENED_AT}" || "${OPENED_AT}" == "null" ]]; then
  echo "FAIL: opened_at not recorded" >&2
  exit 1
fi

cat <<SUMMARY
PASS: chat message smoke test complete
session_id=${SESSION_ID}
message_id=${MESSAGE_ID}
notification_id=${NOTIFICATION_ID}
opened_at=${OPENED_AT}
SUMMARY
