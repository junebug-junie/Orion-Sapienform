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

  sleep 3
fi

NOTIFY_PORT=$(grep -E '^PORT=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2)
NOTIFY_PORT=${NOTIFY_PORT:-7140}
API_TOKEN=$(grep -E '^API_TOKEN=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2-)
AUTH_HEADER=()
if [[ -n "${API_TOKEN}" ]]; then
  AUTH_HEADER=(-H "X-Orion-Notify-Token: ${API_TOKEN}")
fi

SESSION_ID="smoke-session-$(date +%s)"
MESSAGE_PAYLOAD=$(cat <<JSON
{
  "source_service":"smoke-test",
  "session_id":"${SESSION_ID}",
  "preview_text":"Smoke chat message",
  "full_text":"Smoke chat message full text",
  "severity":"info",
  "require_read_receipt": true
}
JSON
)

MESSAGE_ID=$(curl -s -X POST "http://localhost:${NOTIFY_PORT}/chat/message" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${MESSAGE_PAYLOAD}" \
  | jq -r '.message_id')

if [[ -z "${MESSAGE_ID}" || "${MESSAGE_ID}" == "null" ]]; then
  echo "FAIL: chat message did not return message_id"
  exit 1
fi

echo "Chat message created: ${MESSAGE_ID}"

echo "Polling hub notifications..."
for i in {1..10}; do
  FOUND=$(curl -s "http://localhost:8080/api/notifications?limit=10" \
    | jq -r --arg id "${MESSAGE_ID}" '.[] | select(.message_id==$id) | .message_id' \
    | head -n 1)
  if [[ -n "${FOUND}" ]]; then
    echo "PASS: chat message delivered to hub (${FOUND})"
    break
  fi
  sleep 1
done

RECEIPT_PAYLOAD=$(cat <<JSON
{"message_id":"${MESSAGE_ID}","session_id":"${SESSION_ID}","receipt_type":"opened"}
JSON
)

curl -s -X POST "http://localhost:${NOTIFY_PORT}/chat/message/${MESSAGE_ID}/receipt" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${RECEIPT_PAYLOAD}" \
  | cat

echo ""

OPENED_FOUND=$(curl -s "http://localhost:${NOTIFY_PORT}/chat/messages?status=seen&limit=10&session_id=${SESSION_ID}" \
  "${AUTH_HEADER[@]}" \
  | jq -r --arg id "${MESSAGE_ID}" '.[] | select(.message_id==$id) | .opened_at' \
  | head -n 1)

if [[ -z "${OPENED_FOUND}" || "${OPENED_FOUND}" == "null" ]]; then
  echo "FAIL: chat message not marked opened"
  exit 1
fi

echo "PASS: chat message receipt recorded"
