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

ATTN_PAYLOAD=$(cat <<JSON
{
  "source_service":"smoke-test",
  "reason":"I want to talk",
  "severity":"warning",
  "message":"Smoke attention request",
  "require_ack": true
}
JSON
)

ATTENTION_ID=$(curl -s -X POST "http://localhost:${NOTIFY_PORT}/attention/request" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${ATTN_PAYLOAD}" \
  | jq -r '.attention_id')

if [[ -z "${ATTENTION_ID}" || "${ATTENTION_ID}" == "null" ]]; then
  echo "FAIL: attention request did not return attention_id"
  exit 1
fi

echo "Attention request created: ${ATTENTION_ID}"

echo "Polling hub pending attention..."
for i in {1..10}; do
  FOUND=$(curl -s "http://localhost:8080/api/attention?status=pending&limit=10" \
    | jq -r --arg id "${ATTENTION_ID}" '.[] | select(.attention_id==$id) | .attention_id' \
    | head -n 1)
  if [[ -n "${FOUND}" ]]; then
    echo "PASS: attention item visible in hub"
    break
  fi
  sleep 1
done

ACK_PAYLOAD=$(cat <<JSON
{"attention_id":"${ATTENTION_ID}","ack_type":"dismissed","note":"Smoke test dismiss"}
JSON
)

curl -s -X POST "http://localhost:${NOTIFY_PORT}/attention/${ATTENTION_ID}/ack" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${ACK_PAYLOAD}" \
  | cat

echo ""

ACKED_FOUND=$(curl -s "http://localhost:${NOTIFY_PORT}/attention?status=acked&limit=10" \
  "${AUTH_HEADER[@]}" \
  | jq -r --arg id "${ATTENTION_ID}" '.[] | select(.attention_id==$id) | .attention_id' \
  | head -n 1)

if [[ -z "${ACKED_FOUND}" ]]; then
  echo "FAIL: attention not marked acked"
  exit 1
fi

echo "PASS: attention acked in notify"

SMTP_HOST=$(grep -E '^NOTIFY_EMAIL_SMTP_HOST=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2-)
SMTP_TO=$(grep -E '^NOTIFY_EMAIL_TO=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2-)
if [[ -n "${SMTP_HOST}" && -n "${SMTP_TO}" ]]; then
  echo "Optional: testing escalation (requires email configured)..."
  ESCALATE_PAYLOAD=$(cat <<JSON
{
  "source_service":"smoke-test",
  "reason":"Escalation test",
  "severity":"warning",
  "message":"Escalate immediately",
  "require_ack": true,
  "context": {"ack_deadline_minutes": 0}
}
JSON
)
  curl -s -X POST "http://localhost:${NOTIFY_PORT}/attention/request" \
    -H "Content-Type: application/json" \
    "${AUTH_HEADER[@]}" \
    -d "${ESCALATE_PAYLOAD}" \
    | cat
  echo ""
fi
