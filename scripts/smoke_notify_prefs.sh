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

PROFILE_PAYLOAD=$(cat <<JSON
{
  "display_name": "Juniper",
  "timezone": "America/Denver",
  "quiet_hours_enabled": true,
  "quiet_start_local": "21:00",
  "quiet_end_local": "07:00"
}
JSON
)

curl -s -X PUT "http://localhost:${NOTIFY_PORT}/recipients/juniper_primary" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${PROFILE_PAYLOAD}" \
  | cat

echo ""

PREF_PAYLOAD=$(cat <<JSON
{
  "preferences": [
    {
      "recipient_group": "juniper_primary",
      "scope_type": "event_kind",
      "scope_value": "orion.chat.message",
      "channels_enabled": ["in_app"],
      "escalation_enabled": true,
      "escalation_delay_minutes": 30
    }
  ]
}
JSON
)

curl -s -X PUT "http://localhost:${NOTIFY_PORT}/recipients/juniper_primary/preferences" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${PREF_PAYLOAD}" \
  | cat

echo ""

SESSION_ID="pref-smoke-$(date +%s)"
MESSAGE_PAYLOAD=$(cat <<JSON
{
  "source_service": "smoke-test",
  "session_id": "${SESSION_ID}",
  "preview_text": "Preference smoke message",
  "severity": "info",
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
  echo "FAIL: chat message did not return message_id" >&2
  exit 1
fi

NOTIFICATION_ID=$(curl -s "http://localhost:${NOTIFY_PORT}/notifications?limit=1&event_kind=orion.chat.message" \
  "${AUTH_HEADER[@]}" \
  | jq -r '.[0].notification_id')

if [[ -z "${NOTIFICATION_ID}" || "${NOTIFICATION_ID}" == "null" ]]; then
  echo "FAIL: notification not found" >&2
  exit 1
fi

CHANNELS_FINAL=$(curl -s "http://localhost:${NOTIFY_PORT}/notifications/${NOTIFICATION_ID}" \
  "${AUTH_HEADER[@]}" \
  | jq -r '.context.policy_breakdown.channels_final | join(",")')

if [[ "${CHANNELS_FINAL}" != "in_app" ]]; then
  echo "FAIL: expected channels_final=in_app, got ${CHANNELS_FINAL}" >&2
  exit 1
fi

EMAIL_ATTEMPT=$(curl -s "http://localhost:${NOTIFY_PORT}/notifications/${NOTIFICATION_ID}/attempts" \
  "${AUTH_HEADER[@]}" \
  | jq -r '.[] | select(.channel=="email") | .channel' \
  | head -n 1)

if [[ -n "${EMAIL_ATTEMPT}" ]]; then
  echo "FAIL: email attempt found despite preferences" >&2
  exit 1
fi

echo "PASS: channels_final applied and email skipped"

RESOLVE_PAYLOAD=$(cat <<JSON
{"recipient_group":"juniper_primary","event_kind":"orion.chat.message","severity":"info"}
JSON
)

curl -s -X POST "http://localhost:${NOTIFY_PORT}/preferences/resolve" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${RESOLVE_PAYLOAD}" \
  | jq '.source_breakdown'
