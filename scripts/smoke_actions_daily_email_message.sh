#!/usr/bin/env bash
set -euo pipefail

NOTIFY_ENV="services/orion-notify/.env"
if [[ ! -f "${NOTIFY_ENV}" ]]; then
  echo "FAIL: missing ${NOTIFY_ENV}"
  exit 1
fi

NOTIFY_PORT=$(grep -E '^PORT=' "${NOTIFY_ENV}" | cut -d '=' -f2)
NOTIFY_PORT=${NOTIFY_PORT:-7140}
API_TOKEN=$(grep -E '^API_TOKEN=' "${NOTIFY_ENV}" | cut -d '=' -f2- || true)
AUTH_HEADER=()
if [[ -n "${API_TOKEN}" ]]; then
  AUTH_HEADER=(-H "X-Orion-Notify-Token: ${API_TOKEN}")
fi

TS=$(date -u +%Y%m%dT%H%M%SZ)
PREVIEW="[actions-daily-email-${TS}] Daily pulse summary"
FULL=$'## Orion — Daily Pulse\n\n- Focus: async-output closure\n- Delivery: Hub Message + email requested'

CHAT_PAYLOAD=$(jq -n \
  --arg source "orion-actions" \
  --arg session "${ACTIONS_SESSION_ID:-collapse_mirror}" \
  --arg title "Orion — Daily Pulse" \
  --arg preview "${PREVIEW}" \
  --arg full "${FULL}" \
  '{source_service:$source,session_id:$session,title:$title,preview_text:$preview,full_text:$full,severity:"info",require_read_receipt:true,tags:["actions","daily","pulse"]}')

CHAT_RESP=$(curl -sS -X POST "http://localhost:${NOTIFY_PORT}/chat/message" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${CHAT_PAYLOAD}")
MESSAGE_ID=$(echo "${CHAT_RESP}" | jq -r '.message_id // empty')
if [[ -z "${MESSAGE_ID}" ]]; then
  echo "FAIL: /chat/message did not return message_id"
  echo "Response: ${CHAT_RESP}"
  exit 1
fi

NOTIFY_PAYLOAD=$(jq -n \
  --arg source "orion-actions" \
  --arg title "Orion — Daily Pulse" \
  --arg body "${PREVIEW}" \
  '{source_service:$source,event_kind:"orion.daily.pulse",severity:"info",title:$title,body_text:$body,recipient_group:"juniper_primary",channels_requested:["email"],tags:["actions","daily","pulse"]}')

NOTIFY_RESP=$(curl -sS -X POST "http://localhost:${NOTIFY_PORT}/notify" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${NOTIFY_PAYLOAD}")
NOTIFY_ID=$(echo "${NOTIFY_RESP}" | jq -r '.notification_id // empty')
if [[ -z "${NOTIFY_ID}" ]]; then
  echo "FAIL: /notify did not return notification_id"
  echo "Response: ${NOTIFY_RESP}"
  exit 1
fi

echo "PASS: daily-style chat message and notify(email channel requested) accepted"
echo "message_id=${MESSAGE_ID}"
echo "notification_id=${NOTIFY_ID}"
echo "# natural producer path: orion-actions _execute_daily -> _publish_daily_outputs"
