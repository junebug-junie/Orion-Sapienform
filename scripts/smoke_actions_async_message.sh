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
PREVIEW="[actions-daily-${TS}] Daily pulse summary"
FULL=$'## Orion — Daily Pulse\n\n- Theme: steady execution\n- Focus: finish async plumbing\n- Challenge: verify surfaces'

PAYLOAD=$(jq -n \
  --arg source "orion-actions" \
  --arg session "${ACTIONS_SESSION_ID:-orion_actions_async}" \
  --arg title "Orion — Daily Pulse" \
  --arg preview "${PREVIEW}" \
  --arg full "${FULL}" \
  '{source_service:$source,session_id:$session,title:$title,preview_text:$preview,full_text:$full,severity:"info",require_read_receipt:true,tags:["actions","daily","pulse"]}')

POST_RESP=$(curl -sS -X POST "http://localhost:${NOTIFY_PORT}/chat/message" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${PAYLOAD}")
MESSAGE_ID=$(echo "${POST_RESP}" | jq -r '.message_id // empty')

if [[ -z "${MESSAGE_ID}" ]]; then
  echo "FAIL: /chat/message did not return message_id"
  echo "Response: ${POST_RESP}"
  exit 1
fi

GET_RESP=$(curl -sS "http://localhost:${NOTIFY_PORT}/chat/messages?status=unread&limit=10" "${AUTH_HEADER[@]}")
FOUND=$(echo "${GET_RESP}" | jq -r --arg id "${MESSAGE_ID}" --arg p "${PREVIEW}" '.[] | select(.message_id == $id or .preview_text == $p) | .message_id' | head -n 1)

if [[ -z "${FOUND}" ]]; then
  echo "FAIL: unread message list missing actions-style async message"
  echo "message_id=${MESSAGE_ID}"
  echo "response=${GET_RESP}"
  exit 1
fi

echo "PASS: actions-style async message visible in /chat/messages"
echo "message_id=${MESSAGE_ID}"
echo "# natural producer path: orion-actions _execute_daily -> _publish_daily_outputs -> notify.chat_message"
