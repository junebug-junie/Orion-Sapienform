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
UNIQ="smoke-chat-${TS}"
PREVIEW_TEXT="[${UNIQ}] preview"
FULL_TEXT="[${UNIQ}] full body"

PAYLOAD=$(cat <<JSON
{
  "source_service": "smoke-test",
  "session_id": "smoke-session",
  "title": "[Orion] Chat message smoke",
  "preview_text": "${PREVIEW_TEXT}",
  "full_text": "${FULL_TEXT}",
  "severity": "info",
  "require_read_receipt": true
}
JSON
)

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

GET_RESP=$(curl -sS "http://localhost:${NOTIFY_PORT}/chat/messages?status=unread&limit=10" \
  "${AUTH_HEADER[@]}")

FOUND=$(echo "${GET_RESP}" | jq -r --arg id "${MESSAGE_ID}" --arg p "${PREVIEW_TEXT}" \
  '.[] | select(.message_id == $id or .preview_text == $p) | .message_id' | head -n 1)

if [[ -z "${FOUND}" ]]; then
  echo "FAIL: /chat/messages unread did not include smoke message"
  echo "message_id=${MESSAGE_ID}"
  echo "payload_preview=${PREVIEW_TEXT}"
  echo "response=${GET_RESP}"
  exit 1
fi

cat <<EOF
PASS: smoke notify chat message
message_id=${MESSAGE_ID}
preview_text=${PREVIEW_TEXT}
notify_endpoint=http://localhost:${NOTIFY_PORT}/chat/messages?status=unread&limit=10
# optional hub proxy (if hub is running):
# http://localhost:8080/api/chat/messages?status=unread&limit=10
EOF
