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
UNIQ="smoke-attention-${TS}"
REASON="[Orion] Attention smoke"
MESSAGE="[${UNIQ}] requires acknowledgement"

PAYLOAD=$(cat <<JSON
{
  "source_service": "smoke-test",
  "reason": "${REASON}",
  "severity": "warning",
  "message": "${MESSAGE}",
  "require_ack": true
}
JSON
)

POST_RESP=$(curl -sS -X POST "http://localhost:${NOTIFY_PORT}/attention/request" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${PAYLOAD}")
ATTENTION_ID=$(echo "${POST_RESP}" | jq -r '.attention_id // empty')

if [[ -z "${ATTENTION_ID}" ]]; then
  echo "FAIL: /attention/request did not return attention_id"
  echo "Response: ${POST_RESP}"
  exit 1
fi

GET_RESP=$(curl -sS "http://localhost:${NOTIFY_PORT}/attention?status=pending&limit=10" \
  "${AUTH_HEADER[@]}")

FOUND=$(echo "${GET_RESP}" | jq -r --arg id "${ATTENTION_ID}" --arg m "${MESSAGE}" --arg r "${REASON}" \
  '.[] | select(.attention_id == $id or .message == $m or .reason == $r) | .attention_id' | head -n 1)

if [[ -z "${FOUND}" ]]; then
  echo "FAIL: /attention pending did not include smoke request"
  echo "attention_id=${ATTENTION_ID}"
  echo "message=${MESSAGE}"
  echo "response=${GET_RESP}"
  exit 1
fi

cat <<EOF
PASS: smoke notify attention
attention_id=${ATTENTION_ID}
reason=${REASON}
notify_endpoint=http://localhost:${NOTIFY_PORT}/attention?status=pending&limit=10
# optional hub proxy (if hub is running):
# http://localhost:8080/api/attention?status=pending&limit=10
EOF
