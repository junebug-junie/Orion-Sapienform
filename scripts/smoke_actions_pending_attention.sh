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
SCHEDULE_ID="smoke-schedule-${TS}"
REASON="Workflow schedule needs attention"
MESSAGE="Workflow schedule ${SCHEDULE_ID} is overdue and requires acknowledgement"

CONTEXT=$(jq -n \
  --arg sid "${SCHEDULE_ID}" \
  --arg wf "smoke_workflow" \
  --arg corr "corr-${TS}" \
  '{source_service:"orion-actions",reason:"Workflow schedule needs attention",schedule_id:$sid,schedule_id_short:$sid,workflow_id:$wf,workflow_display_name:"Smoke Workflow",health:"failing",condition:"overdue",transition:"entered",state:"active",is_overdue:true,overdue_seconds:600,missed_run_count:2,needs_attention:true,event_kind:"workflow.schedule.attention.v1",tags:["workflow","schedule","attention"],correlation_id:$corr}')

PAYLOAD=$(jq -n \
  --arg source "orion-actions" \
  --arg reason "${REASON}" \
  --arg message "${MESSAGE}" \
  --argjson context "${CONTEXT}" \
  '{source_service:$source,reason:$reason,severity:"warning",message:$message,require_ack:true,context:$context}')

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

GET_RESP=$(curl -sS "http://localhost:${NOTIFY_PORT}/attention?status=pending&limit=10" "${AUTH_HEADER[@]}")
FOUND=$(echo "${GET_RESP}" | jq -r --arg id "${ATTENTION_ID}" --arg sid "${SCHEDULE_ID}" '.[] | select(.attention_id == $id or .context.schedule_id == $sid) | .attention_id' | head -n 1)

if [[ -z "${FOUND}" ]]; then
  echo "FAIL: pending attention list missing actions-style attention item"
  echo "attention_id=${ATTENTION_ID}"
  echo "response=${GET_RESP}"
  exit 1
fi

echo "PASS: actions-style pending attention visible in /attention"
echo "attention_id=${ATTENTION_ID}"
echo "# natural producer path: orion-actions _publish_workflow_attention_signal -> notify.attention_request (non-recovered transitions)"
