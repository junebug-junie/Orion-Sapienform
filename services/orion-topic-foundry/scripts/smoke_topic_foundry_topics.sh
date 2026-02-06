#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}
RUN_ID=${RUN_ID:-""}
LIMIT=${LIMIT:-20}

if [[ -z "$RUN_ID" ]]; then
  echo "SKIP: RUN_ID is required" >&2
  exit 0
fi

topics=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/topics?run_id=${RUN_ID}&limit=${LIMIT}")
echo "$topics" | jq

count=$(echo "$topics" | jq -r '.items | length')
if [[ "$count" -le 0 ]]; then
  echo "Expected topics > 0" >&2
  exit 1
fi

topic_id=$(echo "$topics" | jq -r '.items[0].topic_id')
segments=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/topics/${topic_id}/segments?run_id=${RUN_ID}&limit=5")
echo "$segments" | jq
echo "$segments" | jq -e '.items' >/dev/null

echo "ok: topics + topic segments"
