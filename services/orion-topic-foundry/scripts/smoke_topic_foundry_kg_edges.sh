#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}
MIN_EDGES=${MIN_EDGES:-5}

run_id=${RUN_ID:-""}
if [[ -z "$run_id" ]]; then
  run_id=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/runs?limit=1" | jq -r '.runs[0].run_id')
fi

if [[ -z "$run_id" || "$run_id" == "null" ]]; then
  echo "No run_id available. Set RUN_ID or ensure runs exist." >&2
  exit 1
fi

curl -fsS -X POST "${TOPIC_FOUNDRY_BASE_URL}/runs/${run_id}/enrich" \
  -H 'Content-Type: application/json' \
  -d '{"limit": 50, "force": false}' | jq

sleep 2

edges_resp=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/edges?run_id=${run_id}&limit=200")
echo "$edges_resp" | jq

edge_count=$(echo "$edges_resp" | jq -r '.edges | length')
if [[ "$edge_count" -lt "$MIN_EDGES" ]]; then
  echo "edge_count ${edge_count} is below MIN_EDGES ${MIN_EDGES}" >&2
  exit 1
fi

echo "KG edges generated: ${edge_count}"
