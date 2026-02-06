#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}
MODEL_NAME=${MODEL_NAME:-"default"}
LIMIT=${LIMIT:-5}

payload=$(jq -n \
  --arg model_name "$MODEL_NAME" \
  '{
    model_name: $model_name,
    window_hours: 24,
    threshold_js: 0.0,
    threshold_outlier: 0.0
  }')

resp=$(curl -fsS -X POST "${TOPIC_FOUNDRY_BASE_URL}/drift/run" \
  -H 'Content-Type: application/json' \
  -d "$payload")

echo "$resp" | jq

drift_id=$(echo "$resp" | jq -r '.drift_id')
status=$(echo "$resp" | jq -r '.status')

if [[ -z "$drift_id" || "$drift_id" == "null" ]]; then
  echo "drift_id missing" >&2
  exit 1
fi

if [[ "$status" == "error" ]]; then
  echo "drift run failed" >&2
  exit 1
fi

list_resp=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/drift?model_name=${MODEL_NAME}&limit=${LIMIT}")
echo "$list_resp" | jq

found=$(echo "$list_resp" | jq -r --arg drift_id "$drift_id" '.records[] | select(.drift_id == $drift_id) | .drift_id' | head -n1 || true)

if [[ -z "$found" ]]; then
  echo "drift record not found for ${drift_id}" >&2
  exit 1
fi

echo "Drift record ${found} verified."
