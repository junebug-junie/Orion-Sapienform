#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}

runs=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/runs?limit=5&format=wrapped")
echo "$runs" | jq

echo "$runs" | jq -e '
  (.items | type == "array")
  and (.limit | type == "number")
  and (.offset | type == "number")
' >/dev/null

run_id=$(echo "$runs" | jq -r '.items[0].run_id // empty')
if [[ -z "$run_id" ]]; then
  echo "No runs found; skipping segment paging assertion."
  exit 0
fi

segments=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/segments?run_id=${run_id}&limit=10&offset=0&include_snippet=true&format=wrapped")
echo "$segments" | jq

segment_count=$(echo "$segments" | jq '.items | length')
if [[ "$segment_count" -gt 10 ]]; then
  echo "Expected <=10 segments, got ${segment_count}" >&2
  exit 1
fi

echo "$segments" | jq -e '
  (.items | type == "array")
  and (.limit | type == "number")
  and (.offset | type == "number")
' >/dev/null
