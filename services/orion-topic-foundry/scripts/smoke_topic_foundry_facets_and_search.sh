#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}
RUN_ID=${RUN_ID:-""}
LIMIT=${LIMIT:-20}
QUERY=${QUERY:-"test"}

if [[ -z "$RUN_ID" ]]; then
  echo "SKIP: RUN_ID is required" >&2
  exit 0
fi

facets=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/segments/facets?run_id=${RUN_ID}")
echo "$facets" | jq
echo "$facets" | jq -e '.aspects and .intents and .friction_buckets and .totals' >/dev/null

segments=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/segments?run_id=${RUN_ID}&q=${QUERY}&sort_by=friction&sort_dir=desc&limit=${LIMIT}&format=wrapped")
echo "$segments" | jq
count=$(echo "$segments" | jq -r '.items | length')
if [[ "$count" -gt "$LIMIT" ]]; then
  echo "Expected <= ${LIMIT} items, got ${count}" >&2
  exit 1
fi

echo "ok: facets + search/sort"
