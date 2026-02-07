#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}
RUN_ID=${RUN_ID:-""}
LIMIT=${LIMIT:-50}
OFFSET=${OFFSET:-0}
PREDICATE=${PREDICATE:-""}
QUERY=${QUERY:-""}

if [[ -z "$RUN_ID" ]]; then
  RUN_ID=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/runs?limit=1" | jq -r '.runs[0].run_id')
fi

if [[ -z "$RUN_ID" || "$RUN_ID" == "null" ]]; then
  echo "No run_id available. Set RUN_ID or ensure runs exist." >&2
  exit 1
fi

url="${TOPIC_FOUNDRY_BASE_URL}/kg/edges?run_id=${RUN_ID}&limit=${LIMIT}&offset=${OFFSET}"
if [[ -n "$PREDICATE" ]]; then
  url="${url}&predicate=${PREDICATE}"
fi
if [[ -n "$QUERY" ]]; then
  url="${url}&q=${QUERY}"
fi

resp=$(curl -fsS "$url")
echo "$resp" | jq
