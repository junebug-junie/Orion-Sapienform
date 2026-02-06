#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}
LIMIT=${LIMIT:-20}
OFFSET=${OFFSET:-0}
KIND=${KIND:-""}

url="${TOPIC_FOUNDRY_BASE_URL}/events?limit=${LIMIT}&offset=${OFFSET}"
if [[ -n "$KIND" ]]; then
  url="${url}&kind=${KIND}"
fi

resp=$(curl -fsS "$url")
echo "$resp" | jq
