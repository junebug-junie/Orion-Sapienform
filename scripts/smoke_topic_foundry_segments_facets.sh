#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8030}
RUN_ID=${RUN_ID:-}

if [[ -z "$RUN_ID" ]]; then
  echo "RUN_ID is required." >&2
  exit 1
fi

status=$(curl -s -o /tmp/topic_foundry_facets.json -w "%{http_code}" \
  "${BASE_URL}/segments/facets?run_id=${RUN_ID}")

if [[ "$status" != 2* ]]; then
  echo "Unexpected status from /segments/facets: ${status}" >&2
  cat /tmp/topic_foundry_facets.json >&2 || true
  exit 1
fi

echo "Segments facets request succeeded."
