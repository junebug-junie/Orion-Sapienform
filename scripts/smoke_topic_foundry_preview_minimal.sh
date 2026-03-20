#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8030}
DATASET_ID=${DATASET_ID:-}

if [[ -z "$DATASET_ID" ]]; then
  echo "DATASET_ID is required." >&2
  exit 1
fi

payload=$(cat <<JSON
{
  "dataset_id": "${DATASET_ID}",
  "windowing": {
    "block_mode": "turn_pairs",
    "time_gap_seconds": 900,
    "max_chars": 6000
  },
  "limit": 5
}
JSON
)

status=$(curl -s -o /tmp/topic_foundry_preview.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$payload" \
  "${BASE_URL}/datasets/preview")

if [[ "$status" != 2* ]]; then
  echo "Unexpected status from /datasets/preview: ${status}" >&2
  cat /tmp/topic_foundry_preview.json >&2 || true
  exit 1
fi

echo "Preview request succeeded."
