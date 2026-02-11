#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${TOPIC_FOUNDRY_BASE_URL:-http://localhost:8615}"
DATASET_ID="${1:-${TOPIC_FOUNDRY_DATASET_ID:-}}"
if [[ -z "$DATASET_ID" ]]; then
  echo "usage: $0 <dataset_id>"
  exit 1
fi

payload=$(cat <<JSON
{
  "dataset_id": "$DATASET_ID",
  "windowing": {
    "windowing_mode": "document",
    "time_gap_minutes": 15,
    "max_chars": 6000
  },
  "limit": 200
}
JSON
)

resp=$(curl -fsS -X POST "$BASE_URL/datasets/preview" -H 'content-type: application/json' -d "$payload")
echo "$resp" | jq '{doc_count,segments_generated,avg_chars,p95_chars,max_chars,samples:(.samples|length)}'
echo "$resp" | jq -e '.doc_count >= 0 and .segments_generated >= 0 and (.samples|type=="array")' >/dev/null
echo "preview smoke passed"
