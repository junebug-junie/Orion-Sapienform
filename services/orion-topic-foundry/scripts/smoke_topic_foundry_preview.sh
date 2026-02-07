#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}
START_AT=${START_AT:-"$(date -u -d '7 days ago' +'%Y-%m-%dT%H:%M:%SZ')"}
END_AT=${END_AT:-"$(date -u +'%Y-%m-%dT%H:%M:%SZ')"}
LIMIT=${LIMIT:-200}
MIN_DOCS=${MIN_DOCS:-20}
MAX_CHARS=${MAX_CHARS:-6000}

SOURCE_TABLE=${SOURCE_TABLE:-chat_history_log}
ID_COLUMN=${ID_COLUMN:-chat_id}
TIME_COLUMN=${TIME_COLUMN:-created_at}
TEXT_COLUMNS=${TEXT_COLUMNS:-prompt,response}

payload=$(jq -n \
  --arg source_table "$SOURCE_TABLE" \
  --arg id_column "$ID_COLUMN" \
  --arg time_column "$TIME_COLUMN" \
  --arg text_columns "$TEXT_COLUMNS" \
  --arg start_at "$START_AT" \
  --arg end_at "$END_AT" \
  --argjson limit "$LIMIT" \
  --argjson max_chars "$MAX_CHARS" \
  '{
    dataset: {
      name: "preview",
      source_table: $source_table,
      id_column: $id_column,
      time_column: $time_column,
      text_columns: ($text_columns | split(",")),
      where_sql: null,
      where_params: null,
      timezone: "UTC"
    },
    windowing: {
      block_mode: "turn_pairs",
      include_roles: ["user", "assistant"],
      time_gap_seconds: 900,
      max_window_seconds: 7200,
      min_blocks_per_segment: 1,
      max_chars: $max_chars
    },
    start_at: $start_at,
    end_at: $end_at,
    limit: $limit
  }')

resp=$(curl -fsS -X POST "${TOPIC_FOUNDRY_BASE_URL}/datasets/preview" \
  -H 'Content-Type: application/json' \
  -d "$payload")

echo "$resp" | jq

DOCS=$(echo "$resp" | jq -r '.docs_generated')
MAX_CHARS_RESP=$(echo "$resp" | jq -r '.max_chars')

if [[ "$DOCS" -lt "$MIN_DOCS" ]]; then
  echo "docs_generated ${DOCS} is below MIN_DOCS ${MIN_DOCS}" >&2
  exit 1
fi

if [[ "$MAX_CHARS_RESP" -gt "$MAX_CHARS" ]]; then
  echo "max_chars ${MAX_CHARS_RESP} exceeds requested max_chars ${MAX_CHARS}" >&2
  exit 1
fi

echo "$resp" | jq -r '.samples[] | "sample: \(.snippet)"'
