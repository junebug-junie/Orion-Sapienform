#!/usr/bin/env bash
set -euo pipefail

resolve_base_url() {
  local cli_base_url="${1:-}"
  if [[ -n "$cli_base_url" ]]; then
    echo "$cli_base_url"
    return
  fi
  if [[ -n "${TOPIC_FOUNDRY_BASE_URL:-}" ]]; then
    echo "$TOPIC_FOUNDRY_BASE_URL"
    return
  fi
  if [[ -n "${HUB_BASE_URL:-}" ]]; then
    echo "${HUB_BASE_URL%/}/api/topic-foundry"
    return
  fi
  echo "http://127.0.0.1:8080/api/topic-foundry"
}

BASE_URL="$(resolve_base_url "${1:-}")"
BASE_URL="${BASE_URL%/}"

datasets=$(curl -fsS "${BASE_URL}/datasets")
dataset=$(jq -c '.datasets[0] // empty' <<<"$datasets")
if [[ -z "$dataset" || "$dataset" == "null" ]]; then
  echo "FAIL: No datasets returned from /datasets" >&2
  exit 1
fi

dataset_id=$(jq -r '.dataset_id' <<<"$dataset")
source_table=$(jq -r '.source_table' <<<"$dataset")
boundary_column=$(jq -r '.boundary_column // empty' <<<"$dataset")
schema="public"
table="$source_table"
if [[ "$source_table" == *.* ]]; then
  schema="${source_table%%.*}"
  table="${source_table##*.}"
fi

if [[ -z "$boundary_column" ]]; then
  columns=$(curl -fsS "${BASE_URL}/introspect/columns?schema=${schema}&table=${table}")
  boundary_column=$(jq -r '.columns[0].column_name // empty' <<<"$columns")
  if [[ -z "$boundary_column" ]]; then
    echo "FAIL: No boundary column candidates found for ${schema}.${table}" >&2
    exit 1
  fi
  patch_payload=$(jq -c --arg boundary_column "$boundary_column" '{"boundary_column": $boundary_column, "boundary_strategy": "column"}' <<<"{}")
  patch_status=$(curl -s -o /tmp/topic_foundry_preview_patch.json -w "%{http_code}" \
    -X PATCH \
    -H "Content-Type: application/json" \
    -d "$patch_payload" \
    "${BASE_URL}/datasets/${dataset_id}")
  if [[ "$patch_status" != 2* ]]; then
    echo "FAIL: Dataset patch failed (status ${patch_status})" >&2
    cat /tmp/topic_foundry_preview_patch.json >&2 || true
    exit 1
  fi
fi

payload=$(jq -c --arg dataset_id "$dataset_id" --arg boundary_column "$boundary_column" '{
  dataset_id: $dataset_id,
  windowing: {
    windowing_mode: "conversation_bound",
    boundary_column: $boundary_column,
    segmentation_mode: "time_gap",
    time_gap_seconds: 900,
    max_window_seconds: 7200,
    min_blocks_per_segment: 1,
    max_chars: 1200
  },
  limit: 100
}' <<<"{}")

status=$(curl -s -o /tmp/topic_foundry_preview_bound.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$payload" \
  "${BASE_URL}/datasets/preview")
if [[ "$status" != 2* ]]; then
  echo "FAIL: conversation_bound preview failed (status ${status})" >&2
  cat /tmp/topic_foundry_preview_bound.json >&2 || true
  exit 1
fi

echo "Topic Foundry conversation_bound preview smoke checks passed."
