#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-https://athena.tail348bbe.ts.net/api/topic-foundry}"
BASE_URL="${BASE_URL%/}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "FAIL: missing dependency '$1'" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd jq

AUTH_HEADER=()
if [[ -n "${TOPIC_FOUNDRY_AUTH_HEADER:-}" ]]; then
  AUTH_HEADER=(-H "${TOPIC_FOUNDRY_AUTH_HEADER}")
fi

curl_api() {
  curl -fsS "${AUTH_HEADER[@]}" "$@"
}

echo "==> Checking introspection endpoints"
curl_api "${BASE_URL}/introspect/schemas" | jq -r '.schemas | length'
curl_api "${BASE_URL}/introspect/tables?schema=public" | jq -r '.tables | length'

echo "==> Loading datasets"
datasets_json=$(curl_api "${BASE_URL}/datasets")
dataset_id=$(echo "$datasets_json" | jq -r '.datasets[0].dataset_id')
if [[ -z "$dataset_id" || "$dataset_id" == "null" ]]; then
  echo "FAIL: No datasets returned from ${BASE_URL}/datasets" >&2
  exit 1
fi

source_table=$(echo "$datasets_json" | jq -r '.datasets[0].source_table')
id_column=$(echo "$datasets_json" | jq -r '.datasets[0].id_column')
time_column=$(echo "$datasets_json" | jq -r '.datasets[0].time_column')
text_columns=$(echo "$datasets_json" | jq -c '.datasets[0].text_columns')

schema="public"
table="$source_table"
if [[ "$source_table" == *.* ]]; then
  schema="${source_table%%.*}"
  table="${source_table##*.}"
fi

echo "==> Resolving columns for ${schema}.${table}"
columns_json=$(curl_api "${BASE_URL}/introspect/columns?schema=${schema}&table=${table}")
boundary_column=$(echo "$columns_json" | jq -r --arg id_col "$id_column" '.columns[] | select(.column_name == $id_col) | .column_name' | head -n 1)
if [[ -z "$boundary_column" ]]; then
  boundary_column=$(echo "$columns_json" | jq -r '.columns[0].column_name')
fi

echo "==> Preview (turn_pairs)"
preview_payload=$(jq -n \
  --arg dataset_id "$dataset_id" \
  '{"dataset_id": $dataset_id, "windowing_spec": {"windowing_mode": "turn_pairs", "block_mode": "turn_pairs", "max_chars": 6000, "min_blocks_per_segment": 1}}')
curl_api -X POST "${BASE_URL}/datasets/preview" \
  -H "Content-Type: application/json" \
  -d "$preview_payload" | jq -r '.segments | length'

echo "==> Update dataset boundary_column=${boundary_column}"
patch_payload=$(jq -n --arg boundary "$boundary_column" '{"boundary_column": $boundary}')
curl_api -X PATCH "${BASE_URL}/datasets/${dataset_id}" \
  -H "Content-Type: application/json" \
  -d "$patch_payload" | jq -r '.boundary_column'

echo "==> Preview (conversation_bound)"
preview_cb_payload=$(jq -n \
  --arg dataset_id "$dataset_id" \
  --arg boundary "$boundary_column" \
  '{"dataset_id": $dataset_id, "windowing_spec": {"windowing_mode": "conversation_bound", "block_mode": "rows", "boundary_column": $boundary, "max_chars": 6000, "min_blocks_per_segment": 1}}')
curl_api -X POST "${BASE_URL}/datasets/preview" \
  -H "Content-Type: application/json" \
  -d "$preview_cb_payload" | jq -r '.segments | length'

echo "==> Ensure model"
models_json=$(curl_api "${BASE_URL}/models")
model_id=$(echo "$models_json" | jq -r --arg dataset_id "$dataset_id" '.models[] | select(.dataset_id == $dataset_id) | .model_id' | head -n 1)
if [[ -z "$model_id" ]]; then
  model_payload=$(jq -n \
    --arg name "smoke-model" \
    --arg version "smoke" \
    --arg dataset_id "$dataset_id" \
    '{"name": $name, "version": $version, "stage": "candidate", "dataset_id": $dataset_id, "model_spec": {"algorithm": "hdbscan", "min_cluster_size": 10, "metric": "cosine", "params": {}}, "windowing_spec": {"windowing_mode": "turn_pairs", "block_mode": "turn_pairs", "max_chars": 6000, "min_blocks_per_segment": 1}, "metadata": {}}')
  model_id=$(curl_api -X POST "${BASE_URL}/models" \
    -H "Content-Type: application/json" \
    -d "$model_payload" | jq -r '.model_id')
fi

echo "==> Train run"
train_payload=$(jq -n \
  --arg model_id "$model_id" \
  --arg dataset_id "$dataset_id" \
  '{"model_id": $model_id, "dataset_id": $dataset_id, "windowing_spec": {"windowing_mode": "turn_pairs", "block_mode": "turn_pairs", "max_chars": 6000, "min_blocks_per_segment": 1}}')
run_id=$(curl_api -X POST "${BASE_URL}/runs/train" \
  -H "Content-Type: application/json" \
  -d "$train_payload" | jq -r '.run_id')

echo "==> Poll run ${run_id}"
status="queued"
for _ in {1..20}; do
  status=$(curl_api "${BASE_URL}/runs/${run_id}" | jq -r '.status')
  echo "status=${status}"
  if [[ "$status" == "complete" || "$status" == "failed" ]]; then
    break
  fi
  sleep 3
done

echo "==> Load segments"
segments_json=$(curl_api "${BASE_URL}/segments?run_id=${run_id}&limit=5&include_snippet=true")
segment_id=$(echo "$segments_json" | jq -r '.segments[0].segment_id')
segment_count=$(echo "$segments_json" | jq -r '.segments | length')
if [[ "$segment_count" -eq 0 ]]; then
  echo "FAIL: No segments returned for run_id=${run_id}" >&2
  exit 1
fi

echo "==> Facets"
curl_api "${BASE_URL}/segments/facets?run_id=${run_id}" | jq -r '.totals.segments'

echo "==> Segment full text"
curl_api "${BASE_URL}/segments/${segment_id}?include_full_text=true" | jq -r '.full_text | length'

echo "Topic Foundry remote smoke checks passed."
