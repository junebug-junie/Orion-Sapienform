#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-${TOPIC_FOUNDRY_BASE_URL:-http://127.0.0.1:8080/api/topic-foundry}}"
BASE_URL="${BASE_URL%/}"

datasets=$(curl -fsS "${BASE_URL}/datasets")
dataset=$(jq -c '.datasets[0] // empty' <<<"$datasets")
if [[ -z "$dataset" || "$dataset" == "null" ]]; then
  echo "FAIL: No datasets returned from /datasets" >&2
  exit 1
fi

run_preview() {
  local mode="$1"
  local extra="$2"
  payload=$(jq -c --argjson dataset "$dataset" --arg mode "$mode" --argjson extra "$extra" '{
    dataset: $dataset,
    windowing: ({windowing_mode:$mode, segmentation_mode:"time_gap", time_gap_seconds:900, max_window_seconds:7200, min_blocks_per_segment:1, max_chars:1200} + $extra),
    limit: 100
  }' <<<"{}")
  curl -fsS -H "Content-Type: application/json" -d "$payload" "${BASE_URL}/datasets/preview"
}

time_gap=$(run_preview "time_gap" "{}")
fixed_k=$(run_preview "fixed_k_rows" '{"fixed_k_rows":3}')

time_sizes=$(jq -r '.segments[]? | .row_ids_count // .size // empty' <<<"$time_gap" | sort -n | uniq | tr '\n' ' ')
fixed_sizes=$(jq -r '.segments[]? | .row_ids_count // .size // empty' <<<"$fixed_k" | sort -n | uniq | tr '\n' ' ')

if [[ -z "$time_sizes" ]]; then
  echo "FAIL: time_gap preview returned no segment sizes" >&2
  exit 1
fi
if [[ -z "$fixed_sizes" ]]; then
  echo "FAIL: fixed_k_rows preview returned no segment sizes" >&2
  exit 1
fi

echo "time_gap sizes: ${time_sizes}"
echo "fixed_k_rows sizes: ${fixed_sizes}"

echo "Topic Foundry windowing smoke checks passed."
