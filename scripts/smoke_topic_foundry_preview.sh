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

payload=$(jq -c --argjson dataset "$dataset" '{
  dataset: $dataset,
  windowing: {
    windowing_mode: "time_gap",
    segmentation_mode: "time_gap",
    time_gap_seconds: 900,
    max_window_seconds: 7200,
    min_blocks_per_segment: 1,
    max_chars: 1200
  },
  limit: 50
}' <<<"{}")

status=$(curl -s -o /tmp/topic_foundry_preview.json -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$payload" \
  "${BASE_URL}/datasets/preview")
if [[ "$status" != 2* ]]; then
  echo "FAIL: Preview request failed (status ${status})" >&2
  cat /tmp/topic_foundry_preview.json >&2 || true
  exit 1
fi

echo "Topic Foundry preview smoke checks passed."
