#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${TOPIC_FOUNDRY_BASE_URL:-http://127.0.0.1:8615}"

runs_resp="$(curl -fsS "$BASE_URL/runs")"
run_id="$(echo "$runs_resp" | jq -r '.runs[0].run_id // empty')"
if [[ -z "$run_id" ]]; then
  echo "[FAIL] no run_id from /runs"
  exit 1
fi

list_resp="$(curl -fsS "$BASE_URL/runs/$run_id/segments?limit=5")"
segment_id="$(echo "$list_resp" | jq -r '.items[0].segment_id // empty')"
preview_len="$(echo "$list_resp" | jq -r '.items[0].text_preview | length')"
if [[ -z "$segment_id" ]]; then
  echo "[FAIL] no segment_id for run_id=$run_id"
  echo "$list_resp"
  exit 1
fi

detail_resp="$(curl -fsS "$BASE_URL/runs/$run_id/segments/$segment_id")"
full_len="$(echo "$detail_resp" | jq -r '.full_text | length')"
if [[ "$full_len" -le "$preview_len" ]]; then
  echo "[FAIL] expected full_text longer than preview"
  echo "preview_len=$preview_len full_len=$full_len"
  exit 1
fi

echo "[PASS] run inspector smoke run_id=$run_id segment_id=$segment_id full_len=$full_len"
