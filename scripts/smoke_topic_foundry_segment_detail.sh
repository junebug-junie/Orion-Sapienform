#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-${TOPIC_FOUNDRY_BASE_URL:-http://127.0.0.1:8080/api/topic-foundry}}"
BASE_URL="${BASE_URL%/}"

runs=$(curl -fsS "${BASE_URL}/runs?limit=1")
run_id=$(jq -r '.runs[0].run_id // empty' <<<"$runs")
if [[ -z "$run_id" ]]; then
  echo "FAIL: No runs available for segment detail smoke" >&2
  exit 1
fi

segments=$(curl -fsS "${BASE_URL}/segments?run_id=${run_id}&include_snippet=true&limit=1")
segment_id=$(jq -r '.segments[0].segment_id // empty' <<<"$segments")
snippet=$(jq -r '.segments[0].snippet // empty' <<<"$segments")
if [[ -z "$segment_id" ]]; then
  echo "FAIL: No segment_id returned for run ${run_id}" >&2
  exit 1
fi

detail=$(curl -fsS "${BASE_URL}/segments/${segment_id}?include_full_text=true")
full_text=$(jq -r '.full_text // empty' <<<"$detail")
if [[ -z "$full_text" ]]; then
  echo "FAIL: full_text missing for segment ${segment_id}" >&2
  exit 1
fi

if [[ -n "$snippet" && ${#full_text} -le ${#snippet} ]]; then
  echo "FAIL: full_text length (${#full_text}) not greater than snippet length (${#snippet})" >&2
  exit 1
fi

echo "Segment detail smoke checks passed."
