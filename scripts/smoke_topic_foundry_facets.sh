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
RUN_ID=${RUN_ID:-}

if [[ -z "$RUN_ID" ]]; then
  runs=$(curl -fsS "${BASE_URL}/runs?limit=1")
  RUN_ID=$(jq -r '.runs[0].run_id // empty' <<<"$runs")
fi
if [[ -z "$RUN_ID" ]]; then
  echo "FAIL: No run_id available for facets check" >&2
  exit 1
fi

status=$(curl -s -o /tmp/topic_foundry_facets.json -w "%{http_code}" \
  "${BASE_URL}/segments/facets?run_id=${RUN_ID}")
if [[ "$status" != 2* ]]; then
  echo "FAIL: Facets request failed (status ${status})" >&2
  cat /tmp/topic_foundry_facets.json >&2 || true
  exit 1
fi

echo "Topic Foundry facets smoke checks passed."
