#!/usr/bin/env bash
set -euo pipefail

HUB_BASE_URL=${HUB_BASE_URL:-http://localhost:8080}

html=$(curl -fsS "${HUB_BASE_URL}/")
if ! printf '%s' "$html" | rg -q "topicStudioPanel"; then
  echo "Topic Studio container not found in hub root HTML." >&2
  exit 1
fi

app_js=$(curl -fsS "${HUB_BASE_URL}/static/js/app.js")
if ! printf '%s' "$app_js" | rg -q "json\\.datasets"; then
  echo "app.js does not reference json.datasets parsing." >&2
  exit 1
fi
if ! printf '%s' "$app_js" | rg -q "datasetsCount="; then
  echo "app.js does not include datasetsCount debug line." >&2
  exit 1
fi

dataset_status=$(curl -s -o /tmp/hub_datasets.json -w "%{http_code}" "${HUB_BASE_URL}/api/topic-foundry/datasets")
if [[ "$dataset_status" == "404" ]]; then
  echo "SKIP: /api/topic-foundry proxy not configured on hub."
  exit 0
fi
if [[ "$dataset_status" != 2* ]]; then
  echo "Unexpected status from /datasets: ${dataset_status}" >&2
  cat /tmp/hub_datasets.json >&2 || true
  exit 1
fi

jq -e '.datasets | length > 0' /tmp/hub_datasets.json >/dev/null
echo "Topic Studio datasets and preview options appear wired."
