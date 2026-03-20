#!/usr/bin/env bash
set -euo pipefail

HUB_BASE_URL=${HUB_BASE_URL:-http://localhost:8080}

html=$(curl -fsS "${HUB_BASE_URL}/")
if ! printf '%s' "$html" | rg -q "topicStudioPanel"; then
  echo "Topic Studio container not found in hub root HTML." >&2
  exit 1
fi

echo "Found topicStudioPanel in hub HTML."

ready_status=$(curl -s -o /tmp/hub_ready.json -w "%{http_code}" "${HUB_BASE_URL}/api/topic-foundry/ready")
cap_status=$(curl -s -o /tmp/hub_cap.json -w "%{http_code}" "${HUB_BASE_URL}/api/topic-foundry/capabilities")

if [[ "$ready_status" == "404" && "$cap_status" == "404" ]]; then
  echo "SKIP: /api/topic-foundry proxy not configured on hub."
  exit 0
fi

if [[ "$ready_status" != 2* ]]; then
  echo "Unexpected status from /ready: ${ready_status}" >&2
  cat /tmp/hub_ready.json >&2 || true
  exit 1
fi

if [[ "$cap_status" != 2* ]]; then
  echo "Unexpected status from /capabilities: ${cap_status}" >&2
  cat /tmp/hub_cap.json >&2 || true
  exit 1
fi

echo "Topic Foundry proxy endpoints responded successfully."
