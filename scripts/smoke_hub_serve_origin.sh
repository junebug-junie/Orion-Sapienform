#!/usr/bin/env bash
set -euo pipefail

HUB_URL=${HUB_URL:-}
if [[ -z "$HUB_URL" ]]; then
  echo "HUB_URL is required (e.g., https://athena.tail... )." >&2
  exit 1
fi

root_status=$(curl -sS -o /dev/null -w "%{http_code}" "${HUB_URL}/")
ready_status=$(curl -sS -o /dev/null -w "%{http_code}" "${HUB_URL}/api/topic-foundry/ready")
cap_status=$(curl -sS -o /dev/null -w "%{http_code}" "${HUB_URL}/api/topic-foundry/capabilities")

echo "Hub root status: ${root_status}"
echo "Topic Foundry /ready status: ${ready_status}"
echo "Topic Foundry /capabilities status: ${cap_status}"

if [[ "$root_status" =~ ^2|^3 ]]; then
  echo "Summary: Hub root reachable."
else
  echo "Summary: Hub root not reachable (status ${root_status})." >&2
fi
