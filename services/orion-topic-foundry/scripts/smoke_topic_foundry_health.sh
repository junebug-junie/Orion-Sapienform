#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}

health_json=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/health")
ready_json=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/ready")

echo "Health: ${health_json}"
echo "Ready: ${ready_json}"

echo "$health_json" | jq -e '.ok == true' >/dev/null

echo "$ready_json" | jq -e '.ok == true' >/dev/null
