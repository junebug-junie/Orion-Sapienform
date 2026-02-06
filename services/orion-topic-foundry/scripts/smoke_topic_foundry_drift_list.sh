#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8615}
TOPIC_FOUNDRY_BASE_URL=${TOPIC_FOUNDRY_BASE_URL:-"http://localhost:${PORT}"}
MODEL_NAME=${MODEL_NAME:-"default"}
LIMIT=${LIMIT:-10}

resp=$(curl -fsS "${TOPIC_FOUNDRY_BASE_URL}/drift?model_name=${MODEL_NAME}&limit=${LIMIT}")
echo "$resp" | jq
