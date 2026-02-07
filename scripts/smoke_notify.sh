#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="services/orion-notify"
ENV_FILE="${SERVICE_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy ${SERVICE_DIR}/.env_example to ${ENV_FILE} and edit SMTP settings." >&2
  exit 1
fi

if [[ -z "${SKIP_COMPOSE:-}" ]]; then
  docker compose -f "${SERVICE_DIR}/docker-compose.yml" up -d --build
  sleep 2
fi

PORT=$(grep -E '^PORT=' "${ENV_FILE}" | cut -d '=' -f2)
PORT=${PORT:-7140}

API_TOKEN=$(grep -E '^API_TOKEN=' "${ENV_FILE}" | cut -d '=' -f2-)
AUTH_HEADER=()
if [[ -n "${API_TOKEN}" ]]; then
  AUTH_HEADER=(-H "X-Orion-Notify-Token: ${API_TOKEN}")
fi

PAYLOAD='{"source_service":"smoke-test","event_kind":"notify.smoke","severity":"error","title":"[Orion] Notify smoke test","body_text":"Notify service smoke test","recipient_group":"juniper_primary","dedupe_key":"smoke-notify-1","dedupe_window_seconds":60}'

echo "Posting first notification..."
FIRST=$(curl -s -X POST "http://localhost:${PORT}/notify" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${PAYLOAD}")

echo "${FIRST}"

echo "Posting duplicate notification..."
SECOND=$(curl -s -X POST "http://localhost:${PORT}/notify" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${PAYLOAD}")

echo "${SECOND}"

echo "Latest notifications:"
curl -s "http://localhost:${PORT}/notifications?limit=2" \
  "${AUTH_HEADER[@]}" \
  | cat

echo ""

export FIRST SECOND
python - <<'PY'
import json
import os

first = json.loads(os.environ["FIRST"]) if os.environ.get("FIRST") else {}
second = json.loads(os.environ["SECOND"]) if os.environ.get("SECOND") else {}
print("Expected first status != 'deduped'; got:", first.get("status"))
print("Expected second status == 'deduped'; got:", second.get("status"))
PY
