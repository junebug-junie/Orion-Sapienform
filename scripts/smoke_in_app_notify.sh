#!/usr/bin/env bash
set -euo pipefail

BUS_DIR="services/orion-bus"
NOTIFY_DIR="services/orion-notify"
HUB_DIR="services/orion-hub"

for dir in "${NOTIFY_DIR}" "${HUB_DIR}"; do
  if [[ ! -f "${dir}/.env" ]]; then
    echo "Missing ${dir}/.env. Copy ${dir}/.env_example to ${dir}/.env and edit settings." >&2
    exit 1
  fi
  if ! grep -q '^ORION_BUS_URL=' "${dir}/.env"; then
    echo "ORION_BUS_URL=redis://localhost:6379/0" >> "${dir}/.env"
  fi
  if ! grep -q '^ORION_BUS_ENABLED=' "${dir}/.env"; then
    echo "ORION_BUS_ENABLED=true" >> "${dir}/.env"
  fi
  if ! grep -q '^ORION_BUS_ENFORCE_CATALOG=' "${dir}/.env"; then
    echo "ORION_BUS_ENFORCE_CATALOG=true" >> "${dir}/.env"
  fi
  if [[ "${dir}" == "${NOTIFY_DIR}" ]]; then
    if ! grep -q '^NOTIFY_IN_APP_ENABLED=' "${dir}/.env"; then
      echo "NOTIFY_IN_APP_ENABLED=true" >> "${dir}/.env"
    fi
    if ! grep -q '^NOTIFY_IN_APP_CHANNEL=' "${dir}/.env"; then
      echo "NOTIFY_IN_APP_CHANNEL=orion:notify:in_app" >> "${dir}/.env"
    fi
  fi
  if [[ "${dir}" == "${HUB_DIR}" ]]; then
    if ! grep -q '^NOTIFY_IN_APP_ENABLED=' "${dir}/.env"; then
      echo "NOTIFY_IN_APP_ENABLED=true" >> "${dir}/.env"
    fi
    if ! grep -q '^NOTIFY_IN_APP_CHANNEL=' "${dir}/.env"; then
      echo "NOTIFY_IN_APP_CHANNEL=orion:notify:in_app" >> "${dir}/.env"
    fi
  fi
done

PROJECT=${PROJECT:-orion-smoke}
TELEMETRY_ROOT=${TELEMETRY_ROOT:-/tmp/orion-telemetry}
mkdir -p "${TELEMETRY_ROOT}/${PROJECT}/bus/data"

if [[ -z "${SKIP_COMPOSE:-}" ]]; then
  PROJECT="${PROJECT}" TELEMETRY_ROOT="${TELEMETRY_ROOT}" \
    docker compose -f "${BUS_DIR}/docker-compose.yml" up -d

  docker compose -f "${NOTIFY_DIR}/docker-compose.yml" up -d --build

  docker compose --env-file "${HUB_DIR}/.env" -f "${HUB_DIR}/docker-compose.yml" up -d --build

  sleep 3
fi

NOTIFY_PORT=$(grep -E '^PORT=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2)
NOTIFY_PORT=${NOTIFY_PORT:-7140}
API_TOKEN=$(grep -E '^API_TOKEN=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2-)
AUTH_HEADER=()
if [[ -n "${API_TOKEN}" ]]; then
  AUTH_HEADER=(-H "X-Orion-Notify-Token: ${API_TOKEN}")
fi

TITLE="Smoke In-App $(date +%s)"
PAYLOAD=$(cat <<JSON
{"source_service":"smoke-test","event_kind":"notify.in_app.smoke","severity":"error","title":"${TITLE}","body_text":"In-app smoke test","recipient_group":"juniper_primary","channels_requested":["in_app"]}
JSON
)

curl -s -X POST "http://localhost:${NOTIFY_PORT}/notify" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${PAYLOAD}" \
  | cat

echo ""

echo "Polling hub notifications..."
for i in {1..10}; do
  FOUND=$(curl -s "http://localhost:8080/api/notifications?limit=10" \
    | jq -r --arg title "${TITLE}" '.[] | select(.title==$title) | .notification_id' \
    | head -n 1)
  if [[ -n "${FOUND}" ]]; then
    echo "PASS: notification delivered to hub (${FOUND})"
    exit 0
  fi
  sleep 1
done

echo "FAIL: notification not found in hub history"
exit 1
