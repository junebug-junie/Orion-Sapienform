#!/usr/bin/env bash
set -euo pipefail

NOTIFY_DIR="services/orion-notify"
DIGEST_DIR="services/orion-notify-digest"

for dir in "${NOTIFY_DIR}" "${DIGEST_DIR}"; do
  if [[ ! -f "${dir}/.env" ]]; then
    echo "Missing ${dir}/.env. Copy ${dir}/.env_example to ${dir}/.env and edit settings." >&2
    exit 1
  fi
  if ! grep -q '^POSTGRES_URI=' "${dir}/.env"; then
    echo "POSTGRES_URI=sqlite:////data/notify.db" >> "${dir}/.env"
  fi
  if ! grep -q '^NOTIFY_SERVICE_URL=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "NOTIFY_SERVICE_URL=http://orion-notify:7140" >> "${dir}/.env"
  fi
  if ! grep -q '^NOTIFY_API_TOKEN=' "${dir}/.env"; then
    echo "NOTIFY_API_TOKEN=" >> "${dir}/.env"
  fi
  if [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    if ! grep -q '^LANDING_PAD_URL=' "${dir}/.env"; then
      echo "LANDING_PAD_URL=http://localhost:8372" >> "${dir}/.env"
    fi
    if ! grep -q '^DRIFT_ALERTS_ENABLED=' "${dir}/.env"; then
      echo "DRIFT_ALERTS_ENABLED=true" >> "${dir}/.env"
    fi
    if ! grep -q '^DRIFT_CHECK_INTERVAL_SECONDS=' "${dir}/.env"; then
      echo "DRIFT_CHECK_INTERVAL_SECONDS=5" >> "${dir}/.env"
    fi
    if ! grep -q '^DRIFT_ALERT_THRESHOLD=' "${dir}/.env"; then
      echo "DRIFT_ALERT_THRESHOLD=0.5" >> "${dir}/.env"
    fi
    if ! grep -q '^DRIFT_ALERT_MAX_ITEMS=' "${dir}/.env"; then
      echo "DRIFT_ALERT_MAX_ITEMS=3" >> "${dir}/.env"
    fi
    if ! grep -q '^DRIFT_ALERT_SEVERITY=' "${dir}/.env"; then
      echo "DRIFT_ALERT_SEVERITY=warning" >> "${dir}/.env"
    fi
    if ! grep -q '^DRIFT_ALERT_EVENT_KIND=' "${dir}/.env"; then
      echo "DRIFT_ALERT_EVENT_KIND=orion.topics.drift" >> "${dir}/.env"
    fi
    if ! grep -q '^DRIFT_ALERT_DEDUPE_WINDOW_SECONDS=' "${dir}/.env"; then
      echo "DRIFT_ALERT_DEDUPE_WINDOW_SECONDS=3600" >> "${dir}/.env"
    fi
    if ! grep -q '^TOPICS_DRIFT_MIN_TURNS=' "${dir}/.env"; then
      echo "TOPICS_DRIFT_MIN_TURNS=1" >> "${dir}/.env"
    fi
    if ! grep -q '^TOPICS_DRIFT_MAX_SESSIONS=' "${dir}/.env"; then
      echo "TOPICS_DRIFT_MAX_SESSIONS=10" >> "${dir}/.env"
    fi
  fi
done

if [[ -z "${SKIP_COMPOSE:-}" ]]; then
  docker compose -f "${NOTIFY_DIR}/docker-compose.yml" up -d --build
  docker compose -f "${DIGEST_DIR}/docker-compose.yml" up -d --build
  sleep 2
fi

DIGEST_CONTAINER=$(docker compose -f "${DIGEST_DIR}/docker-compose.yml" ps -q notify-digest)
if [[ -z "${DIGEST_CONTAINER}" ]]; then
  echo "Digest container not running" >&2
  exit 1
fi

docker exec "${DIGEST_CONTAINER}" bash -c 'cat > /tmp/mock_landing_pad.py <<"PY"
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/topics/drift"):
            payload = {
                "topics": [
                    {"topic": "security", "drift_score": 0.9},
                    {"topic": "memory", "drift_score": 0.6},
                ]
            }
        elif self.path.startswith("/api/topics/summary"):
            payload = {
                "topics": [
                    {"topic": "security", "count": 5},
                    {"topic": "memory", "count": 3},
                ]
            }
        else:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

HTTPServer(("0.0.0.0", 8372), Handler).serve_forever()
PY'

docker exec -d "${DIGEST_CONTAINER}" python /tmp/mock_landing_pad.py

NOTIFY_PORT=$(grep -E '^PORT=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2)
NOTIFY_PORT=${NOTIFY_PORT:-7140}
API_TOKEN=$(grep -E '^API_TOKEN=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2-)
AUTH_HEADER=()
if [[ -n "${API_TOKEN}" ]]; then
  AUTH_HEADER=(-H "X-Orion-Notify-Token: ${API_TOKEN}")
fi

echo "Waiting for drift alert..."
FOUND=""
for _ in {1..10}; do
  FOUND=$(curl -s "http://localhost:${NOTIFY_PORT}/notifications?limit=10&event_kind=orion.topics.drift" "${AUTH_HEADER[@]}" \
    | jq -r '.[] | select(.event_kind=="orion.topics.drift") | .notification_id' \
    | head -n 1)
  if [[ -n "${FOUND}" && "${FOUND}" != "null" ]]; then
    echo "PASS: drift alert notification found (${FOUND})"
    exit 0
  fi
  sleep 2
done

echo "FAIL: drift alert notification not found" >&2
exit 1
