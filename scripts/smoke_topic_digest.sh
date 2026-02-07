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
  if ! grep -q '^DIGEST_RUN_ON_START=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "DIGEST_RUN_ON_START=false" >> "${dir}/.env"
  fi
  if ! grep -q '^DIGEST_ENABLED=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "DIGEST_ENABLED=true" >> "${dir}/.env"
  fi
  if ! grep -q '^DIGEST_WINDOW_HOURS=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "DIGEST_WINDOW_HOURS=24" >> "${dir}/.env"
  fi
  if ! grep -q '^DIGEST_TIME_LOCAL=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "DIGEST_TIME_LOCAL=07:30" >> "${dir}/.env"
  fi
  if ! grep -q '^DIGEST_RECIPIENT_GROUP=' "${dir}/.env" && [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    echo "DIGEST_RECIPIENT_GROUP=juniper_primary" >> "${dir}/.env"
  fi
  if ! grep -q '^NOTIFY_API_TOKEN=' "${dir}/.env"; then
    echo "NOTIFY_API_TOKEN=" >> "${dir}/.env"
  fi
  if [[ "${dir}" == "${DIGEST_DIR}" ]]; then
    if ! grep -q '^LANDING_PAD_URL=' "${dir}/.env"; then
      echo "LANDING_PAD_URL=http://localhost:8371" >> "${dir}/.env"
    fi
    if ! grep -q '^TOPICS_MAX_TOPICS=' "${dir}/.env"; then
      echo "TOPICS_MAX_TOPICS=5" >> "${dir}/.env"
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
        if self.path.startswith("/api/topics/summary"):
            payload = {
                "topics": [
                    {"topic": "security", "count": 5},
                    {"topic": "memory", "count": 3},
                ]
            }
        elif self.path.startswith("/api/topics/drift"):
            payload = {
                "topics": [
                    {"topic": "security", "drift_score": 0.7},
                    {"topic": "memory", "drift_score": 0.4},
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

HTTPServer(("0.0.0.0", 8371), Handler).serve_forever()
PY'

docker exec -d "${DIGEST_CONTAINER}" python /tmp/mock_landing_pad.py

NOTIFY_PORT=$(grep -E '^PORT=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2)
NOTIFY_PORT=${NOTIFY_PORT:-7140}
API_TOKEN=$(grep -E '^API_TOKEN=' "${NOTIFY_DIR}/.env" | cut -d '=' -f2-)
AUTH_HEADER=()
if [[ -n "${API_TOKEN}" ]]; then
  AUTH_HEADER=(-H "X-Orion-Notify-Token: ${API_TOKEN}")
fi

seed_payload() {
  local kind=$1
  local severity=$2
  curl -s -X POST "http://localhost:${NOTIFY_PORT}/notify" \
    -H "Content-Type: application/json" \
    "${AUTH_HEADER[@]}" \
    -d "{\"source_service\":\"smoke-test\",\"event_kind\":\"${kind}\",\"severity\":\"${severity}\",\"title\":\"Smoke ${kind}\",\"body_text\":\"Smoke body\",\"recipient_group\":\"juniper_primary\"}"
}

echo "Seeding notifications..."
seed_payload "smoke.topics" "warning" | cat
echo

echo "Running digest on demand..."
docker exec -t "${DIGEST_CONTAINER}" python -m app.run_digest --window-hours 1

DIGEST_BODY=$(curl -s "http://localhost:${NOTIFY_PORT}/notifications?limit=5" "${AUTH_HEADER[@]}" \
  | jq -r '.[] | select(.event_kind=="orion.digest.daily") | .body_text' \
  | head -n 1)

if [[ -z "${DIGEST_BODY}" || "${DIGEST_BODY}" == "null" ]]; then
  echo "FAIL: digest notification not found" >&2
  exit 1
fi

if ! echo "${DIGEST_BODY}" | grep -q "Top Topics"; then
  echo "FAIL: digest missing Top Topics section" >&2
  exit 1
fi

echo "PASS: digest includes Top Topics section"
