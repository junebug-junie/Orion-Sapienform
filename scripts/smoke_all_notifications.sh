#!/usr/bin/env bash
set -euo pipefail

BUS_DIR="services/orion-bus"
NOTIFY_DIR="services/orion-notify"
HUB_DIR="services/orion-hub"
DIGEST_DIR="services/orion-notify-digest"

TEST_TIMEOUT_SECONDS=${TEST_TIMEOUT_SECONDS:-120}
KEEP_UP=0
ONLY_REGEX=""
LIST_ONLY=0

usage() {
  cat <<USAGE
Usage: $0 [--keep-up|--no-down] [--only <regex>] [--list]

Options:
  --keep-up, --no-down   Leave containers running after tests
  --only <regex>         Run only tests whose name matches regex
  --list                 List discovered smoke tests
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-up|--no-down)
      KEEP_UP=1
      shift
      ;;
    --only)
      ONLY_REGEX=${2:-}
      shift 2
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PROJECT=${PROJECT:-orion-smoke}
TELEMETRY_ROOT=${TELEMETRY_ROOT:-/tmp/orion-telemetry}
LOG_DIR="logs/smoke"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${LOG_DIR}"
mkdir -p "${TELEMETRY_ROOT}/${PROJECT}/bus/data"

TESTS=()
add_test() {
  local name=$1
  local path=$2
  if [[ -f "${path}" ]]; then
    TESTS+=("${name}:${path}")
  fi
}

add_test "notify" "scripts/smoke_notify.sh"
add_test "in_app" "scripts/smoke_in_app_notify.sh"
add_test "attention" "scripts/smoke_attention.sh"
add_test "chat_message" "scripts/smoke_chat_message.sh"
add_test "notify_prefs" "scripts/smoke_notify_prefs.sh"
add_test "digest" "scripts/smoke_digest.sh"
add_test "topic_digest" "scripts/smoke_topic_digest.sh"
add_test "topic_drift_alert" "scripts/smoke_topic_drift_alert.sh"

if [[ ${LIST_ONLY} -eq 1 ]]; then
  for entry in "${TESTS[@]}"; do
    echo "${entry%%:*}"
  done
  exit 0
fi

wait_for_health() {
  local url=$1
  local name=$2
  local retries=20
  local delay=1
  for _ in $(seq 1 ${retries}); do
    if curl -fsS "${url}" >/dev/null; then
      echo "PASS: ${name} healthy"
      return 0
    fi
    sleep ${delay}
    delay=$((delay + 1))
  done
  echo "FAIL: ${name} health check failed (${url})" >&2
  return 1
}

start_stack() {
  PROJECT="${PROJECT}" TELEMETRY_ROOT="${TELEMETRY_ROOT}" \
    docker compose -f "${BUS_DIR}/docker-compose.yml" up -d

  docker compose -f "${NOTIFY_DIR}/docker-compose.yml" up -d --build

  docker compose --env-file "${HUB_DIR}/.env" -f "${HUB_DIR}/docker-compose.yml" up -d --build

  docker compose -f "${DIGEST_DIR}/docker-compose.yml" up -d --build

  wait_for_health "http://localhost:7140/health" "notify"
  wait_for_health "http://localhost:8080/health" "hub"
  wait_for_health "http://localhost:7150/health" "digest"
}

stop_stack() {
  docker compose -f "${DIGEST_DIR}/docker-compose.yml" down || true
  docker compose --env-file "${HUB_DIR}/.env" -f "${HUB_DIR}/docker-compose.yml" down || true
  docker compose -f "${NOTIFY_DIR}/docker-compose.yml" down || true
  PROJECT="${PROJECT}" TELEMETRY_ROOT="${TELEMETRY_ROOT}" \
    docker compose -f "${BUS_DIR}/docker-compose.yml" down || true
}

if [[ ${KEEP_UP} -eq 0 ]]; then
  trap stop_stack EXIT
fi

start_stack

export SKIP_COMPOSE=1

for entry in "${TESTS[@]}"; do
  name=${entry%%:*}
  path=${entry#*:}
  if [[ -n "${ONLY_REGEX}" && ! "${name}" =~ ${ONLY_REGEX} ]]; then
    continue
  fi
  echo "========================================"
  echo "Running smoke test: ${name}"
  echo "----------------------------------------"
  log_file="${LOG_DIR}/${TIMESTAMP}_${name}.log"

  set +e
  timeout "${TEST_TIMEOUT_SECONDS}s" bash "${path}" >"${log_file}" 2>&1
  status=$?
  set -e

  if [[ ${status} -ne 0 ]]; then
    echo "FAIL: ${name} (exit ${status})" >&2
    echo "--- tail ${log_file} ---" >&2
    tail -n 80 "${log_file}" >&2 || true
    exit ${status}
  fi

  echo "PASS: ${name}"
 done

echo "All notification smoke tests completed."

if [[ ${KEEP_UP} -eq 1 ]]; then
  echo "Containers left running (--keep-up)."
fi
