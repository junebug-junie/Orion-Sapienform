#!/usr/bin/env bash
# Orion Signals mesh smoke checks — bus Redis + signal-gateway HTTP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIGNALS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SIGNALS_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${ENV_FILE}"
  set +a
fi

PROJECT="${PROJECT:-orion-athena}"
GATEWAY_PORT="${SIGNAL_GATEWAY_HTTP_PORT:-8879}"
BUS_CONTAINER="orion-${PROJECT}-bus-core"
FAILURES=0

pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }

echo "=== Orion Signals smoke (project=${PROJECT}) ==="

# Bus-core Redis ping (if container is running)
if docker ps --format '{{.Names}}' | grep -qx "${BUS_CONTAINER}"; then
  if docker exec "${BUS_CONTAINER}" redis-cli ping 2>/dev/null | grep -q PONG; then
    pass "bus-core redis ping (${BUS_CONTAINER})"
  else
    fail "bus-core redis ping (${BUS_CONTAINER})"
  fi
else
  echo "SKIP: bus-core container '${BUS_CONTAINER}' not running"
fi

# Signal gateway HTTP
if curl -fsS "http://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; then
  pass "signal-gateway /health (port ${GATEWAY_PORT})"
else
  fail "signal-gateway /health (port ${GATEWAY_PORT})"
fi

if curl -fsS "http://127.0.0.1:${GATEWAY_PORT}/signals/active" >/dev/null 2>&1; then
  pass "signal-gateway /signals/active (port ${GATEWAY_PORT})"
else
  fail "signal-gateway /signals/active (port ${GATEWAY_PORT})"
fi

echo ""
if [[ "${FAILURES}" -gt 0 ]]; then
  echo "Smoke failed (${FAILURES} check(s))." >&2
  exit 1
fi
echo "Smoke passed."
