#!/bin/sh
set -e

GATEWAY_URL="${FCC_LLAMACPP_BASE_URL:-http://orion-llm-gateway:8210/v1}"
GATEWAY_HEALTH="${GATEWAY_URL%/v1}/health"

if [ -z "${FCC_SKIP_GATEWAY_WAIT:-}" ]; then
  echo "Waiting for LLM gateway at ${GATEWAY_HEALTH} ..."
  i=0
  while [ "$i" -lt 60 ]; do
    if curl -fsS --max-time 2 "$GATEWAY_HEALTH" >/dev/null 2>&1; then
      echo "LLM gateway reachable"
      break
    fi
    sleep 2
    i=$((i + 1))
  done
fi

export FCC_OPEN_BROWSER="${FCC_OPEN_BROWSER:-false}"
export LOG_FILE="${FCC_LOG_FILE:-/tmp/fcc-server.log}"

if [ ! -f /root/.fcc/.env ]; then
  echo "Missing /root/.fcc/.env — mount operator config (see config/fcc.env_example)"
  exit 1
fi

echo "Starting fcc-server on :${PORT:-8082} (secrets from mounted ~/.fcc/.env)"
exec "$@"
