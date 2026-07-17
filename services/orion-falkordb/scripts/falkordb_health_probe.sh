#!/usr/bin/env sh
# Probe FalkorDB via Redis PING on the host port or inside the container.
# Exit 0 when healthy; non-zero when unreachable.
set -eu

_env_get() {
  key="$1"
  if [ ! -f .env ]; then
    return 0
  fi
  grep -E "^${key}=" .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'"
}

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-$(_env_get FALKORDB_PORT)}"
PORT="${PORT:-6380}"
NODE_NAME="${NODE_NAME:-$(_env_get NODE_NAME)}"
NODE_NAME="${NODE_NAME:-athena}"
CONTAINER="orion-${NODE_NAME}-falkordb"

_ping() {
  if command -v redis-cli >/dev/null 2>&1; then
    redis-cli -h "${HOST}" -p "${PORT}" PING 2>/dev/null | grep -q PONG
    return $?
  fi
  return 1
}

_docker_ping() {
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi
  docker exec "${CONTAINER}" redis-cli PING 2>/dev/null | grep -q PONG
}

if _ping || _docker_ping; then
  echo "falkordb_health_probe: ok (${HOST}:${PORT}, container=${CONTAINER})"
  exit 0
fi

echo "falkordb_health_probe: FalkorDB not reachable on ${HOST}:${PORT} or via docker exec ${CONTAINER}" >&2
exit 1
