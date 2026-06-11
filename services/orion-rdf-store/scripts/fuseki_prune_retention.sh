#!/usr/bin/env bash
# SPARQL retention prune against Fuseki. Safe to cron weekly (off-peak).
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ROOT}/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"

_load_env_key() {
  local key="$1"
  local default="${2:-}"
  if [ ! -f "${ENV_FILE}" ]; then
    return 0
  fi
  local val
  val="$(grep -E "^${key}=" "${ENV_FILE}" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'" || true)"
  if [ -n "${val}" ]; then
    export "${key}=${val}"
  elif [ -n "${default}" ]; then
    export "${key}=${default}"
  fi
}

for key in \
  FUSEKI_PORT FUSEKI_ADMIN_PASSWORD RDF_STORE_DATASET RDF_STORE_QUERY_URL RDF_STORE_UPDATE_URL \
  RDF_STORE_USER RDF_STORE_PASS RDF_RETENTION_ENABLED RDF_RETENTION_DRY_RUN \
  RDF_RETENTION_TIMEOUT_SEC RDF_RETENTION_POLICIES; do
  _load_env_key "${key}"
done

PYTHON="${PYTHON:-python3}"
if [ -x "${REPO_ROOT}/orion_dev/bin/python" ]; then
  PYTHON="${REPO_ROOT}/orion_dev/bin/python"
elif [ -x "${REPO_ROOT}/venv/bin/python" ]; then
  PYTHON="${REPO_ROOT}/venv/bin/python"
fi

export RDF_STORE_QUERY_URL="${RDF_STORE_QUERY_URL:-http://127.0.0.1:${FUSEKI_PORT:-3030}/${RDF_STORE_DATASET:-orion}/query}"
export RDF_STORE_UPDATE_URL="${RDF_STORE_UPDATE_URL:-http://127.0.0.1:${FUSEKI_PORT:-3030}/${RDF_STORE_DATASET:-orion}/update}"
export RDF_STORE_USER="${RDF_STORE_USER:-admin}"
export RDF_STORE_PASS="${RDF_STORE_PASS:-${FUSEKI_ADMIN_PASSWORD:-orion}}"

exec env PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${REPO_ROOT}/scripts/run_fuseki_retention.py" "$@"
