#!/usr/bin/env sh
# Probe Fuseki ping + authenticated SPARQL query + graph-store write.
# Exit 0 when healthy; non-zero when the store cannot serve writes (e.g. TDB lock exhaustion).
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

HOST="${FUSEKI_HOST:-127.0.0.1}"
PORT="${FUSEKI_PORT:-$(_env_get FUSEKI_PORT)}"
PORT="${PORT:-3030}"
BASE="http://${HOST}:${PORT}"
DATASET="${RDF_STORE_DATASET:-$(_env_get RDF_STORE_DATASET)}"
DATASET="${DATASET:-${FUSEKI_DATASET:-orion}}"
USER="${FUSEKI_USER:-admin}"
PASS="${FUSEKI_ADMIN_PASSWORD:-$(_env_get FUSEKI_ADMIN_PASSWORD)}"
PASS="${PASS:-${FUSEKI_PASS:-orion}}"
CURL="${CURL:-curl}"
PROBE_TURTLE='@prefix ex: <http://example.org/> . ex:probe ex:ok true .'
MAX_ATTEMPTS="${FUSEKI_HEALTH_PROBE_MAX_ATTEMPTS:-1}"
INTERVAL_SEC="${FUSEKI_HEALTH_PROBE_INTERVAL_SEC:-5}"

if ! command -v "${CURL}" >/dev/null 2>&1; then
  echo "fuseki_health_probe: curl not found" >&2
  exit 2
fi

_attempt=1
while [ "${_attempt}" -le "${MAX_ATTEMPTS}" ]; do
  if "${CURL}" -fsS "${BASE}/\$/ping" >/dev/null 2>&1; then
    break
  fi
  if [ "${_attempt}" -ge "${MAX_ATTEMPTS}" ]; then
    echo "fuseki_health_probe: ping failed after ${MAX_ATTEMPTS} attempt(s)" >&2
    exit 1
  fi
  echo "fuseki_health_probe: ping not ready (attempt ${_attempt}/${MAX_ATTEMPTS}); waiting ${INTERVAL_SEC}s" >&2
  sleep "${INTERVAL_SEC}"
  _attempt=$((_attempt + 1))
done

_attempt=1
while [ "${_attempt}" -le "${MAX_ATTEMPTS}" ]; do
  query_code=$("${CURL}" -sS -o /tmp/fuseki_health_probe_query.out -w '%{http_code}' \
    -u "${USER}:${PASS}" \
    -X POST "${BASE}/${DATASET}/query" \
    -H 'Accept: application/sparql-results+json' \
    --data-urlencode 'query=ASK { ?s ?p ?o }' 2>/dev/null || echo "000")
  if [ "${query_code}" = "200" ]; then
    break
  fi
  if [ "${_attempt}" -ge "${MAX_ATTEMPTS}" ]; then
    echo "fuseki_health_probe: query HTTP ${query_code} after ${MAX_ATTEMPTS} attempt(s)" >&2
    cat /tmp/fuseki_health_probe_query.out >&2 || true
    exit 1
  fi
  echo "fuseki_health_probe: query HTTP ${query_code} (attempt ${_attempt}/${MAX_ATTEMPTS}); waiting ${INTERVAL_SEC}s" >&2
  sleep "${INTERVAL_SEC}"
  _attempt=$((_attempt + 1))
done

_attempt=1
while [ "${_attempt}" -le "${MAX_ATTEMPTS}" ]; do
  write_code=$(printf '%s' "${PROBE_TURTLE}" | "${CURL}" -sS -o /tmp/fuseki_health_probe_write.out -w '%{http_code}' \
    -u "${USER}:${PASS}" \
    -X POST "${BASE}/${DATASET}/data" \
    -H 'Content-Type: text/turtle' \
    --data-binary @- 2>/dev/null || echo "000")
  case "${write_code}" in
    200|201|204)
      exit 0
      ;;
  esac
  if [ "${_attempt}" -ge "${MAX_ATTEMPTS}" ]; then
    echo "fuseki_health_probe: graph-store HTTP ${write_code} after ${MAX_ATTEMPTS} attempt(s)" >&2
    cat /tmp/fuseki_health_probe_write.out >&2 || true
    exit 1
  fi
  echo "fuseki_health_probe: graph-store HTTP ${write_code} (attempt ${_attempt}/${MAX_ATTEMPTS}); waiting ${INTERVAL_SEC}s" >&2
  sleep "${INTERVAL_SEC}"
  _attempt=$((_attempt + 1))
done

exit 0
