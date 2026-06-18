#!/usr/bin/env sh
# Restart Fuseki when the write probe fails (clears TDB lock exhaustion until next buildup).
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SERVICE="${FUSEKI_SERVICE_NAME:-orion-athena-fuseki}"
WAIT_SEC="${FUSEKI_RECOVER_WAIT_SEC:-20}"
COMPACT_LOCK="${FUSEKI_COMPACT_LOCK:-${FUSEKI_DATA_DIR:-/mnt/graphdb/rdf-store/fuseki}/.compact-in-progress}"

if [ -f "${COMPACT_LOCK}" ]; then
  echo "fuseki_recover: compact in progress (${COMPACT_LOCK}); skipping"
  exit 0
fi

if ./scripts/fuseki_health_probe.sh; then
  echo "fuseki_recover: ${SERVICE} healthy"
  exit 0
fi

echo "fuseki_recover: ${SERVICE} unhealthy — restarting" >&2
docker compose restart "${SERVICE}"
sleep "${WAIT_SEC}"

if ./scripts/fuseki_health_probe.sh; then
  echo "fuseki_recover: ${SERVICE} recovered"
  exit 0
fi

echo "fuseki_recover: ${SERVICE} still unhealthy after restart" >&2
exit 1
