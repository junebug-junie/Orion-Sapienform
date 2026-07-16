#!/usr/bin/env sh
# Restart Fuseki when the write probe fails (clears TDB lock exhaustion until next buildup).
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SERVICE="${FUSEKI_SERVICE_NAME:-orion-athena-fuseki}"
WAIT_SEC="${FUSEKI_RECOVER_WAIT_SEC:-20}"
COMPACT_LOCK="${FUSEKI_COMPACT_LOCK:-${FUSEKI_DATA_DIR:-/mnt/graphdb/rdf-store/fuseki}/.compact-in-progress}"

# 2026-07-15 fix: acquire the SAME lock fuseki_tdb_compact.sh holds via flock,
# not a plain `[ -f lockfile ]` existence check. The old check-then-act was a
# real TOCTOU race: this cron fires every 20 min, including the exact minute
# compact's own schedule fires on, and compact didn't create its marker file
# until after a du -sb that can take 30s+ on a large dataset -- leaving a
# window where this script could see "no lock yet" and restart Fuseki mid-
# compact. flock -n is a kernel-atomic test: if compact holds the lock, this
# fails immediately and consistently, no gap.
#
# Deliberately held (not released) for the rest of this script, through the
# restart + reprobe below -- an earlier draft released it right after this
# check, which only proved the lock was free at that instant, not for the
# duration of the restart action that follows. A compact starting in that
# gap would race a live `docker compose restart` the same way the original
# incident did, just through a narrower window. Held-until-exit (same as
# fuseki_tdb_compact.sh) closes that too: the kernel releases the lock
# automatically when this script's process exits, on every exit path.
exec 9>"${COMPACT_LOCK}"
if ! flock -n 9; then
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
