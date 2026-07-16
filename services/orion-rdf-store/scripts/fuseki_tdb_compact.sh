#!/usr/bin/env bash
# Offline TDB2 compact (Jena 5): in-place rebuild via tdb2.tdbcompact --loc --deleteOld.
# Stops Fuseki, compacts the dataset directory, restarts Fuseki.
#
# Requires free space on the dataset filesystem >= source size (peak during compact).
#
# Example:
#   SOURCE=/mnt/graphdb/rdf-store/fuseki/databases/orion \
#   ./scripts/fuseki_tdb_compact.sh
# -E (errtrace) is load-bearing, not decorative: without it, the ERR trap
# below does NOT fire when a function's failure comes from its own last
# command failing naturally (no explicit `return N`) -- which is exactly the
# shape of _run_tdbcompact and _ensure_jena below. Verified empirically
# (2026-07-15 review): identical function structure without -E silently never
# triggers the trap for the tdb2.tdbcompact failure this whole notify
# mechanism exists to catch -- the literal failure mode from the incident.
set -Eeuo pipefail

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${ROOT}/../.." && pwd)"
cd "${ROOT}"

SOURCE="${SOURCE:-/mnt/graphdb/rdf-store/fuseki/databases/orion}"
JENA_VERSION="${JENA_VERSION:-5.1.0}"
JENA_CACHE="${JENA_CACHE:-/tmp/apache-jena-${JENA_VERSION}}"
JENA_IMAGE="${JENA_IMAGE:-eclipse-temurin:21-jre-jammy}"
SERVICE="${FUSEKI_SERVICE_NAME:-orion-athena-fuseki}"
WRITER_SERVICE="${RDF_WRITER_SERVICE_NAME:-orion-athena-rdf-writer}"
DRY_RUN="${DRY_RUN:-0}"
DELETE_OLD="${DELETE_OLD:-1}"
# Same derivation as fuseki_recover.sh (FUSEKI_DATA_DIR-based, not
# SOURCE-derived) -- 2026-07-15 review finding: the two scripts previously
# computed this path via different formulas that only happened to agree
# under the current default layout. A deployment that overrides SOURCE to a
# sibling dataset without also setting FUSEKI_DATA_DIR to match could
# silently defeat the flock coordination (each script locking a different,
# uncontended file). Single formula now, both scripts, always identical.
COMPACT_LOCK="${FUSEKI_COMPACT_LOCK:-${FUSEKI_DATA_DIR:-/mnt/graphdb/rdf-store/fuseki}/.compact-in-progress}"
NOTIFY_BASE_URL="${NOTIFY_BASE_URL:-http://localhost:7140}"
NOTIFY_API_TOKEN="${NOTIFY_API_TOKEN:-}"

# Same interpreter-resolution fallback as fuseki_prune_retention.sh in this
# same directory (repo venv over bare python3, since the athena cron host's
# bare python3 isn't guaranteed to have a usable stdlib/PATH entry).
PYTHON="${PYTHON:-python3}"
if [ -x "${REPO_ROOT}/orion_dev/bin/python" ]; then
  PYTHON="${REPO_ROOT}/orion_dev/bin/python"
elif [ -x "${REPO_ROOT}/venv/bin/python" ]; then
  PYTHON="${REPO_ROOT}/venv/bin/python"
fi

_bytes() {
  local dir="$1"
  du -sb "${dir}" 2>/dev/null | awk '{print $1}'
}

_free_bytes() {
  local dir="$1"
  df -Pk "${dir}" | awk 'NR==2 {print $4 * 1024}'
}

# 2026-07-15 fix: on any failure past this point, post an operator attention
# request to orion-notify so it surfaces in Hub's Pending Attention panel
# (same pattern orion-mesh-guardian uses -- see
# services/orion-mesh-guardian/app/attention.py -- NOT the cognitive-loop-only
# PendingAttentionCardV1 schema, which is unrelated and schema-locked against
# non-cognitive sources). Best-effort: curl/python failures here must never
# mask the real compact failure, so errors are swallowed (`|| return 0` /
# trailing `|| true`).
_notify_failure() {
  local exit_code="$1"
  local failed_command="$2"
  local line_no="$3"
  local message="Fuseki TDB compact failed (exit ${exit_code}) at line ${line_no}: ${failed_command}. See /mnt/graphdb/rdf_logs/fuseki-compact-run.log on the host for full output."
  local payload
  payload=$("${PYTHON}" -c '
import json, sys
print(json.dumps({
    "source_service": "orion-rdf-store-compact",
    "reason": "fuseki_tdb_compact_failed",
    "severity": "error",
    "message": sys.argv[1],
    "require_ack": True,
    "context": {
        "source_service": "orion-rdf-store-compact",
        "event_kind": "orion.rdf_store.compact.failed.v1",
        "reason": "fuseki_tdb_compact_failed",
        "exit_code": sys.argv[2],
        "failed_command": sys.argv[3],
        "line_no": sys.argv[4],
        "source_dataset": sys.argv[5],
    },
}))
' "${message}" "${exit_code}" "${failed_command}" "${line_no}" "${SOURCE}" 2>/dev/null) || {
    echo "compact: WARNING -- could not build notify payload (is ${PYTHON} usable?); Pending Attention will NOT see this failure" >&2
    return 0
  }
  if ! curl -fsS -X POST "${NOTIFY_BASE_URL%/}/attention/request" \
    -H "Content-Type: application/json" \
    ${NOTIFY_API_TOKEN:+-H "X-Orion-Notify-Token: ${NOTIFY_API_TOKEN}"} \
    -d "${payload}" \
    --max-time 10 >/dev/null 2>&1; then
    echo "compact: WARNING -- notify POST to ${NOTIFY_BASE_URL} failed; Pending Attention will NOT see this failure (this message is the only record)" >&2
  fi
}
# Catches any UNANTICIPATED command failure under set -e (docker/curl/tar/mv
# calls in _ensure_jena, the tdb2.tdbcompact call itself -- the actual failure
# mode from the incident this fix addresses). Does NOT fire for explicit
# `exit N` statements -- bash's ERR trap is documented to skip those (verified
# empirically: `exit 1` inside an `if ! cmd; then exit 1; fi` block never
# triggers a registered ERR trap, only real non-zero command exits do). The
# three explicit-exit paths below (bad dataset shape, insufficient free space,
# health-probe failure) each call _fail(), which notifies explicitly for
# exactly this reason -- 2026-07-15 review finding, confirmed live before
# fixing.
trap '_notify_failure "$?" "${BASH_COMMAND}" "${LINENO}"' ERR

_fail() {
  local msg="$1"
  echo "compact: ${msg}" >&2
  _notify_failure "1" "${msg}" "${LINENO}"
  exit 1
}

_ensure_jena() {
  if [ -x "${JENA_CACHE}/bin/tdb2.tdbcompact" ]; then
    return 0
  fi
  mkdir -p "$(dirname "${JENA_CACHE}")"
  tarball="/tmp/apache-jena-${JENA_VERSION}.tar.gz"
  if [ ! -f "${tarball}" ]; then
    echo "==> Downloading Apache Jena ${JENA_VERSION}"
    curl -fsSL -o "${tarball}" \
      "https://archive.apache.org/dist/jena/binaries/apache-jena-${JENA_VERSION}.tar.gz"
  fi
  rm -rf "${JENA_CACHE}"
  tar -xzf "${tarball}" -C /tmp
  mv "/tmp/apache-jena-${JENA_VERSION}" "${JENA_CACHE}"
}

_run_tdbcompact() {
  local loc="$1"
  local delete_old_flag=()
  if [ "${DELETE_OLD}" = "1" ]; then
    delete_old_flag=(--deleteOld)
  fi

  if command -v java >/dev/null 2>&1; then
    "${JENA_CACHE}/bin/tdb2.tdbcompact" --loc="${loc}" "${delete_old_flag[@]}"
    return 0
  fi

  echo "==> No host java; running tdb2.tdbcompact in ${JENA_IMAGE}"
  docker run --rm \
    -v "${JENA_CACHE}:/jena:ro" \
    -v "${loc}:/db:rw" \
    "${JENA_IMAGE}" \
    /jena/bin/tdb2.tdbcompact --loc=/db "${delete_old_flag[@]}"
}

# 2026-07-15 fix: acquire the compact lock FIRST, atomically, before any other
# work -- including before the du -sb / free-space checks below. The previous
# version touched a plain marker file only after those checks (du -sb alone
# can take 30s+ on a 300GB+ dataset) and fuseki_recover.sh's cron (every 20
# min, including the exact minute this schedule fires on) only did a plain
# `[ -f lockfile ]` existence check -- neither half was atomic, so recover
# could restart Fuseki mid-compact in the window before the marker existed,
# leaving compact fighting a live tdb.lock it didn't expect ("held by process
# N"). flock on the same lock file, tested via a non-blocking acquire, closes
# that race for both scripts: whichever side's flock() syscall wins the
# exclusive lock first is guaranteed consistent, kernel-atomic mutual
# exclusion -- no check-then-act gap. This exact race caused 4 consecutive
# weekly compact failures (Jun 21/28, Jul 5/12 2026) after the one successful
# run on Jun 17-18.
#
# DRY_RUN still runs the dataset-shape and free-space checks below (matching
# pre-2026-07-15 behavior, and this file's own "Dry-run (checks disk space,
# prints plan)" description in the README) -- it just does so under the lock,
# which is harmless since DRY_RUN never stops Fuseki or writes anything; the
# lock releases automatically when the script exits either way.
exec 200>"${COMPACT_LOCK}"
if ! flock -n 200; then
  echo "compact: another compact (or a probing recover) holds the lock; exiting" >&2
  exit 0
fi

if ! compgen -G "${SOURCE}/Data-*" >/dev/null && [ ! -f "${SOURCE}/tdb.lock" ]; then
  _fail "source does not look like a TDB2 dataset: ${SOURCE}"
fi

src_bytes="$(_bytes "${SOURCE}")"
fs_free="$(_free_bytes "${SOURCE}")"
echo "==> Source ${SOURCE}: ${src_bytes} bytes"
echo "==> Filesystem free: ${fs_free} bytes"

if [ "${fs_free}" -lt "${src_bytes}" ]; then
  _fail "need >= source bytes free on dataset filesystem (have ${fs_free}, need ${src_bytes})"
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1: would stop ${SERVICE} (+ ${WRITER_SERVICE}), compact in-place ${SOURCE}, restart ${SERVICE}"
  exit 0
fi

echo "==> Stopping ${SERVICE} and ${WRITER_SERVICE}"
docker stop "${SERVICE}" "${WRITER_SERVICE}" 2>/dev/null || true
rm -f "${SOURCE}/tdb.lock" 2>/dev/null || true

_ensure_jena

echo "==> Compacting in-place (this can take a long time)"
start_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
_run_tdbcompact "${SOURCE}"
end_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

dest_bytes="$(_bytes "${SOURCE}")"
echo "==> Compacted size: ${dest_bytes} bytes (was ${src_bytes})"
echo "==> compact_started=${start_ts} compact_finished=${end_ts}"

echo "==> Fixing dataset ownership for Fuseki (uid 100:101)"
docker run --rm \
  -v "${SOURCE}:/db:rw" \
  alpine sh -c 'chown -R 100:101 /db && chmod -R u+rwX,g+rwX /db'

echo "==> Restarting ${SERVICE}"
docker compose -f docker-compose.yml up -d "${SERVICE}"
docker start "${WRITER_SERVICE}" 2>/dev/null || true

echo "==> Verifying health (waiting for Fuseki boot; start_period=90s)"
PROBE_ATTEMPTS="${FUSEKI_COMPACT_HEALTH_PROBE_MAX_ATTEMPTS:-24}"
PROBE_INTERVAL="${FUSEKI_COMPACT_HEALTH_PROBE_INTERVAL_SEC:-5}"
if ! FUSEKI_HEALTH_PROBE_MAX_ATTEMPTS="${PROBE_ATTEMPTS}" \
     FUSEKI_HEALTH_PROBE_INTERVAL_SEC="${PROBE_INTERVAL}" \
     make health-probe; then
  _fail "health-probe failed after restart (${PROBE_ATTEMPTS} attempts x ${PROBE_INTERVAL}s)"
fi

echo "==> Compact complete"
