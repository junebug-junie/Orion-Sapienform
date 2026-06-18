#!/usr/bin/env bash
# Offline TDB2 compact (Jena 5): in-place rebuild via tdb2.tdbcompact --loc --deleteOld.
# Stops Fuseki, compacts the dataset directory, restarts Fuseki.
#
# Requires free space on the dataset filesystem >= source size (peak during compact).
#
# Example:
#   SOURCE=/mnt/graphdb/rdf-store/fuseki/databases/orion \
#   ./scripts/fuseki_tdb_compact.sh
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SOURCE="${SOURCE:-/mnt/graphdb/rdf-store/fuseki/databases/orion}"
JENA_VERSION="${JENA_VERSION:-5.1.0}"
JENA_CACHE="${JENA_CACHE:-/tmp/apache-jena-${JENA_VERSION}}"
JENA_IMAGE="${JENA_IMAGE:-eclipse-temurin:21-jre-jammy}"
SERVICE="${FUSEKI_SERVICE_NAME:-orion-athena-fuseki}"
WRITER_SERVICE="${RDF_WRITER_SERVICE_NAME:-orion-athena-rdf-writer}"
DRY_RUN="${DRY_RUN:-0}"
DELETE_OLD="${DELETE_OLD:-1}"
COMPACT_LOCK="${FUSEKI_COMPACT_LOCK:-$(dirname "$(dirname "${SOURCE}")")/.compact-in-progress}"

_bytes() {
  local dir="$1"
  du -sb "${dir}" 2>/dev/null | awk '{print $1}'
}

_free_bytes() {
  local dir="$1"
  df -Pk "${dir}" | awk 'NR==2 {print $4 * 1024}'
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

if ! compgen -G "${SOURCE}/Data-*" >/dev/null && [ ! -f "${SOURCE}/tdb.lock" ]; then
  echo "compact: source does not look like a TDB2 dataset: ${SOURCE}" >&2
  exit 1
fi

src_bytes="$(_bytes "${SOURCE}")"
fs_free="$(_free_bytes "${SOURCE}")"
echo "==> Source ${SOURCE}: ${src_bytes} bytes"
echo "==> Filesystem free: ${fs_free} bytes"

if [ "${fs_free}" -lt "${src_bytes}" ]; then
  echo "compact: need >= source bytes free on dataset filesystem (have ${fs_free}, need ${src_bytes})" >&2
  exit 1
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1: would stop ${SERVICE} (+ ${WRITER_SERVICE}), compact in-place ${SOURCE}, restart ${SERVICE}"
  exit 0
fi

_cleanup() {
  rm -f "${COMPACT_LOCK}"
}
trap _cleanup EXIT INT TERM

touch "${COMPACT_LOCK}"

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
docker compose -f docker-compose.yml restart "${SERVICE}"
docker start "${WRITER_SERVICE}" 2>/dev/null || true

echo "==> Verifying health"
if ! make health-probe; then
  echo "compact: health-probe failed after restart" >&2
  exit 1
fi

echo "==> Compact complete"
