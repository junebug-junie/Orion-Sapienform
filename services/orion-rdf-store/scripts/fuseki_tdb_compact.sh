#!/usr/bin/env bash
# Offline TDB2 compact: read SOURCE dataset dir, write compacted tree to DEST, swap into place.
# Requires free space on DEST filesystem >= source size (check first).
#
# Example:
#   SOURCE=/mnt/storage-lukewarm/rdf-store/fuseki/databases/orion \
#   DEST=/mnt/graphdb/rdf-store/fuseki-compact/databases/orion \
#   ./scripts/fuseki_tdb_compact.sh
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SOURCE="${SOURCE:-/mnt/graphdb/rdf-store/fuseki/databases/orion}"
DEST="${DEST:-/mnt/graphdb/rdf-store/fuseki-compact/databases/orion}"
JENA_VERSION="${JENA_VERSION:-5.1.0}"
JENA_CACHE="${JENA_CACHE:-/tmp/apache-jena-${JENA_VERSION}}"
SERVICE="${FUSEKI_SERVICE_NAME:-orion-athena-fuseki}"
FUSEKI_ROOT="$(dirname "$(dirname "${SOURCE}")")"
DRY_RUN="${DRY_RUN:-0}"

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
  mkdir -p "${JENA_CACHE}"
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

if [ ! -d "${SOURCE}/Data-0001" ] && [ ! -f "${SOURCE}/tdb.lock" ]; then
  echo "compact: source does not look like a TDB2 dataset: ${SOURCE}" >&2
  exit 1
fi

src_bytes="$(_bytes "${SOURCE}")"
dest_parent="$(dirname "${DEST}")"
dest_free="$(_free_bytes "${dest_parent}")"
echo "==> Source ${SOURCE}: ${src_bytes} bytes"
echo "==> Dest parent ${dest_parent}: ${dest_free} bytes free"

if [ "${dest_free}" -lt "${src_bytes}" ]; then
  echo "compact: need >= source bytes free on dest filesystem (have ${dest_free}, need ${src_bytes})" >&2
  exit 1
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1: would stop ${SERVICE}, compact ${SOURCE} -> ${DEST}, swap into ${FUSEKI_ROOT}"
  exit 0
fi

echo "==> Stopping ${SERVICE}"
docker stop "${SERVICE}" 2>/dev/null || true

_ensure_jena
rm -rf "${DEST}"
mkdir -p "$(dirname "${DEST}")"

echo "==> Compacting (this can take a long time)"
"${JENA_CACHE}/bin/tdb2.tdbcompact" --loc="${SOURCE}" --loc2="${DEST}"

dest_bytes="$(_bytes "${DEST}")"
echo "==> Compacted size: ${dest_bytes} bytes (was ${src_bytes})"

backup="${SOURCE}.pre-compact-$(date +%Y%m%d%H%M%S)"
echo "==> Swapping ${SOURCE} -> ${backup}"
docker stop "${SERVICE}" 2>/dev/null || true
docker run --rm \
  -v "$(dirname "${SOURCE}"):/parent" \
  alpine sh -c "mv /parent/$(basename "${SOURCE}") /parent/$(basename "${backup}")"
docker run --rm \
  -v "$(dirname "${DEST}"):/parent" \
  -v "$(dirname "${SOURCE}"):/destparent" \
  alpine sh -c "mv /parent/$(basename "${DEST}") /destparent/$(basename "${SOURCE}")"

echo "==> Restarting Fuseki"
docker compose -f docker-compose.yml up -d

echo "==> After verifying health, remove backup: rm -rf ${backup}"
echo "==> Compact complete"
