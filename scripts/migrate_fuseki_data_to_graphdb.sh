#!/usr/bin/env bash
# Relocate existing Fuseki TDB + spark/concepts data from lukewarm HDD to /mnt/graphdb (nvme).
# Uses docker when /mnt/graphdb is root-owned (no passwordless sudo required).
set -euo pipefail

SRC_FUSEKI="${SRC_FUSEKI:-/mnt/storage-lukewarm/rdf-store/fuseki}"
DST_ROOT="${DST_ROOT:-/mnt/graphdb}"
DST_FUSEKI="${DST_FUSEKI:-${DST_ROOT}/rdf-store/fuseki}"

_has_content() {
  local dir="$1"
  [[ -d "${dir}" ]] && [[ -n "$(find "${dir}" -mindepth 1 -print -quit 2>/dev/null)" ]]
}

echo "==> Creating destination directories under ${DST_ROOT}"
docker run --rm -v "${DST_ROOT}:/dst" alpine sh -c '
  mkdir -p \
    /dst/rdf-store/fuseki \
    /dst/rdf-store/fuseki-backups \
    /dst/orion/spark \
    /dst/orion/concepts \
    /dst/rdf_logs \
    /dst/sql_logs \
    /dst/vector_logs \
    /dst/logs/biometrics/logs \
    /dst/logs/dream/logs \
    /dst/rag-files/test-txt
'

if _has_content "${SRC_FUSEKI}"; then
  echo "==> Stopping Fuseki before TDB rsync"
  docker stop orion-athena-fuseki 2>/dev/null || true
  echo "==> rsync Fuseki TDB: ${SRC_FUSEKI} -> ${DST_FUSEKI}"
  docker rm -f fuseki-rsync-to-graphdb 2>/dev/null || true
  docker run --name fuseki-rsync-to-graphdb \
    -v "${SRC_FUSEKI}:/src:ro" \
    -v "${DST_FUSEKI}:/dst" \
    alpine sh -c 'apk add --no-cache rsync >/dev/null && rsync -aH --info=progress2 /src/ /dst/'
else
  echo "==> No source Fuseki data at ${SRC_FUSEKI}; skipping TDB rsync"
fi

for pair in \
  "/mnt/storage-lukewarm/orion/spark:${DST_ROOT}/orion/spark" \
  "/mnt/storage-lukewarm/orion/concepts:${DST_ROOT}/orion/concepts" \
  "/mnt/storage-lukewarm/rdf_logs:${DST_ROOT}/rdf_logs" \
  "/mnt/storage-lukewarm/rag-files:${DST_ROOT}/rag-files"; do
  src="${pair%%:*}"
  dst="${pair##*:}"
  if _has_content "${src}"; then
    echo "==> rsync ${src} -> ${dst}"
    docker run --rm -v "${src}:/src:ro" -v "${dst}:/dst" alpine sh -c 'cp -a /src/. /dst/'
  fi
done

echo "==> Done. Restart Fuseki from services/orion-rdf-store with FUSEKI_DATA_DIR=${DST_FUSEKI}"
