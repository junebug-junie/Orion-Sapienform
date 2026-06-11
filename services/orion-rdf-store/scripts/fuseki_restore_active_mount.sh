#!/usr/bin/env bash
# Point FUSEKI_DATA_DIR at the dataset tree that still has a full /orion TDB and restart.
# Use when the mounted copy was truncated but the other side of migration still has data.
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

LUKEWARM="${LUKEWARM_FUSEKI_ROOT:-/mnt/storage-lukewarm/rdf-store/fuseki}"
GRAPHDB="${GRAPHDB_FUSEKI_ROOT:-/mnt/graphdb/rdf-store/fuseki}"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
SERVICE="${FUSEKI_SERVICE_NAME:-orion-athena-fuseki}"
MIN_DATASET_BYTES="${MIN_DATASET_BYTES:-10737418240}"
DRY_RUN="${DRY_RUN:-0}"

_dataset_bytes() {
  local dir="$1/databases/orion"
  if [ ! -d "${dir}" ]; then
    echo 0
    return
  fi
  du -sb "${dir}" 2>/dev/null | awk '{print $1}'
}

l_bytes="$(_dataset_bytes "${LUKEWARM}")"
g_bytes="$(_dataset_bytes "${GRAPHDB}")"
echo "lukewarm dataset bytes: ${l_bytes}"
echo "graphdb dataset bytes:  ${g_bytes}"

target=""
if [ "${l_bytes}" -ge "${MIN_DATASET_BYTES}" ] && [ "${l_bytes}" -ge "${g_bytes}" ]; then
  target="${LUKEWARM}"
elif [ "${g_bytes}" -ge "${MIN_DATASET_BYTES}" ]; then
  target="${GRAPHDB}"
else
  echo "restore_mount: no dataset >= ${MIN_DATASET_BYTES} bytes on either path" >&2
  exit 1
fi

echo "target=${target}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1: would set FUSEKI_DATA_DIR=${target} and restart ${SERVICE}"
  exit 0
fi

docker stop "${SERVICE}" 2>/dev/null || true

if grep -q '^FUSEKI_DATA_DIR=' "${ENV_FILE}"; then
  sed -i "s|^FUSEKI_DATA_DIR=.*|FUSEKI_DATA_DIR=${target}|" "${ENV_FILE}"
else
  printf '\nFUSEKI_DATA_DIR=%s\n' "${target}" >> "${ENV_FILE}"
fi

case "${target}" in
  "${GRAPHDB}"*) sed -i 's|^RDF_STORE_DATA_ROOT=.*|RDF_STORE_DATA_ROOT=/mnt/graphdb/rdf-store|' "${ENV_FILE}" ;;
  "${LUKEWARM}"*) sed -i 's|^RDF_STORE_DATA_ROOT=.*|RDF_STORE_DATA_ROOT=/mnt/storage-lukewarm/rdf-store|' "${ENV_FILE}" ;;
esac

docker compose --env-file "${ENV_FILE}" -f docker-compose.yml up -d

for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:3030/\$/ping >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

./scripts/fuseki_storage_status.sh
