#!/usr/bin/env bash
# Export GraphDB collapse repo to N-Quads via storage-tool (GraphDB must be stopped).
set -euo pipefail

CONTAINER="${GRAPHDB_CONTAINER:-orion-athena-graphdb}"
IMAGE="${GRAPHDB_IMAGE:-orion-athena-graphdb:11.0.0}"
GRAPHDB_HOME="${GRAPHDB_HOME:-/mnt/graphdb/collapse-mirrors}"
STORAGE_PATH="/opt/graphdb/home/data/repositories/collapse/storage"
DEST_IN_CONTAINER="/root/graphdb-import/collapse-export.nq"
HOST_DEST="${HOST_DEST:-${GRAPHDB_HOME}/import/collapse-export.nq}"
SRC_INDEX="${SRC_INDEX:-pos}"

mkdir -p "${GRAPHDB_HOME}/import"

echo "==> Stopping ${CONTAINER} for offline export"
docker stop "${CONTAINER}"

echo "==> Exporting collapse storage -> ${HOST_DEST}"
docker run --rm \
  -v "${GRAPHDB_HOME}:/opt/graphdb/home" \
  -v "${GRAPHDB_HOME}/import:/root/graphdb-import" \
  --entrypoint /opt/graphdb/dist/bin/storage-tool \
  "${IMAGE}" \
  export -s "${STORAGE_PATH}" -r "${SRC_INDEX}" -f "${DEST_IN_CONTAINER}" -i 60

echo "==> Export written to ${HOST_DEST}"
ls -lh "${HOST_DEST}"

echo "==> Restarting ${CONTAINER}"
docker start "${CONTAINER}"
