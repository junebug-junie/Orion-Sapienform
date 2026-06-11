#!/usr/bin/env bash
# Load GraphDB offline export (N-Quads) into Fuseki /orion dataset.
set -euo pipefail

EXPORT_FILE="${EXPORT_FILE:-/mnt/graphdb/collapse-mirrors/import/collapse-export.nq}"
FUSEKI_DATA_URL="${FUSEKI_DATA_URL:-http://localhost:3030/orion/data}"
CHUNK_LINES="${CHUNK_LINES:-500000}"

if [[ ! -f "${EXPORT_FILE}" ]]; then
  echo "missing export file: ${EXPORT_FILE}" >&2
  exit 1
fi

echo "==> Splitting ${EXPORT_FILE} into ${CHUNK_LINES}-line chunks"
WORKDIR="$(mktemp -d)"
split -l "${CHUNK_LINES}" -d -a 4 "${EXPORT_FILE}" "${WORKDIR}/chunk_"

total="$(find "${WORKDIR}" -name 'chunk_*' | wc -l)"
i=0
for chunk in "${WORKDIR}"/chunk_*; do
  i=$((i + 1))
  echo "==> POST chunk ${i}/${total}: $(basename "${chunk}")"
  curl -sf -X POST \
    -H 'Content-Type: application/n-quads' \
    --data-binary @"${chunk}" \
    "${FUSEKI_DATA_URL}"
done

rm -rf "${WORKDIR}"
echo "==> Import complete"
