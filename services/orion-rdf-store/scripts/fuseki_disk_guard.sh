#!/usr/bin/env sh
# Fail when FUSEKI_DATA_DIR filesystem has less than FUSEKI_MIN_FREE_GB free.
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

_env_get() {
  key="$1"
  if [ ! -f .env ]; then
    return 0
  fi
  grep -E "^${key}=" .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'"
}

DATA_DIR="${FUSEKI_DATA_DIR:-$(_env_get FUSEKI_DATA_DIR)}"
DATA_DIR="${DATA_DIR:-/mnt/graphdb/rdf-store/fuseki}"
MIN_GB="${FUSEKI_MIN_FREE_GB:-$(_env_get FUSEKI_MIN_FREE_GB)}"
MIN_GB="${MIN_GB:-50}"

if [ ! -d "${DATA_DIR}" ]; then
  echo "fuseki_disk_guard: missing data dir ${DATA_DIR}" >&2
  exit 1
fi

avail_kb="$(df -Pk "${DATA_DIR}" | awk 'NR==2 {print $4}')"
avail_gb=$((avail_kb / 1024 / 1024))
fs="$(df -Pk "${DATA_DIR}" | awk 'NR==2 {print $1 " " $6}')"

if [ "${avail_gb}" -lt "${MIN_GB}" ]; then
  echo "fuseki_disk_guard: ${fs} has ${avail_gb}G free (< ${MIN_GB}G required) for ${DATA_DIR}" >&2
  exit 1
fi

echo "fuseki_disk_guard: ok ${avail_gb}G free on ${fs} (${DATA_DIR})"
exit 0
