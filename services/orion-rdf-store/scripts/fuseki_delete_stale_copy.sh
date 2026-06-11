#!/usr/bin/env bash
# Remove the non-active Fuseki data tree after migration (typically ~1.3T on lukewarm).
# Requires CONFIRM=1. Never deletes the path currently mounted at /fuseki.
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

SERVICE="${FUSEKI_SERVICE_NAME:-orion-athena-fuseki}"
LUKEWARM="${LUKEWARM_FUSEKI_ROOT:-/mnt/storage-lukewarm/rdf-store/fuseki}"
GRAPHDB="${GRAPHDB_FUSEKI_ROOT:-/mnt/graphdb/rdf-store/fuseki}"
CONFIRM="${CONFIRM:-0}"
DRY_RUN="${DRY_RUN:-0}"

active="$(docker inspect "${SERVICE}" --format '{{range .Mounts}}{{if eq .Destination "/fuseki"}}{{.Source}}{{end}}{{end}}' 2>/dev/null || true)"
if [ -z "${active}" ]; then
  echo "delete_stale: ${SERVICE} has no /fuseki mount — start Fuseki first or set FUSEKI_DATA_DIR" >&2
  exit 1
fi

stale=""
for candidate in "${LUKEWARM}" "${GRAPHDB}"; do
  if [ "${candidate}" != "${active}" ] && [ -d "${candidate}" ]; then
    if [ -n "$(find "${candidate}" -mindepth 1 -print -quit 2>/dev/null)" ]; then
      stale="${candidate}"
      break
    fi
  fi
done

if [ -z "${stale}" ]; then
  echo "delete_stale: no stale copy found (active=${active})"
  exit 0
fi

size="$(du -sh "${stale}" 2>/dev/null | awk '{print $1}')"
echo "active=${active}"
echo "stale=${stale} (${size})"

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1: would remove ${stale}"
  exit 0
fi

if [ "${CONFIRM}" != "1" ]; then
  echo "delete_stale: set CONFIRM=1 to remove ${stale}" >&2
  exit 1
fi

echo "==> Removing stale copy ${stale}"
docker run --rm -v "${stale}:/target" alpine sh -c 'rm -rf /target/* /target/.[!.]* /target/..?* 2>/dev/null; true'

echo "==> Done"
df -h "${stale}" "${active}" 2>/dev/null || true
