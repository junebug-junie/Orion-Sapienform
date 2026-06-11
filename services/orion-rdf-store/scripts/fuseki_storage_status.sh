#!/usr/bin/env sh
# Report active Fuseki mount, dataset size, and stale duplicate if present.
set -eu

SERVICE="${FUSEKI_SERVICE_NAME:-orion-athena-fuseki}"
LUKEWARM="${LUKEWARM_FUSEKI_ROOT:-/mnt/storage-lukewarm/rdf-store/fuseki}"
GRAPHDB="${GRAPHDB_FUSEKI_ROOT:-/mnt/graphdb/rdf-store/fuseki}"

active="$(docker inspect "${SERVICE}" --format '{{range .Mounts}}{{if eq .Destination "/fuseki"}}{{.Source}}{{end}}{{end}}' 2>/dev/null || true)"
if [ -z "${active}" ]; then
  echo "storage_status: ${SERVICE} not running or no /fuseki mount" >&2
  exit 1
fi

echo "active_mount=${active}"
for root in "${LUKEWARM}" "${GRAPHDB}"; do
  if [ -d "${root}/databases/orion" ]; then
    size="$(du -sh "${root}/databases/orion" 2>/dev/null | awk '{print $1}')"
    df_line="$(df -h "${root}" | awk 'NR==2 {print $3 " used / " $2 " total, " $4 " free (" $5 ")"}')"
    stale=""
    if [ "${root}" != "${active}" ]; then
      stale=" STALE"
    fi
    echo "  ${root}${stale}: dataset=${size}, fs=${df_line}"
  else
    echo "  ${root}: (no orion dataset)"
  fi
done
