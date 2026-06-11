#!/usr/bin/env bash
# Finish migration after Fuseki TDB rsync + GraphDB offline export complete.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPORT_FILE="${EXPORT_FILE:-/mnt/graphdb/collapse-mirrors/import/collapse-export.nq}"
FUSEKI_DATA_DIR="${FUSEKI_DATA_DIR:-/mnt/graphdb/rdf-store/fuseki}"

wait_for_rsync() {
  while docker ps --format '{{.Names}}' | grep -q '^fuseki-rsync-to-graphdb$'; do
    du -sh "${FUSEKI_DATA_DIR}" 2>/dev/null || true
    echo "waiting for fuseki-rsync-to-graphdb ..."
    sleep 60
  done
}

echo "==> Ensuring Fuseki is stopped before final data check"
docker stop orion-athena-fuseki 2>/dev/null || true

wait_for_rsync
docker rm fuseki-rsync-to-graphdb 2>/dev/null || true

echo "==> Starting Fuseki on ${FUSEKI_DATA_DIR}"
docker compose -f "${REPO_ROOT}/services/orion-rdf-store/docker-compose.yml" \
  --env-file "${REPO_ROOT}/services/orion-rdf-store/.env" up -d

for _ in $(seq 1 30); do
  if curl -sf http://localhost:3030/ >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "==> Fuseki mount:"
docker inspect orion-athena-fuseki --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'

if [[ -f "${EXPORT_FILE}" ]]; then
  echo "==> Importing GraphDB export into Fuseki"
  EXPORT_FILE="${EXPORT_FILE}" "${REPO_ROOT}/scripts/import_graphdb_export_to_fuseki.sh"
else
  echo "==> No GraphDB export at ${EXPORT_FILE}; skipping import"
fi

echo "==> SPARQL triple count (sample)"
curl -s -X POST http://localhost:3030/orion/sparql \
  -H 'Content-Type: application/sparql-query' \
  --data 'SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o }' | head -20

echo "==> Done"
