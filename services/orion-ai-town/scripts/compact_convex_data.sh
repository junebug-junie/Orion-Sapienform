#!/usr/bin/env bash
# Reclaim disk/RAM bloat on the self-hosted Convex backend by exporting live
# data, resetting the on-disk SQLite file, and reimporting.
#
# Why this exists (found live 2026-07-29): self-hosted Convex retains a full
# revision history for every document -- it never overwrites in place. The
# `world`/`engines` rows get rewritten on every single engine tick, so a
# continuously-running town accumulates history forever with no built-in
# compaction. `VACUUM` cannot reclaim this (confirmed live: a 23.56GB file
# only shrank to 22.24GB after a 14-minute VACUUM) because none of those
# revisions are logically deleted from Convex's point of view -- they're
# still "live" as far as SQLite is concerned. The only way to actually shed
# accumulated history is to export current logical state and reimport it
# into a fresh (empty) database, which starts a new history from that point
# while preserving the live game world exactly as it was.
#
# Resetting the SQLite file wipes *everything* stored in it, not just app
# data: the deployed Convex function code/modules and all `npx convex env`
# variables live in the same file. Confirmed live 2026-07-29 -- after the
# first real run of this script, the town had intact data but zero deployed
# functions (every query/mutation call failed with "Could not find function")
# and an empty env list (LLM_API_URL/LLM_MODEL/LLM_EMBEDDING_MODEL gone, so
# NPC chat completion would have failed silently). Both are now captured
# before the reset and restored after, and the world is explicitly
# heartbeated back to "running" at the end so this doesn't depend on someone
# happening to reload the frontend tab afterward.
#
# Usage:
#   compact_convex_data.sh --check            # report current size, exit 0
#   compact_convex_data.sh                    # compact only if over threshold
#   compact_convex_data.sh --force             # compact regardless of size
#
# Env overrides:
#   AITOWN_COMPACT_THRESHOLD_BYTES  (default 5368709120 = 5GiB)
#   AITOWN_COMPACT_HEALTH_TIMEOUT_SEC (default 180)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${ROOT}/upstream"
COMPOSE=(docker compose)
THRESHOLD_BYTES="${AITOWN_COMPACT_THRESHOLD_BYTES:-5368709120}"
HEALTH_TIMEOUT_SEC="${AITOWN_COMPACT_HEALTH_TIMEOUT_SEC:-180}"
JOB_DIR="/tmp/aitown-compact-$(date -u +%Y%m%d-%H%M%S)"

CHECK_ONLY=0
FORCE=0
for arg in "$@"; do
  case "${arg}" in
    --check) CHECK_ONLY=1 ;;
    --force) FORCE=1 ;;
    *) echo "unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ ! -d "${UPSTREAM}/convex" ]]; then
  echo "missing ${UPSTREAM}/convex; clone upstream first (see README.md)" >&2
  exit 1
fi

cd "${ROOT}"

current_size() {
  "${COMPOSE[@]}" exec -T backend stat -c%s /convex/data/db.sqlite3 2>/dev/null | tr -d '\r'
}

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

SIZE_BEFORE="$(current_size || echo 0)"
if [[ -z "${SIZE_BEFORE}" || "${SIZE_BEFORE}" == "0" ]]; then
  echo "could not read current db.sqlite3 size (backend down or path changed)" >&2
  exit 1
fi

log "current db.sqlite3 size: ${SIZE_BEFORE} bytes"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  exit 0
fi

# Resolve the actual data volume from the running container's mounts rather
# than assuming Compose's default "<project>_<volume>" naming -- a
# COMPOSE_PROJECT_NAME override would otherwise make the backup/rename steps
# below silently target a nonexistent (docker-auto-created, empty) volume
# while the real data volume goes untouched.
VOLUME_NAME="$(docker inspect "$("${COMPOSE[@]}" ps -q backend)" \
  --format '{{range .Mounts}}{{if eq .Destination "/convex/data"}}{{.Name}}{{end}}{{end}}')"
if [[ -z "${VOLUME_NAME}" ]]; then
  echo "could not resolve the /convex/data volume from the running backend container -- aborting" >&2
  exit 1
fi

if [[ "${FORCE}" != "1" && "${SIZE_BEFORE}" -lt "${THRESHOLD_BYTES}" ]]; then
  log "below threshold (${THRESHOLD_BYTES} bytes); skipping. Pass --force to override."
  exit 0
fi

mkdir -p "${JOB_DIR}"
PROGRESS="${JOB_DIR}/progress.log"
exec > >(tee -a "${PROGRESS}") 2>&1

log "job dir: ${JOB_DIR}"
log "step 1/7: exporting live data (no downtime)"
(cd "${UPSTREAM}" && npx convex export --path "${JOB_DIR}/export.zip")
EXPORT_SIZE="$(stat -c%s "${JOB_DIR}/export.zip" 2>/dev/null || echo 0)"
if [[ "${EXPORT_SIZE}" -lt 1024 ]]; then
  echo "export.zip suspiciously small (${EXPORT_SIZE} bytes) -- aborting before touching anything" >&2
  exit 1
fi
log "export ok: ${JOB_DIR}/export.zip (${EXPORT_SIZE} bytes)"

log "step 1b/7: capturing deployed env vars (also wiped by the reset)"
(cd "${UPSTREAM}" && npx convex env list) > "${JOB_DIR}/env.backup" || true
log "env backup: ${JOB_DIR}/env.backup ($(wc -l < "${JOB_DIR}/env.backup" 2>/dev/null || echo 0) vars)"

log "step 2/7: stopping backend"
"${COMPOSE[@]}" stop backend

log "step 3/7: snapshotting current db.sqlite3 out of the volume (belt-and-suspenders alongside the export)"
BACKEND_CID="$("${COMPOSE[@]}" ps -a -q backend 2>/dev/null || true)"
if [[ -n "${BACKEND_CID}" ]] && docker cp "${BACKEND_CID}:/convex/data/db.sqlite3" "${JOB_DIR}/db.sqlite3.pre-compact.bak" 2>/dev/null; then
  :
else
  docker run --rm -v "${VOLUME_NAME}:/data" -v "${JOB_DIR}:/backup" alpine \
    cp /data/db.sqlite3 "/backup/db.sqlite3.pre-compact.bak"
fi
log "backup: ${JOB_DIR}/db.sqlite3.pre-compact.bak"

log "step 4/7: renaming (not deleting) db.sqlite3 inside the volume so a fresh one gets created on start"
RENAME_TS="$(date +%s)"
if ! docker run --rm -v "${VOLUME_NAME}:/data" alpine \
     sh -c "test -f /data/db.sqlite3 && mv /data/db.sqlite3 /data/db.sqlite3.pre-compact-${RENAME_TS}"; then
  echo "rename failed (or db.sqlite3 was already missing) -- aborting before starting the backend" >&2
  echo "RECOVERY: the export at ${JOB_DIR}/export.zip and backup at ${JOB_DIR}/db.sqlite3.pre-compact.bak are intact; investigate the volume '${VOLUME_NAME}' manually before restarting backend" >&2
  exit 1
fi
log "renamed to db.sqlite3.pre-compact-${RENAME_TS} inside the volume"

log "step 5/7: starting backend fresh and waiting for health"
"${COMPOSE[@]}" start backend
deadline=$((SECONDS + HEALTH_TIMEOUT_SEC))
until curl -fsS -m 3 http://127.0.0.1:"${PORT:-3210}"/version >/dev/null 2>&1; do
  if (( SECONDS > deadline )); then
    echo "backend did not become healthy within ${HEALTH_TIMEOUT_SEC}s" >&2
    echo "RECOVERY: restore ${JOB_DIR}/db.sqlite3.pre-compact.bak or the in-volume db.sqlite3.pre-compact-${RENAME_TS} and restart" >&2
    exit 1
  fi
  sleep 3
done
log "backend healthy"

log "step 5b/7: redeploying Convex functions (fresh DB has no code until this runs)"
(cd "${UPSTREAM}" && npx convex dev --once)

log "step 5c/7: restoring deployed env vars"
if [[ -s "${JOB_DIR}/env.backup" ]]; then
  (cd "${UPSTREAM}" && npx convex env set --from-file "${JOB_DIR}/env.backup")
else
  log "no env vars to restore (env.backup empty)"
fi

log "step 6/7: reimporting exported data (--replace-all)"
(cd "${UPSTREAM}" && npx convex import --replace-all -y "${JOB_DIR}/export.zip")

log "step 7/7: heartbeating the default world back to running (don't depend on a frontend reload)"
DEFAULT_WORLD_ID="$(cd "${UPSTREAM}" && npx convex run world:defaultWorldStatus 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("worldId",""))' 2>/dev/null || true)"
if [[ -n "${DEFAULT_WORLD_ID}" ]]; then
  (cd "${UPSTREAM}" && npx convex run world:heartbeatWorld "{\"worldId\": \"${DEFAULT_WORLD_ID}\"}") || log "heartbeat failed -- reload the frontend tab manually to resume the world"
else
  log "could not resolve default worldId -- reload the frontend tab manually to resume the world"
fi

SIZE_AFTER="$(current_size || echo 0)"
log "done. size before=${SIZE_BEFORE} after=${SIZE_AFTER} (job dir: ${JOB_DIR})"

cat > "${JOB_DIR}/report.md" <<EOF
# AI Town Convex compaction report

- Started: job dir ${JOB_DIR}
- Size before: ${SIZE_BEFORE} bytes
- Size after: ${SIZE_AFTER} bytes
- Export: ${JOB_DIR}/export.zip (${EXPORT_SIZE} bytes)
- Pre-compact backup: ${JOB_DIR}/db.sqlite3.pre-compact.bak
- In-volume renamed original: db.sqlite3.pre-compact-${RENAME_TS} (safe to delete once verified)

Verify the town looks correct (players/agents/conversations) before deleting backups.
EOF
log "report written: ${JOB_DIR}/report.md"
