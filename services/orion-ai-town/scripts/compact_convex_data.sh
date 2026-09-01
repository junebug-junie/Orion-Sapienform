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
# IF THIS SCRIPT DIES PAST STEP 4, DO NOT RE-RUN IT. Step 4 renames the live
# database aside and step 5 starts the backend on a fresh empty one, so step 1
# of a second run would export that EMPTY database over the good export and
# step 6 would reimport it -- permanent data loss from a recoverable state.
# Run `resume_compact_convex_data.sh <job_dir>` instead: it replays steps
# 5b-7 against the original job dir, whose export.zip still holds the real
# data. (Confirmed live 2026-08-31; see the step 0 preflight comment below.)
#
# Env overrides:
#   AITOWN_COMPACT_THRESHOLD_BYTES  (default 5368709120 = 5GiB)
#   AITOWN_COMPACT_HEALTH_TIMEOUT_SEC (default 180)
#   AITOWN_COMPACT_JOB_DIR_BASE     (default /tmp) -- step 3 writes a full copy
#     of the database here, so point this at a filesystem with room for it when
#     /tmp is small or shares a disk with the OS. The step 0 preflight checks
#     whichever filesystem this resolves to.
#   AITOWN_COMPACT_SKIP_RAW_BACKUP  (default 0) -- skip step 3's raw db copy and
#     relax the preflight to export-sized headroom. Recovery then rests on
#     export.zip plus step 4's in-volume rename, which are two artifacts, not
#     zero. Use when the database is too large a fraction of its filesystem for
#     a full copy to fit -- otherwise the gate blocks the only operation that
#     shrinks the database, precisely when it is most needed.
#   AITOWN_COMPACT_ALLOW_UNKNOWN_HOME_FREE (default 0) -- proceed when free space
#     under $HOME cannot be read. Off by default: that check guards step 5b,
#     which runs after the database has already been reset.
#
# These are read from the ENVIRONMENT, not from services/orion-ai-town/.env --
# this is a host script and never sources that file. Export them in the crontab
# entry or the invoking shell.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${ROOT}/upstream"
COMPOSE=(docker compose)
THRESHOLD_BYTES="${AITOWN_COMPACT_THRESHOLD_BYTES:-5368709120}"
HEALTH_TIMEOUT_SEC="${AITOWN_COMPACT_HEALTH_TIMEOUT_SEC:-180}"
JOB_DIR="${AITOWN_COMPACT_JOB_DIR_BASE:-/tmp}/aitown-compact-$(date -u +%Y%m%d-%H%M%S)"

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

# RESUMABLE: set to 1 once step 4 has renamed the live database aside.
# COMPLETED: set to 1 only after step 7 and the report have been written.
#
# The warning gates on COMPLETED, not on the exit status, because `$?` inside an
# EXIT trap is the status of the last COMPLETED command, not the signal that
# killed the script. Verified on this host: with `trap on_exit EXIT` alone, all
# of SIGINT/SIGTERM/SIGHUP enter the trap with rc=0, so a status-gated warning
# stays silent in the three cases most likely to happen for real -- Ctrl-C on a
# hung step 5b (the obvious operator reaction to this exact incident), a dropped
# SSH session, and a cron/systemd timeout. SIGINT additionally exits 0, so a cron
# wrapper would record the run as a success while the town sat empty.
RESUMABLE=0
COMPLETED=0
on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if (( RESUMABLE == 1 )) && (( COMPLETED == 0 )); then
    # a signal death is not a success, whatever $? happens to hold
    (( rc == 0 )) && rc=1
    echo "" >&2
    echo "FAILED after step 4 -- the live database has already been reset and" >&2
    echo "the backend may be serving an EMPTY world. Do NOT re-run this script:" >&2
    echo "its step 1 would export that empty database over the good export at" >&2
    echo "${JOB_DIR}/export.zip and step 6 would reimport it. Recover with:" >&2
    echo "  ${ROOT}/scripts/resume_compact_convex_data.sh ${JOB_DIR}" >&2
  fi
  exit "${rc}"
}
trap on_exit EXIT INT TERM HUP

SIZE_BEFORE="$(current_size || echo 0)"
if [[ -z "${SIZE_BEFORE}" || "${SIZE_BEFORE}" == "0" ]]; then
  echo "could not read current db.sqlite3 size (backend down or path changed)" >&2
  exit 1
fi

log "current db.sqlite3 size: ${SIZE_BEFORE} bytes"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  exit 0
fi

# Resolve the actual data mount from the running container's mounts rather
# than assuming Compose's default "<project>_<volume>" naming -- a
# COMPOSE_PROJECT_NAME override would otherwise make the backup/rename steps
# below silently target a nonexistent (docker-auto-created, empty) volume
# while the real data volume goes untouched. Falls back to the mount's
# .Source (host path) when it's a bind mount rather than a named volume --
# `docker run -v <arg>:/data` accepts either a volume name or a host path
# interchangeably, so the rest of the script is unaffected either way.
VOLUME_NAME="$(docker inspect "$("${COMPOSE[@]}" ps -q backend)" \
  --format '{{range .Mounts}}{{if eq .Destination "/convex/data"}}{{if .Name}}{{.Name}}{{else}}{{.Source}}{{end}}{{end}}{{end}}')"
if [[ -z "${VOLUME_NAME}" ]]; then
  echo "could not resolve the /convex/data mount from the running backend container -- aborting" >&2
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

# Confirmed live 2026-08-31: this script wrote an 11GB db.sqlite3 backup into
# /tmp on a filesystem that did not have 11GB free, filled the root disk to
# 100%, and then died at step 5b with `npm error nospc` -- AFTER step 4 had
# already renamed the live database aside and step 5 had started the backend on
# a fresh empty one. AI Town then served an empty world for 33 hours while its
# container reported "healthy". Nothing in this script looked at df.
#
# This has to fail BEFORE step 1, not at the point of the write: steps 1-3 are
# recoverable, but past step 4 the only way back is a manual reimport, and
# re-running this script at that point would export the now-empty database over
# the good export and destroy the data for real. Two filesystems matter -- the
# one holding JOB_DIR (step 3 writes a full copy of the database there) and the
# one holding $HOME (step 5b's npm/convex cache is what actually hit ENOSPC).
log "step 0/7: disk preflight"
# `|| true` is load-bearing under `set -euo pipefail`: df exits non-zero on an
# unreadable path, pipefail promotes that to the pipeline's status, and the
# command substitution would then abort the whole script instead of yielding an
# empty string for the caller to handle. A preflight that kills the run it is
# supposed to be protecting is worse than no preflight.
avail_bytes() { df -PB1 "$1" 2>/dev/null | awk 'NR==2 {print $4}' || true; }

device_of() { stat -c%d "$1" 2>/dev/null || true; }

# step 3 copies the whole live database, so budget its size plus 5% and 512MiB.
# AITOWN_COMPACT_SKIP_RAW_BACKUP=1 skips that copy: it is explicitly
# belt-and-suspenders alongside export.zip, which is the real recovery artifact
# and is orders of magnitude smaller (10.2GB db -> 314MB export, 2026-08-31).
# Without this door, the gate refuses a compaction once the database passes ~48%
# of its filesystem -- i.e. it blocks the only thing that shrinks the database,
# exactly when that is most needed.
SKIP_RAW_BACKUP="${AITOWN_COMPACT_SKIP_RAW_BACKUP:-0}"
if [[ "${SKIP_RAW_BACKUP}" == "1" ]]; then
  BACKUP_NEED=$(( SIZE_BEFORE / 10 + 536870912 ))
  log "preflight: AITOWN_COMPACT_SKIP_RAW_BACKUP=1 -- step 3 raw copy will be skipped"
else
  BACKUP_NEED=$(( SIZE_BEFORE + SIZE_BEFORE / 20 + 536870912 ))
fi

# step 5b runs `npx convex dev --once`, which writes to the npm/convex cache
# under $HOME. That write is what actually failed in the 2026-08-31 incident.
HOME_NEED=2147483648
HOME_DIR="${HOME:-/root}"

JOB_AVAIL="$(avail_bytes "${JOB_DIR}")"
if [[ -z "${JOB_AVAIL}" ]]; then
  echo "could not read free space for ${JOB_DIR} -- aborting" >&2
  exit 1
fi

# Checking the two demands independently is not enough when they land on the
# SAME filesystem -- which is the incident's own topology and, verified
# 2026-09-01, still true on circe today (/tmp, $HOME and / are all one device).
# Independent checks each pass while their sum does not fit, step 3 consumes the
# shared pool, and step 5b hits ENOSPC with the preflight showing green. Sum them.
HOME_AVAIL="$(avail_bytes "${HOME_DIR}")"
JOB_DEV="$(device_of "${JOB_DIR}")"
HOME_DEV="$(device_of "${HOME_DIR}")"
SHARED=0
if [[ -n "${JOB_DEV}" && -n "${HOME_DEV}" && "${JOB_DEV}" == "${HOME_DEV}" ]]; then
  SHARED=1
fi

if (( SHARED == 1 )); then
  COMBINED_NEED=$(( BACKUP_NEED + HOME_NEED ))
  log "preflight: ${JOB_DIR} and ${HOME_DIR} are the same filesystem (dev ${JOB_DEV}) -- requirements summed"
  if (( JOB_AVAIL < COMBINED_NEED )); then
    echo "insufficient space for the compaction" >&2
    echo "  ${JOB_DIR} and ${HOME_DIR} share one filesystem: ${JOB_AVAIL} bytes free" >&2
    echo "  need ${COMBINED_NEED} = ${BACKUP_NEED} (step 3 backup) + ${HOME_NEED} (step 5b npm cache)" >&2
    echo "  db is ${SIZE_BEFORE} bytes" >&2
    echo "  free space, set AITOWN_COMPACT_JOB_DIR_BASE to another filesystem," >&2
    echo "  or set AITOWN_COMPACT_SKIP_RAW_BACKUP=1 to skip the redundant raw copy" >&2
    exit 1
  fi
  log "preflight: ${JOB_AVAIL} bytes free, need ${COMBINED_NEED} combined"
else
  if (( JOB_AVAIL < BACKUP_NEED )); then
    echo "insufficient space for the step 3 backup" >&2
    echo "  ${JOB_DIR}: ${JOB_AVAIL} bytes free, need ${BACKUP_NEED} (db is ${SIZE_BEFORE})" >&2
    echo "  free space, set AITOWN_COMPACT_JOB_DIR_BASE to a bigger filesystem," >&2
    echo "  or set AITOWN_COMPACT_SKIP_RAW_BACKUP=1 to skip the redundant raw copy" >&2
    exit 1
  fi
  log "preflight: ${JOB_DIR} has ${JOB_AVAIL} bytes free, need ${BACKUP_NEED}"

  # Fail CLOSED on an unreadable $HOME. This check guards the step that actually
  # broke, and it runs after the database has been reset, so "could not measure"
  # must not read as "fine". AITOWN_COMPACT_ALLOW_UNKNOWN_HOME_FREE=1 overrides.
  if [[ -z "${HOME_AVAIL}" ]]; then
    if [[ "${AITOWN_COMPACT_ALLOW_UNKNOWN_HOME_FREE:-0}" == "1" ]]; then
      log "WARNING: free space under ${HOME_DIR} is unreadable; proceeding on explicit override"
    else
      echo "could not read free space under ${HOME_DIR} -- aborting" >&2
      echo "  step 5b's npm cache writes there, after the database has been reset" >&2
      echo "  set AITOWN_COMPACT_ALLOW_UNKNOWN_HOME_FREE=1 to proceed anyway" >&2
      exit 1
    fi
  elif (( HOME_AVAIL < HOME_NEED )); then
    echo "insufficient space under ${HOME_DIR} for the step 5b function redeploy" >&2
    echo "  ${HOME_AVAIL} bytes free, need ${HOME_NEED}" >&2
    echo "  step 5b runs after the live database has already been reset -- refusing to start" >&2
    exit 1
  fi
  log "preflight: ${HOME_DIR} has ${HOME_AVAIL:-unknown} bytes free, need ${HOME_NEED}"
fi

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

if [[ "${SKIP_RAW_BACKUP}" == "1" ]]; then
  # Step 4 renames rather than deletes, so the pre-compact database still exists
  # in the volume as db.sqlite3.pre-compact-<ts> even without this copy -- along
  # with export.zip, that is two independent recovery artifacts, not zero.
  log "step 3/7: SKIPPED (AITOWN_COMPACT_SKIP_RAW_BACKUP=1); recovery rests on"
  log "          ${JOB_DIR}/export.zip and the in-volume rename from step 4"
else
  log "step 3/7: snapshotting current db.sqlite3 out of the volume (belt-and-suspenders alongside the export)"
  BACKEND_CID="$("${COMPOSE[@]}" ps -a -q backend 2>/dev/null || true)"
  if [[ -n "${BACKEND_CID}" ]] && docker cp "${BACKEND_CID}:/convex/data/db.sqlite3" "${JOB_DIR}/db.sqlite3.pre-compact.bak" 2>/dev/null; then
    :
  else
    docker run --rm -v "${VOLUME_NAME}:/data" -v "${JOB_DIR}:/backup" alpine \
      cp /data/db.sqlite3 "/backup/db.sqlite3.pre-compact.bak"
  fi
  log "backup: ${JOB_DIR}/db.sqlite3.pre-compact.bak"
fi

log "step 4/7: renaming (not deleting) db.sqlite3 inside the volume so a fresh one gets created on start"
RENAME_TS="$(date +%s)"
if ! docker run --rm -v "${VOLUME_NAME}:/data" alpine \
     sh -c "test -f /data/db.sqlite3 && mv /data/db.sqlite3 /data/db.sqlite3.pre-compact-${RENAME_TS}"; then
  echo "rename failed (or db.sqlite3 was already missing) -- aborting before starting the backend" >&2
  echo "RECOVERY: the export at ${JOB_DIR}/export.zip and backup at ${JOB_DIR}/db.sqlite3.pre-compact.bak are intact; investigate the volume '${VOLUME_NAME}' manually before restarting backend" >&2
  exit 1
fi
log "renamed to db.sqlite3.pre-compact-${RENAME_TS} inside the volume"

# Point of no return. From here the live database is gone and only the job dir
# can restore it, so every subsequent failure -- not just the two that happen to
# have an explicit RECOVERY line -- must say so. The 2026-08-31 outage failed at
# step 5b, which had no such line, and the state it left behind looked to the
# next operator like something a re-run would fix. It is not.
RESUMABLE=1

log "step 5/7: starting backend fresh and waiting for health"
"${COMPOSE[@]}" start backend
deadline=$((SECONDS + HEALTH_TIMEOUT_SEC))
until curl -fsS -m 3 http://127.0.0.1:"${PORT:-3210}"/version >/dev/null 2>&1; do
  if (( SECONDS > deadline )); then
    echo "backend did not become healthy within ${HEALTH_TIMEOUT_SEC}s" >&2
    echo "the pre-compact database is still available as ${JOB_DIR}/db.sqlite3.pre-compact.bak" >&2
    echo "(when step 3 ran) and as the in-volume db.sqlite3.pre-compact-${RENAME_TS}." >&2
    echo "See the recovery instructions printed below before doing anything else." >&2
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

# This restore is a straight replay of whatever was backed up in step 1b --
# including a stale LLM_MODEL if one was already silently pointed at circe
# before this script ever ran (confirmed live 2026-07-30: exactly this
# happened, undetected, for weeks). Don't abort the compaction over it --
# data reimport (step 6) still needs to happen regardless -- but self-heal
# it here rather than silently perpetuating the same drift on every future
# compaction. See check_llm_route_not_circe.py's docstring for the incident.
if ! python3 "${ROOT}/scripts/check_llm_route_not_circe.py"; then
  log "WARNING: restored LLM_MODEL resolves to circe -- correcting to the safe default"
  (cd "${UPSTREAM}" && npx convex env set LLM_MODEL "${AITOWN_LLM_CHAT_ROUTE:-quick_background}")
  if python3 "${ROOT}/scripts/check_llm_route_not_circe.py"; then
    log "corrected: LLM_MODEL is no longer on circe"
  else
    log "WARNING: could not auto-correct LLM_MODEL off circe -- investigate manually before relying on NPC dialogue"
  fi
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

# Every step landed; the exit trap must not warn about a resume that isn't needed.
COMPLETED=1
