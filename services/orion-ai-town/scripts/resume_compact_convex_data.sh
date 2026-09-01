#!/usr/bin/env bash
# Resume a compact_convex_data.sh run that died after step 4.
#
# Why this exists (confirmed live 2026-08-31): compact_convex_data.sh is
# `set -euo pipefail` with no resume path. It died at step 5b with
# `npm error nospc` -- after step 4 had renamed the live database aside and
# step 5 had started the backend on a fresh empty one. The town then served an
# empty world for 33 hours while its container reported "healthy", because the
# Convex healthcheck proves the process answers, not that it has functions or
# data. compact_convex_data.sh now has a step 0 disk preflight so that specific
# cause cannot recur, but ANY failure between step 4 and step 7 leaves the same
# state, and there was no documented way out of it.
#
# The trap this script exists to avoid: re-running compact_convex_data.sh in
# that state is the single worst move available. Its step 1 exports the LIVE
# database -- which is now empty -- into a new job dir, and its step 6 reimports
# that. A recoverable outage becomes permanent data loss. This script instead
# replays steps 5b -> 7 against the ORIGINAL job dir, whose export.zip still
# holds the real data.
#
# Usage:
#   resume_compact_convex_data.sh                 # newest /tmp/aitown-compact-*
#   resume_compact_convex_data.sh <job_dir>       # an explicit job dir
#
# Run as the same user that ran the compaction (the npm/convex cache lives in
# that user's $HOME), NOT as root.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${ROOT}/upstream"
MIN_FREE_BYTES="${AITOWN_RESUME_MIN_FREE_BYTES:-$((15 * 1024 * 1024 * 1024))}"
MIN_EXPORT_DOCS=1000
MAX_JOB_AGE_SEC=$((6 * 3600))
CONFIRM="${AITOWN_RESUME_CONFIRM:-0}"

die() { echo "ABORT: $*" >&2; exit 1; }
ok()  { echo "  [ok] $*"; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

[ "$(id -u)" -ne 0 ] || die "run as the compaction user, not root"

JOB_DIR="${1:-}"
if [[ -z "${JOB_DIR}" ]]; then
  JOB_DIR="$(ls -1dt "${AITOWN_COMPACT_JOB_DIR_BASE:-/tmp}"/aitown-compact-* 2>/dev/null | head -1)"
  [[ -n "${JOB_DIR}" ]] || die "no job dir found; pass one explicitly"
fi
[[ -d "${JOB_DIR}" ]] || die "job dir not found: ${JOB_DIR}"

exec > >(tee -a "${JOB_DIR}/progress.log") 2>&1
echo
log "=== RESUME from step 5b against ${JOB_DIR} ==="

echo "== step 0: preflight =="

# Validate the job-dir artifacts FIRST. They are the only copy of the real data,
# and checking them costs nothing -- there is no point hunting for a node
# runtime before knowing whether there is anything worth importing.
command -v python3 >/dev/null 2>&1 \
  || die "python3 required (used to inspect export.zip and parse world status)"

[[ -s "${JOB_DIR}/export.zip" ]] || die "missing ${JOB_DIR}/export.zip"

# If someone already re-ran compact_convex_data.sh while the database was empty,
# this file is an export OF that empty database and importing it would turn a
# recoverable outage into permanent data loss. Refuse rather than replay it.
DOCS="$(python3 - "${JOB_DIR}/export.zip" <<'PY'
import sys, zipfile
try:
    z = zipfile.ZipFile(sys.argv[1])
except Exception:
    print(0); raise SystemExit(0)
total = 0
for info in z.infolist():
    if not info.filename.endswith("documents.jsonl"):
        continue
    # stream rather than read() whole members: this runs on a host that is, by
    # construction, short on disk and RAM
    with z.open(info) as fh:
        while chunk := fh.read(1 << 20):
            total += chunk.count(b"\n")
print(total)
PY
)"
(( ${DOCS:-0} >= MIN_EXPORT_DOCS )) \
  || die "export.zip holds only ${DOCS:-0} documents (min ${MIN_EXPORT_DOCS}) -- refusing to import what looks like an export of an already-emptied database"
ok "export.zip holds ${DOCS} documents"

[[ -s "${JOB_DIR}/env.backup" ]] || die "missing or empty ${JOB_DIR}/env.backup"
ok "env.backup has $(grep -c . "${JOB_DIR}/env.backup") vars"

# The doc-count guard catches an EMPTY export. It cannot catch a STALE one, and
# a stale export is just as destructive: `import --replace-all` would silently
# roll the world back to whenever that export was taken. If today's failed run
# never wrote an export.zip, an operator following the recovery path lands on
# yesterday's job dir and gets exactly that.
EXPORT_AGE=$(( $(date +%s) - $(stat -c%Y "${JOB_DIR}/export.zip") ))
printf '  [ok] export.zip is %dh %dm old (%s)\n' \
  $(( EXPORT_AGE / 3600 )) $(( (EXPORT_AGE % 3600) / 60 )) \
  "$(date -d "@$(stat -c%Y "${JOB_DIR}/export.zip")" -u '+%Y-%m-%d %H:%M UTC')"
if (( EXPORT_AGE > MAX_JOB_AGE_SEC )) && [[ "${CONFIRM}" != "1" ]]; then
  die "export.zip is older than $(( MAX_JOB_AGE_SEC / 3600 ))h -- importing it would roll the world back that far.
  If this really is the right job dir, re-run with AITOWN_RESUME_CONFIRM=1"
fi

# Resolve a node runtime. A plain non-interactive shell often has none even on
# a host where the compaction previously succeeded, because nvm (and editor
# remote-servers, which bundle their own node) only land on PATH in an
# interactive shell -- exactly how the 2026-08-31 run got one.
NODE=""
if command -v node >/dev/null 2>&1; then
  NODE="$(command -v node)"
else
  # Prefer a real nvm install over an arbitrary `node` that may be a vendored,
  # possibly wrong-arch binary buried in some node_modules tree.
  NODE="$(ls -1 "${HOME}"/.nvm/versions/node/*/bin/node 2>/dev/null | sort -V | tail -1)"
  [[ -n "${NODE}" ]] || \
    NODE="$(find "${HOME}" -maxdepth 6 -type f -name node -perm -u+x 2>/dev/null | head -1)"
fi
[[ -n "${NODE}" && -x "${NODE}" ]] || die "no node runtime found; put node on PATH and re-run"
ok "node $("${NODE}" --version) (${NODE})"

CLI="${UPSTREAM}/node_modules/convex/bin/main.js"
[[ -f "${CLI}" ]] || die "convex CLI missing at ${CLI}; run npm install in ${UPSTREAM}"
CONVEX=("${NODE}" "${CLI}")
ok "convex CLI $(cd "${UPSTREAM}" && "${CONVEX[@]}" --version 2>&1 | tail -1)"

# Check where the writes actually land -- the npm/convex cache under $HOME and
# the job dir -- not a hardcoded /. Same reasoning as the compactor's step 0.
for target in "${HOME:-/root}" "${JOB_DIR}"; do
  FREE="$(df -PB1 "${target}" 2>/dev/null | awk 'NR==2 {print $4}' || true)"
  [[ -n "${FREE}" ]] || die "could not read free space for ${target}"
  (( FREE >= MIN_FREE_BYTES )) || die \
    "only ${FREE} bytes free for ${target} -- need ${MIN_FREE_BYTES} (override with AITOWN_RESUME_MIN_FREE_BYTES)"
  ok "${FREE} bytes free for ${target}"
done

curl -fsS -m 5 "http://127.0.0.1:${PORT:-3210}/version" >/dev/null 2>&1 \
  || die "convex backend not answering on 127.0.0.1:${PORT:-3210}"
ok "backend reachable"

# Confirm the deployment actually IS in the broken state. Everything above
# validates the job dir; nothing so far looks at the live town. Run on a healthy
# deployment by mistake -- a wrong job-dir argument, or the no-argument guess at
# the newest /tmp/aitown-compact-* -- this would redeploy, overwrite the live env
# vars, and `--replace-all` the real data with the export. That is strictly worse
# than the state it is meant to repair, so require positive evidence or consent.
WORLD_PROBE="$(cd "${UPSTREAM}" && "${CONVEX[@]}" run world:defaultWorldStatus 2>/dev/null || true)"
if [[ -n "${WORLD_PROBE}" ]] && grep -q '"worldId"' <<<"${WORLD_PROBE}"; then
  if [[ "${CONFIRM}" != "1" ]]; then
    die "this deployment already answers world:defaultWorldStatus -- functions AND data
  are present, so it is not in the post-step-4 broken state this script repairs.
  Running anyway would --replace-all the live town with ${JOB_DIR}/export.zip.
  If that is genuinely what you want, re-run with AITOWN_RESUME_CONFIRM=1"
  fi
  log "WARNING: deployment looks healthy; proceeding on explicit AITOWN_RESUME_CONFIRM=1"
else
  ok "deployment does not answer world:defaultWorldStatus -- consistent with the broken state"
fi

cd "${UPSTREAM}" || die "cannot cd ${UPSTREAM}"

log "step 5b/7: redeploying Convex functions (fresh DB has no code until this runs)"
"${CONVEX[@]}" dev --once || die "function deploy failed"
ok "functions deployed"

log "step 5c/7: restoring deployed env vars"
"${CONVEX[@]}" env set --from-file "${JOB_DIR}/env.backup" || die "env restore failed"
ok "env vars restored"

if ! python3 "${ROOT}/scripts/check_llm_route_not_circe.py"; then
  log "WARNING: restored LLM_MODEL resolves to circe -- correcting to the safe default"
  "${CONVEX[@]}" env set LLM_MODEL "${AITOWN_LLM_CHAT_ROUTE:-quick_background}"
  python3 "${ROOT}/scripts/check_llm_route_not_circe.py" \
    && log "corrected: LLM_MODEL is no longer on circe" \
    || log "WARNING: could not auto-correct LLM_MODEL off circe"
else
  ok "LLM route is not circe"
fi

log "step 6/7: reimporting exported data (--replace-all)"
"${CONVEX[@]}" import --replace-all -y "${JOB_DIR}/export.zip" || die "import failed -- data NOT restored"
ok "import completed"

log "step 7/7: heartbeating the default world back to running"
WORLD_ID="$("${CONVEX[@]}" run world:defaultWorldStatus 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("worldId",""))' 2>/dev/null || true)"
if [[ -n "${WORLD_ID}" ]]; then
  "${CONVEX[@]}" run world:heartbeatWorld "{\"worldId\": \"${WORLD_ID}\"}" \
    && ok "world ${WORLD_ID} heartbeated to running" \
    || log "heartbeat failed -- reload the frontend tab to resume the world"
else
  log "could not resolve default worldId -- reload the frontend tab to resume the world"
fi

# A populated database is not a running town, and "healthy" is not evidence of
# either -- that is the exact signal that hid the 33-hour outage. defaultWorldStatus
# only answers when functions AND data are both back.
echo "== verify =="
[[ -n "${WORLD_ID}" ]] && ok "defaultWorldStatus answered -> functions deployed AND data present"
echo "  deployed env vars:"; "${CONVEX[@]}" env list 2>/dev/null | sed 's/^/    /'
echo
log "=== RESUME COMPLETE ==="
echo
echo "Confirm the town is live in the UI before reclaiming anything. The engine"
echo "rewrites world/engines every tick, so a growing db.sqlite3 is the check:"
echo "  watch -n5 'docker compose -f ${ROOT}/docker-compose.yml exec -T backend stat -c%s /convex/data/db.sqlite3'"
echo
echo "Only once it is confirmed good, these reclaim the pre-compaction copies:"
echo "  rm -f ${JOB_DIR}/db.sqlite3.pre-compact.bak"
echo "  the in-volume db.sqlite3.pre-compact-<ts> beside the live db.sqlite3"
echo "KEEP ${JOB_DIR}/export.zip and env.backup until then."
