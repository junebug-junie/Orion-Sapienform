#!/usr/bin/env bash
# GPU pool stage 7.1 bake-off on circe gpu2: Bonsai at --parallel 2 x 131K, flash attention on.
# Spec: docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md (acceptance checks 1-2).
# Field note: docs/2026-10-01-gpu-pool-stage7-1-bakeoff.md (fill from the run's fieldnote_draft.md)
#
#   stage7_1_bakeoff.sh preflight             read-only: what is on gpu2, would `run` start?
#   stage7_1_bakeoff.sh run --yes [--with-q4] the bake-off (~60-80 min; +~50 min with --with-q4)
#   stage7_1_bakeoff.sh cleanup               stop the worker, release the hold, resume the pool
#
# Run it inside tmux (`tmux new -s bench71 '...'`): an ssh drop sends HUP, which aborts the run
# (safely: cleanup still runs). Needs docker + nvidia-smi + python3 (stdlib) on circe. No sudo.
# circe has no venv, so pool controls run inside the Bonsai image (scripts/bench/stage7_1_pool_ctl.py).
#
# What `run` changes, in order, and undoes on EVERY exit (trap EXIT/INT/TERM/HUP):
#   1. takes an operator hold on the diffusion role. diffusion-host loads ~24 GB on demand; the hold
#      makes that work queue instead of running out of memory next to Bonsai (world-model work waits
#      too: serialize_with). It comes first because the pool refuses operator holds while paused.
#   2. pauses pool actuation (no model load/unload anywhere until resumed),
#   3. starts the Bonsai worker on gpu2 (port 8017, LLM_ROLE=bonsai-bakeoff: the pool lists it as
#      unclaimed and routes nothing to it), runs the bench and the bleed canary against it,
#   4. cleanup: stop the worker, release the hold, resume actuation (only a pause the pool records
#      as this bake-off's), print pool state. If the worker cannot be stopped, the hold and the
#      pause are KEPT (resuming next to a stuck 24 GB worker would let the pool load onto it) and
#      the manual steps are printed.
# It never edits a .env. BONSAI_CUDA_VISIBLE_DEVICES=2 is forced in the compose environment (it
# overrides the service .env's value), checked in `compose config` before `up`, and checked again
# with docker inspect and gpu2's memory growth before any request is sent.
#
# Docker compose is called directly, not through scripts/safe_docker_build.sh: this never builds
# (--no-build, an image that already exists), so the shared-checkout rebuild incident that wrapper
# guards against cannot happen, and the wrapper refuses circe's primary checkout where the service
# .env lives.
#
# Env: DRY_RUN=1 prints every mutating step instead of doing it (read-only probes still run).
#      BENCH_FAIL_AT=<step> simulates a failure at a named step (tests). DRY_RUN_STEP_SLEEP=<s>
#      makes dry-run bench/canary steps sleep (signal tests). DRY_CTL_<ACTION>='<json>' and
#      DRY_CTL_<ACTION>_RC=<n> fake one pool-control reply in a dry run (tests).
#      RESULTS_ROOT (default ~/stage7_1_bakeoff), POOL_URL, BAKE_PORT, BONSAI_IMAGE, Q4_IMAGE,
#      BONSAI_ENV_FILE (default: the service .env of this worktree, else of the primary checkout).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SVC=orion-llamacpp-bonsai-host
COMPOSE="$ROOT/services/$SVC/docker-compose.yml"
CLIENT="$ROOT/scripts/bench/stage7_1_client.py"
# AGENTS.md: the bus is always athena's redis over tailscale.
ORION_BUS_URL="redis://100.92.216.81:6379/0"
POOL_URL="${POOL_URL:-http://100.92.216.81:8127}"
BAKE_PORT="${BAKE_PORT:-8017}"
CARD_INDEX=2
BONSAI_IMAGE="${BONSAI_IMAGE:-llamacpp-bonsai-prism:server-local-volta}"
BONSAI_PROFILE="ternary-bonsai2-27b-pq2-v100-32gb-circe-np4"
Q4_IMAGE="${Q4_IMAGE:-orion-llamacpp-host:0.1.0}"
Q4_PROFILE="qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex"
CTL_IMAGE="${CTL_IMAGE:-$BONSAI_IMAGE}"
PROJECT_NAME=orion-bench71
CONTAINER="$PROJECT_NAME-bonsai-worker"
ACTOR=stage7-1-bakeoff
MAX_GPU2_MIB="${MAX_GPU2_MIB:-4096}"
MIN_LOAD_MIB="${MIN_LOAD_MIB:-8000}"   # gpu2 must grow this much once the worker is up
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/stage7_1_bakeoff}"
DRY_RUN="${DRY_RUN:-0}"
# Markers for "this run paused the pool" / "this run holds lease X", read by cleanup. A dry run keeps
# its own, so it exercises the same trap path without ever making a real cleanup resume anything.
STATE_DIR="$RESULTS_ROOT/.state"
if [ "$DRY_RUN" = "1" ]; then STATE_DIR="$RESULTS_ROOT/.state-dryrun"; fi
BENCH_FAIL_AT="${BENCH_FAIL_AT:-}"
BONSAI_CANARY_MIN="${BONSAI_CANARY_MIN:-40}"
Q4_CANARY_MIN="${Q4_CANARY_MIN:-35}"

RUN_DIR=""
CHILD=""
SAMPLER=""
CLEANED=0
CTL_REPLY="$(mktemp)"

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

say() {
    local line
    line="$(date -u +%H:%M:%SZ) $*"
    echo "$line"
    if [ -n "$RUN_DIR" ]; then echo "$line" >> "$RUN_DIR/run.log"; fi
}

# Every command that changes anything goes through mut: DRY_RUN prints it instead.
mut() {
    if [ "$DRY_RUN" = "1" ]; then
        say "DRY_RUN: $*"
        return 0
    fi
    "$@"
}

fail_point() {
    if [ -n "$BENCH_FAIL_AT" ] && [ "$BENCH_FAIL_AT" = "$1" ]; then
        say "SIMULATED FAILURE at $1 (BENCH_FAIL_AT)"
        exit 1
    fi
}

die() { say "ABORT: $*"; exit 1; }

# One lock for run and cleanup: a second run (or a cleanup from another terminal) must not stop
# this run's worker, release its hold, or resume its pause.
take_lock() {
    mkdir -p "$STATE_DIR"
    exec 9>"$STATE_DIR/lock"
    if ! flock -n 9; then
        say "REFUSE: another bake-off run or cleanup is active (lock $STATE_DIR/lock)"
        exit 4
    fi
}

svc_env_file() {
    # BONSAI_ENV_FILE if set, else the worktree's own service .env, else the primary checkout's (a
    # fresh worktree has none: it is gitignored). Card and port are forced below either way.
    if [ -n "${BONSAI_ENV_FILE:-}" ]; then echo "$BONSAI_ENV_FILE"; return; fi
    local f="$ROOT/services/$SVC/.env" common
    if [ -f "$f" ]; then echo "$f"; return; fi
    common="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
    if [ -n "$common" ] && [ -f "$(dirname "$common")/services/$SVC/.env" ]; then
        echo "$(dirname "$common")/services/$SVC/.env"
        return
    fi
    echo ""
}

pool_json() { curl -fsS -m 10 "$POOL_URL/v1/pool"; }

gpu2_used_mib() {
    nvidia-smi -i "$CARD_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

port_busy() { curl -fsS -m 3 -o /dev/null "http://127.0.0.1:$BAKE_PORT/health" 2>/dev/null; }

container_exists() { docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; }

# Pool control inside the Bonsai image (circe has no venv). The pool's JSON reply goes to
# $CTL_REPLY (a file, so log lines can never get mixed into what callers parse) and to the log.
ctl() {
    local action="$1" rc; shift
    if [ "$DRY_RUN" = "1" ]; then
        say "DRY_RUN: pool $action $* (docker run $CTL_IMAGE scripts/bench/stage7_1_pool_ctl.py)"
        local var="DRY_CTL_${action^^}" rcvar="DRY_CTL_${action^^}_RC"
        if [ -n "${!var:-}" ]; then
            echo "${!var}" > "$CTL_REPLY"
            return "${!rcvar:-0}"
        fi
        case "$action" in
            pause) echo '{"ok": true, "reason": "paused", "detail": {}}' > "$CTL_REPLY" ;;
            hold) echo '{"ok": true, "reason": null, "detail": {"lease_id": "dry-run-lease", "status": "granted"}}' > "$CTL_REPLY" ;;
            *) echo '{"ok": true, "reason": "dry_run", "detail": {}}' > "$CTL_REPLY" ;;
        esac
        return 0
    fi
    docker run --rm --network host -v "$ROOT:/repo:ro" -e ORION_BUS_URL="$ORION_BUS_URL" \
        -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 "$CTL_IMAGE" \
        /repo/scripts/bench/stage7_1_pool_ctl.py "$action" --actor "$ACTOR" "$@" > "$CTL_REPLY"
    rc=$?
    say "pool $action -> $(tail -c 400 "$CTL_REPLY")"
    return $rc
}

json_get() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1], {}, {'d': d}))" "$1"; }

# mine / other / running / unknown: who the POOL says paused it (not our marker file).
pool_paused_by() {
    local pj out
    pj="$(mktemp)"
    if pool_json > "$pj" 2>/dev/null; then
        out="$(python3 "$CLIENT" paused-by --pool-json "$pj" --actor "$ACTOR")"
    else
        out=unknown
    fi
    rm -f "$pj"
    echo "${out:-unknown}"
}

preflight() {
    local pj used busy=""
    pj="$(mktemp)"
    if ! pool_json > "$pj"; then
        rm -f "$pj"
        say "REFUSE: cannot read pool state from $POOL_URL/v1/pool"
        return 3
    fi
    used="$(gpu2_used_mib)"
    if port_busy; then busy="--bake-port-busy"; fi
    say "gpu2 processes:"
    nvidia-smi -i "$CARD_INDEX" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    if container_exists; then say "REFUSE: container $CONTAINER already exists (run cleanup)"; rm -f "$pj"; return 3; fi
    python3 "$CLIENT" preflight-check --pool-json "$pj" --gpu2-used-mib "${used:-unknown}" \
        --max-gpu2-mib "$MAX_GPU2_MIB" --actor "$ACTOR" $busy
    local rc=$?
    rm -f "$pj"
    return $rc
}

write_override() {
    local ctx="$1"
    cat > "$RUN_DIR/override-$2.yml" <<EOF
# generated by stage7_1_bakeoff.sh: 2 slots, ctx split evenly by llama-server
services:
  bonsai-worker:
    environment:
      - SERVICE_NAME=orion-stage7-1-bakeoff
      - LLAMACPP_N_PARALLEL_OVERRIDE=2
      - LLAMACPP_CTX_SIZE_OVERRIDE=$ctx
EOF
}

compose() {   # label image profile envfile compose-args...
    local label="$1" image="$2" profile="$3" envf="$4"; shift 4
    mut env PROJECT="$PROJECT_NAME" BONSAI_CUDA_VISIBLE_DEVICES="$CARD_INDEX" BONSAI_HOST_PORT="$BAKE_PORT" \
        BONSAI_LLAMACPP_IMAGE="$image" BONSAI_PROFILE_NAME="$profile" \
        docker compose -p "$PROJECT_NAME" --env-file "$envf" -f "$COMPOSE" -f "$RUN_DIR/override-$label.yml" "$@"
}

start_worker() {   # label image profile ctx
    local label="$1" image="$2" profile="$3" ctx="$4" envf before=""
    envf="$(svc_env_file)"
    [ -n "$envf" ] || die "no services/$SVC/.env in this worktree or the primary checkout (set BONSAI_ENV_FILE)"
    write_override "$ctx" "$label"
    say "starting $label: image=$image profile=$profile --parallel 2 ctx=$ctx card=gpu$CARD_INDEX port=$BAKE_PORT env=$envf"
    if [ "$DRY_RUN" != "1" ]; then
        compose "$label" "$image" "$profile" "$envf" config > "$RUN_DIR/$label/compose-config.yml" \
            || die "compose config failed"
        grep -q "CUDA_VISIBLE_DEVICES_OVERRIDE: \"$CARD_INDEX\"" "$RUN_DIR/$label/compose-config.yml" \
            || die "compose config does not put the worker on card $CARD_INDEX: refusing to start"
        before="$(gpu2_used_mib)"
    fi
    compose "$label" "$image" "$profile" "$envf" up -d --no-build bonsai-worker || die "compose up failed"
    fail_point "after_start_$label"
    if [ "$DRY_RUN" = "1" ]; then return 0; fi
    local dev
    dev="$(docker inspect "$CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^CUDA_VISIBLE_DEVICES_OVERRIDE=' | cut -d= -f2)"
    [ "$dev" = "$CARD_INDEX" ] || die "container card is '$dev', expected $CARD_INDEX: refusing to send load"
    local i
    for i in $(seq 1 120); do
        if port_busy; then break; fi
        if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
            docker logs --tail 40 "$CONTAINER" > "$RUN_DIR/$label/boot-fail.log" 2>&1
            die "$label worker exited during boot (see $label/boot-fail.log)"
        fi
        sleep 10
    done
    port_busy || die "$label worker not healthy after 20 min"
    local after
    after="$(gpu2_used_mib)"
    say "gpu2 memory: ${before:-?} MiB before, ${after:-?} MiB with the $label worker up"
    if [ -n "$before" ] && [ -n "$after" ] && [ $((after - before)) -lt "$MIN_LOAD_MIB" ]; then
        die "gpu2 grew only $((after - before)) MiB: the worker is not on gpu2"
    fi
    docker logs "$CONTAINER" 2>&1 | grep -m3 -E 'llama-server|--parallel' > "$RUN_DIR/$label/argv.txt" || true
    curl -fsS -m 10 "http://127.0.0.1:$BAKE_PORT/props" > "$RUN_DIR/$label/props.json"
    local slots nctx
    slots="$(json_get "d.get('total_slots')" < "$RUN_DIR/$label/props.json")"
    nctx="$(json_get "d['default_generation_settings'].get('n_ctx')" < "$RUN_DIR/$label/props.json")"
    say "$label up: slots=$slots ctx/slot=$nctx build=$(json_get "d.get('build_info')" < "$RUN_DIR/$label/props.json")"
    [ "$slots" = "2" ] && [ "$nctx" = "$((ctx / 2))" ] || die "$label layout is $slots x $nctx, expected 2 x $((ctx / 2))"
    if ! grep -q -- '--flash-attn' "$RUN_DIR/$label/argv.txt"; then
        say "WARNING: --flash-attn not seen in the $label launch line (argv.txt); recorded, not fatal"
    fi
    nvidia-smi -i "$CARD_INDEX" --query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw \
        --format=csv,noheader -l 5 > "$RUN_DIR/$label/gpu2.csv" 2>&1 &
    SAMPLER=$!
}

# 0 = the worker is gone; 1 = it could not be removed (cleanup then keeps the hold and the pause).
stop_worker() {
    if [ -n "$SAMPLER" ]; then kill "$SAMPLER" 2>/dev/null; SAMPLER=""; fi
    if [ "$DRY_RUN" = "1" ]; then
        say "stopping $CONTAINER"
        mut docker stop -t 30 "$CONTAINER"
        mut docker rm -f "$CONTAINER"
        return 0
    fi
    if container_exists; then
        say "stopping $CONTAINER"
        docker stop -t 30 "$CONTAINER" >/dev/null 2>&1 || docker kill "$CONTAINER" >/dev/null 2>&1
        docker rm -f "$CONTAINER" >/dev/null 2>&1
    fi
    if container_exists; then
        say "ERROR: $CONTAINER could not be removed"
        return 1
    fi
    local i used
    for i in $(seq 1 24); do
        used="$(gpu2_used_mib)"
        if [ -n "$used" ] && [ "$used" -le "$MAX_GPU2_MIB" ]; then return 0; fi
        sleep 5
    done
    say "WARNING: worker removed but gpu2 still shows ${used:-?} MiB (something else? check nvidia-smi -i 2)"
    return 0
}

# Long steps run in the background + wait, so INT/TERM/HUP reach the trap at once instead of after
# a 30-minute child finishes.
run_step() {
    if [ "$DRY_RUN" = "1" ]; then
        say "DRY_RUN: $*"
        [ -n "${DRY_RUN_STEP_SLEEP:-}" ] || return 0
        sleep "$DRY_RUN_STEP_SLEEP" &
    else
        "$@" &
    fi
    CHILD=$!
    wait "$CHILD"
    local rc=$?
    CHILD=""
    return $rc
}

pass_run() {   # label image profile ctx depths canary_minutes
    local label="$1" depths="$5" minutes="$6"
    mkdir -p "$RUN_DIR/$label"
    start_worker "$1" "$2" "$3" "$4"
    say "== $label bench (depths $depths): 1 and 2 concurrent runs"
    run_step python3 -u "$CLIENT" bench --url "http://127.0.0.1:$BAKE_PORT" --out "$RUN_DIR/$label/bench.json" \
        --log "$RUN_DIR/$label/bench.log" --depths "$depths" || say "WARNING: $label bench exited non-zero"
    fail_point "after_bench_$label"
    say "== $label canary (cap ${minutes} min)"
    run_step python3 -u "$CLIENT" canary --url "http://127.0.0.1:$BAKE_PORT" --out "$RUN_DIR/$label/canary.json" \
        --log "$RUN_DIR/$label/canary.log" --deadline-min "$minutes" || say "WARNING: $label canary exited non-zero"
    stop_worker || die "could not stop the $label worker"
}

print_pool() {
    local pj
    pj="$(mktemp)"
    if ! pool_json > "$pj" 2>/dev/null; then say "pool state: UNREADABLE from $POOL_URL"; rm -f "$pj"; return; fi
    python3 - "$pj" "$ACTOR" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
p = d.get("actuation_paused")
print(f"  pool actuation: {'PAUSED by ' + str(p.get('by')) + ' since ' + str(p.get('since')) if p else 'running'}")
for c in d.get("cards") or []:
    if c.get("card") == "gpu2":
        print(f"  gpu2: swap_state={c.get('swap_state')} swapped_in={c.get('swapped_in')}")
for r in d.get("roles") or []:
    if r.get("role") in ("agent-gpu2", "diffusion", "world"):
        print(f"  role {r['role']}: {r.get('status')} slots={r.get('slots')}")
mine = [l for l in d.get("leases") or [] if l.get("holder") == f"operator:{sys.argv[2]}" and l.get("status") in ("granted", "recalling", "queued")]
print(f"  bake-off holds still live: {[l['lease_id'] for l in mine] or 'none'}")
PY
    rm -f "$pj"
}

manual_steps() {
    say "MANUAL STEPS:"
    say "  docker rm -f $CONTAINER && nvidia-smi -i $CARD_INDEX"
    say "  $0 cleanup        # releases the hold and resumes the pool once the worker is gone"
    say "  or the Hub GPU pool panel: release the operator:$ACTOR hold on diffusion, then Resume."
}

cleanup() {
    [ "$CLEANED" = "1" ] && return
    CLEANED=1
    # A second Ctrl-C / TERM / ssh hangup must not cut cleanup short: release and resume come last.
    trap 'say "signal ignored: cleanup in progress"' INT TERM HUP
    trap - EXIT
    say "== cleanup (always runs)"
    if [ -n "$CHILD" ]; then kill "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null; CHILD=""; fi
    if ! stop_worker; then
        say "KEEPING the diffusion hold and the pause: the worker is still on gpu2, and releasing either"
        say "would let the pool load ~24 GB onto a card that is already full."
        print_pool
        manual_steps
        rm -f "$CTL_REPLY"
        return
    fi
    # Release: the recorded lease, plus any live hold this actor still has (a lost reply, a crash).
    local ids="" pj
    if [ -f "$STATE_DIR/hold_lease" ]; then ids="$(cat "$STATE_DIR/hold_lease")"; fi
    pj="$(mktemp)"
    if pool_json > "$pj" 2>/dev/null; then
        ids="$ids $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(l['lease_id'] for l in d.get('leases') or [] if l.get('holder')=='operator:'+sys.argv[2] and l.get('status') in ('granted','recalling','queued')))" "$pj" "$ACTOR")"
    fi
    rm -f "$pj"
    local id released=1
    for id in $(echo "$ids" | tr ' ' '\n' | sort -u); do
        say "releasing diffusion hold $id"
        if ctl release --lease-id "$id"; then :; else say "WARNING: release of $id failed"; released=0; fi
    done
    if [ "$released" = "1" ]; then rm -f "$STATE_DIR/hold_lease"; else manual_steps; fi
    # Resume only a pause the POOL says is this bake-off's. The marker file decides only when the
    # pool cannot be read (and in a dry run, where the pause is fake).
    local owner
    if [ "$DRY_RUN" = "1" ]; then
        if [ -f "$STATE_DIR/paused" ]; then owner=mine; else owner=running; fi
    else
        owner="$(pool_paused_by)"
        if [ "$owner" = "unknown" ]; then
            if [ -f "$STATE_DIR/paused" ]; then owner=mine; else owner=running; fi
            say "pool state unreadable: deciding from the marker ($owner)"
        fi
    fi
    case "$owner" in
        mine)
            say "resuming pool actuation"
            ctl resume --timeout-sec 60 || say "resume reply was not ok; checking pool state"
            if [ "$DRY_RUN" = "1" ] || [ "$(pool_paused_by)" != "mine" ]; then
                rm -f "$STATE_DIR/paused"
            else
                say "WARNING: the pool still says paused by $ACTOR: run '$0 cleanup' again, or Resume in the Hub GPU pool panel"
            fi
            ;;
        other) say "pool actuation is paused by someone else: leaving their pause alone"; rm -f "$STATE_DIR/paused" ;;
        *) say "pool actuation is not paused by this bake-off: nothing to resume"; rm -f "$STATE_DIR/paused" ;;
    esac
    say "pool state after cleanup:"
    print_pool
    rm -f "$CTL_REPLY"
    if [ -n "$RUN_DIR" ] && [ "$DRY_RUN" != "1" ] && ls "$RUN_DIR"/*/bench.json "$RUN_DIR"/*/canary.json >/dev/null 2>&1; then
        python3 "$CLIENT" summarize --results-dir "$RUN_DIR" || true
        say "results: $RUN_DIR (summary.json, fieldnote_draft.md, run.log)"
    fi
}

# After the hold: if the pool began loading agent-gpu2 meanwhile, our (owner-priority) diffusion
# hold would be draining it. Abort before pausing (cleanup releases the hold).
post_hold_check() {
    local bad
    bad="$(pool_json | json_get "[str(c) for c in d.get('cards') or [] if c.get('card')=='gpu2' and (c.get('swap_state')!='idle' or 'agent-gpu2' in (c.get('swapped_in') or []) or (c.get('actuation') and not c['actuation'].get('finished_at')))]")"
    [ "$bad" = "[]" ] || die "gpu2 changed between preflight and the hold: $bad"
}

recheck_card() {
    local used
    used="$(gpu2_used_mib)"
    if [ "$DRY_RUN" != "1" ] && { [ -z "$used" ] || [ "$used" -gt "$MAX_GPU2_MIB" ]; }; then
        die "gpu2 shows ${used:-?} MiB after pausing: something loaded meanwhile"
    fi
    say "recheck OK: gpu2 ${used:-?} MiB"
}

cmd_run() {
    local yes=0 with_q4=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --yes) yes=1 ;;
            --with-q4) with_q4=1 ;;
            *) echo "unknown flag $1" >&2; exit 2 ;;
        esac
        shift
    done
    take_lock
    say "stage 7.1 bake-off preflight (read-only)"
    preflight || { say "not starting: fix the above, or wait for the seat to go idle"; exit 3; }
    if [ "$yes" != "1" ]; then
        say "preflight OK. Nothing has been changed. Re-run with --yes to hold diffusion, pause the pool and start (~60-80 min)."
        exit 0
    fi
    RUN_DIR="$RESULTS_ROOT/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$RUN_DIR"
    say "results -> $RUN_DIR"
    [ "$DRY_RUN" = "1" ] && say "DRY_RUN=1: mutating steps are printed, not run"
    ctl check || die "pool control tool cannot run in $CTL_IMAGE (imports or bus): nothing changed"

    trap cleanup EXIT
    trap 'say "interrupted"; exit 130' INT
    trap 'say "terminated"; exit 143' TERM
    trap 'say "hangup"; exit 129' HUP

    local reply lease status reason
    ctl hold || die "diffusion hold refused: $(cat "$CTL_REPLY")"
    reply="$(cat "$CTL_REPLY")"
    lease="$(echo "$reply" | json_get "d['detail'].get('lease_id')")"
    status="$(echo "$reply" | json_get "d['detail'].get('status')")"
    [ -n "$lease" ] && [ "$lease" != "None" ] || die "diffusion hold returned no lease id: $reply"
    echo "$lease" > "$STATE_DIR/hold_lease"
    say "diffusion hold $lease: $status"
    local i
    for i in $(seq 1 36); do
        [ "$status" = "granted" ] && break
        sleep 5
        status="$(pool_json | json_get "next((l['status'] for l in d.get('leases') or [] if l.get('lease_id')=='$lease'), 'missing')")"
    done
    [ "$status" = "granted" ] || die "diffusion hold not granted after 3 min (status $status): diffusion is busy"
    fail_point after_hold
    post_hold_check

    # The marker goes down BEFORE the pause is sent, so a lost reply is still resumed. Cleanup asks
    # the pool who paused it; the marker is only the fallback when the pool is unreadable.
    touch "$STATE_DIR/paused"
    ctl pause || die "pause failed: $(cat "$CTL_REPLY")"
    reason="$(json_get "d.get('reason')" < "$CTL_REPLY")"
    if [ "$reason" != "paused" ]; then
        rm -f "$STATE_DIR/paused"
        die "pool answered '$reason' to pause (someone else paused it after preflight): not touching their stop"
    fi
    say "pool actuation paused"
    fail_point after_pause

    recheck_card
    pass_run bonsai "$BONSAI_IMAGE" "$BONSAI_PROFILE" 262144 14000,32000,61000,100000 "$BONSAI_CANARY_MIN"
    if [ "$with_q4" = "1" ]; then
        # The Q4 27B cannot hold 2 x 131K on a 32 GB card (spec): 2 x 65K, depths that fit a 65K slot.
        pass_run q4 "$Q4_IMAGE" "$Q4_PROFILE" 131072 14000,32000,61000 "$Q4_CANARY_MIN"
    fi
    say "bake-off finished"
}

cmd_cleanup() {
    take_lock
    cleanup
}

case "${1:-}" in
    preflight) shift; preflight; rc=$?; rm -f "$CTL_REPLY"; exit $rc ;;
    run) shift; cmd_run "$@" ;;
    cleanup) shift; cmd_cleanup ;;
    *) usage; rm -f "$CTL_REPLY"; exit 2 ;;
esac
