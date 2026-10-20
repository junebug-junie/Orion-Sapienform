#!/usr/bin/env bash
# Rebuild only docker-compose services affected by recent git changes.
#
# Intended for post-merge (git pull) opt-in automation and manual operator use.
# Does NOT run the full mesh-utilities/common/up_all_services_batched.sh sweep.
#
# Usage:
#   scripts/rebuild_services_from_git_diff.sh [--dry-run] [--base REF] [--list-only]
#
# Requires:
#   - python3
#   - scripts/safe_docker_build.sh (honours worktree/shared-checkout policy)
#   - docker compose + per-service .env files for services being rebuilt
#
# Opt-in post-merge hook: set ORION_POST_MERGE_REBUILD=1 or create
#   .orion/post-merge-rebuild-enabled
# at repo root (see scripts/git_hooks/post-merge).
# Every compose step is logged (start/end/exit code) to $ORION_REBUILD_LOG,
# default ~/.orion/post_pull_rebuild.log.

set -e

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

DRY_RUN=0
LIST_ONLY=0
BASE_REF=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --list-only)
            LIST_ONLY=1
            ;;
        --base)
            shift
            BASE_REF="${1:-}"
            if [ -z "$BASE_REF" ]; then
                echo "rebuild_services_from_git_diff.sh: --base requires a ref" >&2
                exit 2
            fi
            ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "rebuild_services_from_git_diff.sh: unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

if ! command -v python3 >/dev/null 2>&1; then
    echo "rebuild_services_from_git_diff.sh: python3 not found" >&2
    exit 2
fi

PY="$REPO_ROOT/scripts/rebuild_affected_services.py"
if [ ! -f "$PY" ]; then
    echo "rebuild_services_from_git_diff.sh: missing $PY" >&2
    exit 2
fi

# Run log: one line per compose step (start, end, exit code), appended to
# $ORION_REBUILD_LOG (default ~/.orion/post_pull_rebuild.log). The post-merge
# hook's output only reaches the terminal that ran `git pull`, and docker keeps
# about a minute of events here, so without this file nobody can tell later
# whether a pull's rebuild recreated a container, failed, or was still running.
# 2026-10-16: orion-vision-retina-walkway looked "not recreated" at 18:56; the
# hook's step was in fact mid-recreate (old container inside its 30 s
# stop_grace_period), and a manual --force-recreate raced it. Best effort:
# a log write never fails the rebuild.
REBUILD_LOG="${ORION_REBUILD_LOG:-${HOME:-/tmp}/.orion/post_pull_rebuild.log}"
REBUILD_HEAD=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
rebuild_log() {
    (
        mkdir -p "$(dirname "$REBUILD_LOG")" &&
            printf '%s pid=%s head=%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$REBUILD_HEAD" "$*" >>"$REBUILD_LOG"
    ) 2>/dev/null || true
}

if [ -n "$BASE_REF" ]; then
    set -- --base "$BASE_REF"
else
    set --
fi

# --plan prints "<service>\t<build|recreate>". recreate = a changed repo path
# is bind-mounted into the service's containers, so an unchanged image would
# not recreate them (and a single-file mount keeps the old inode) -- those
# get --force-recreate below.
set +e
PLAN=$(python3 "$PY" --plan "$@")
PY_RC=$?
set -e
if [ "$PY_RC" -ne 0 ]; then
    if [ "$DRY_RUN" != "1" ] && [ "$LIST_ONLY" != "1" ]; then
        rebuild_log "PLANNER_FAILED rc=$PY_RC base=${BASE_REF:-default}"
    fi
    exit "$PY_RC"
fi
SERVICES=$(printf '%s\n' "$PLAN" | awk -F'\t' 'NF {print $1}')
RECREATE=$(printf '%s\n' "$PLAN" | awk -F'\t' '$2 == "recreate" {print $1}')

# Hand-written SQL migrations changed in this pull. They never rebuild
# anything (2026-10-15: a migration-only pull recreated orion-athena-sql-db,
# an unclean Postgres restart under the whole fleet). Their consumers stay
# refused by safe_docker_build.sh's migration gate until applied, so BEFORE any
# rebuild, scripts/apply_post_pull_migrations.py applies the ones that are
# NOT APPLIED and safe (every statement on an additive-DDL allowlist, valid
# REQUIRED-BY header, not rollback/DESTRUCTIVE), in filename order with
# lock_timeout=10s, only from the clean primary main checkout synced with
# origin/main; the first file that fails or needs a human holds the rest. Kill switch:
# ORION_POST_PULL_APPLY_MIGRATIONS=0. Then the read-only report shows the live
# verdict and the manual command for anything still unapplied. Only on a host
# whose service list includes orion-sql-db (the classifier lists none on circe).
#
# A running backup (2026-10-17: nightly pg_dump held locks ~45 min, the apply
# hit lock_timeout, hub and sql-writer stayed refused until applied by hand):
# this first pass runs with --defer-if-backup, so it never waits; the applier
# exits 3 when it deferred files behind a backup. Every rebuild then runs as
# usual (consumers of the deferred files are refused by the deploy gate, the
# rest deploy), and only after that does apply_deferred_migrations wait for
# the backup, apply, and re-run the rebuild of the refused consumers.
APPLY_RC=0
APPLIER=""
CHECK_PY=python3
MIGRATIONS=""
if [ "$LIST_ONLY" != "1" ]; then
    set +e
    MIGRATIONS=$(python3 "$PY" --migrations "$@" 2>/dev/null)
    set -e
    if [ -n "${MIGRATIONS//[$'\n\r\t ']}" ]; then
        echo "rebuild_services_from_git_diff.sh: hand-written SQL migration(s) changed -- no rebuild runs these; safe ones are applied first, before any rebuild:"
        CHECKER="$REPO_ROOT/scripts/check_sql_migrations_applied.py"
        # Same interpreter lookup as safe_docker_build.sh: the repo venv has
        # psycopg2; a linked worktree reaches the main checkout's venv.
        CHECK_PY=python3
        COMMON_DIR=$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null || true)
        case "$COMMON_DIR" in
            ""|/*) ;;
            *) COMMON_DIR="$REPO_ROOT/$COMMON_DIR" ;;
        esac
        if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
            CHECK_PY="$REPO_ROOT/.venv/bin/python"
        elif [ -n "$COMMON_DIR" ] && [ -x "$COMMON_DIR/../.venv/bin/python" ]; then
            CHECK_PY="$COMMON_DIR/../.venv/bin/python"
        fi
        RAW_APPLY="docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 <"
        APPLIER="$REPO_ROOT/scripts/apply_post_pull_migrations.py"
        if [ -f "$APPLIER" ]; then
            APPLY_ARGS="--defer-if-backup"
            if [ "$DRY_RUN" = "1" ]; then
                APPLY_ARGS="--dry-run"
            fi
            # Never fails the pull: the deploy gate still refuses any service
            # whose migration did not land.
            set +e
            # shellcheck disable=SC2086
            (cd "$REPO_ROOT" && "$CHECK_PY" "$APPLIER" $APPLY_ARGS $MIGRATIONS </dev/null 2>&1)
            APPLY_RC=$?
            set -e
        else
            APPLIER=""
        fi
        REPORT_RC=1
        if [ -f "$CHECKER" ]; then
            # One read-only run for every file: required-by, live state, apply
            # command (rollback/DESTRUCTIVE files get none).
            set +e
            # shellcheck disable=SC2086
            REPORT=$(cd "$REPO_ROOT" && "$CHECK_PY" "$CHECKER" --post-pull-report $MIGRATIONS </dev/null 2>&1)
            REPORT_RC=$?
            set -e
            printf '%s\n' "$REPORT"
        fi
        if [ "$REPORT_RC" != "0" ]; then
            echo "  live state: UNKNOWN (scripts/check_sql_migrations_applied.py missing or failed; no answer is not 'applied')"
            for mig_path in $MIGRATIONS; do
                echo "  - $mig_path"
                case "$mig_path" in
                    *_rollback.sql)
                        echo "      rollback file: never applied on deploy"
                        continue
                        ;;
                esac
                echo "      apply only after reading its header (never a *_rollback.sql or -- DESTRUCTIVE file): $RAW_APPLY $mig_path"
            done
        fi
    fi
fi

# cortex-exec and cortex-orch are coupled at runtime, not just by the
# import-graph classifier above: cortex-exec's self_study.py round-trips
# CortexClientRequest calls through cortex-orch (verb resolution + verb
# activation live there), so a cortex-exec change can require an orch
# rebuild even when the affected-services classifier -- which walks .py
# import statements -- doesn't happen to attribute the changed path to
# orch (verb YAML/prompt-template changes under orion/cognition/ are the
# concrete case: they carry no Python import statement of their own to
# scan). Confirmed live 2026-09-04: a cortex-exec-only rebuild shipped a
# new verb, orch's stale image never learned about it, and the verb
# resolved as inactive on the live bus until orch was rebuilt by hand.
# So: whenever cortex-exec is rebuilt, always rebuild cortex-orch too,
# regardless of what the classifier attributed to orch on its own.
CORTEX_EXEC_SVC="orion-cortex-exec"
CORTEX_ORCH_SVC="orion-cortex-orch"
if printf '%s\n' $SERVICES | grep -qx "$CORTEX_EXEC_SVC" 2>/dev/null; then
    if ! printf '%s\n' $SERVICES | grep -qx "$CORTEX_ORCH_SVC" 2>/dev/null; then
        echo "rebuild_services_from_git_diff.sh: $CORTEX_EXEC_SVC is affected -- adding $CORTEX_ORCH_SVC (coupled at runtime, see script comment)"
        SERVICES="$SERVICES
$CORTEX_ORCH_SVC"
    fi
fi

# Second pass for files the first pass deferred behind a running backup: wait
# for it (bounded: ORION_MIGRATION_BACKUP_MAX_WAIT_S, default 90 min), then
# apply. Sets DEFERRED_RAN=1 when the pass got to the files (exit 0, or 1 =
# some file still refused/failed); the caller then retries the refused
# consumers, and the deploy gate re-refuses any whose file did not land.
# Runs after the rebuilds, so nothing else waits -- but the post-merge hook is
# synchronous, so `git pull` itself does not return until this is done.
DEFERRED_RAN=0
apply_deferred_migrations() {
    DEFERRED_RAN=0
    [ "$APPLY_RC" = "3" ] && [ -n "$APPLIER" ] && [ "$DRY_RUN" != "1" ] || return 0
    echo ""
    echo "rebuild_services_from_git_diff.sh: migration(s) were deferred behind a running backup; waiting for it, then applying (Ctrl-C is safe: lock_timeout keeps every apply short; rerun this script or apply by hand later)"
    echo "rebuild_services_from_git_diff.sh: git pull will not return until the backup finishes or ${ORION_MIGRATION_BACKUP_MAX_WAIT_S:-5400}s pass (ORION_MIGRATION_BACKUP_MAX_WAIT_S)" >&2
    rebuild_log "DEFERRED-APPLY start"
    set +e
    # shellcheck disable=SC2086
    (cd "$REPO_ROOT" && "$CHECK_PY" "$APPLIER" $MIGRATIONS </dev/null 2>&1)
    local rc=$?
    set -e
    rebuild_log "DEFERRED-APPLY end rc=$rc"
    if [ "$rc" = "0" ] || [ "$rc" = "1" ]; then
        DEFERRED_RAN=1
    fi
    if [ "$rc" != "0" ]; then
        echo "rebuild_services_from_git_diff.sh: deferred migration apply did not finish (exit $rc); consumers of any file still unapplied stay refused by the deploy gate until it is applied" >&2
    fi
}

if [ -z "${SERVICES//[$'\n\r\t ']}" ]; then
    # A failed second pass is reported on stderr but, like the first pass, never
    # fails the pull: there is nothing to rebuild, and the deploy gate holds.
    apply_deferred_migrations
    echo "rebuild_services_from_git_diff.sh: no affected services to rebuild on this host"
    echo "rebuild_services_from_git_diff.sh: (see rebuild_affected_services stderr above for host filter details)"
    if [ "$DRY_RUN" != "1" ] && [ "$LIST_ONLY" != "1" ]; then
        rebuild_log "NOTHING base=${BASE_REF:-default} (no affected services on this host)"
    fi
    exit 0
fi

echo "rebuild_services_from_git_diff.sh: affected services:"
printf '  - %s\n' $SERVICES

if [ "$LIST_ONLY" = "1" ]; then
    printf '%s\n' "$SERVICES"
    exit 0
fi

SAFE_BUILD="$REPO_ROOT/scripts/safe_docker_build.sh"
if [ ! -x "$SAFE_BUILD" ]; then
    echo "rebuild_services_from_git_diff.sh: missing executable $SAFE_BUILD" >&2
    exit 2
fi

# Post-merge on the primary checkout is an explicit operator opt-in — allow
# safe_docker_build's escape hatch in that case only.
ALLOW_SHARED=""
if [ "$ORION_POST_MERGE_REBUILD" = "1" ] || [ -f "$REPO_ROOT/.orion/post-merge-rebuild-enabled" ]; then
    ALLOW_SHARED="${ORION_ALLOW_SHARED_CHECKOUT_WRITE:-1}"
fi

CORTEX_EXEC="orion-cortex-exec"
CORTEX_LANES="cortex-exec cortex-exec-chat cortex-exec-spark cortex-exec-background"
ANALYTICS="orion-analytics"

FAILED=""
REBUILT=0

# run_safe_build <service-dir> <compose args...>; sets rc.
run_safe_build() {
    local started
    started=$(date +%s) || started=0
    rebuild_log "START $*"
    set +e
    if [ -n "$ALLOW_SHARED" ]; then
        ORION_ALLOW_SHARED_CHECKOUT_WRITE="$ALLOW_SHARED" "$SAFE_BUILD" "$@"
    else
        "$SAFE_BUILD" "$@"
    fi
    rc=$?
    set -e
    rebuild_log "END rc=$rc seconds=$(($(date +%s) - started)) $*"
}

if [ "$DRY_RUN" != "1" ]; then
    rebuild_log "RUN base=${BASE_REF:-default} services=$(printf '%s ' $SERVICES)recreate=$(printf '%s ' $RECREATE)"
fi

# rebuild_one <service>: the full rebuild of one stack; adds it to FAILED or
# counts it in REBUILT.
rebuild_one() {
    local svc="$1"
    RECREATE_FLAG=""
    if printf '%s\n' $RECREATE | grep -qx "$svc" 2>/dev/null; then
        RECREATE_FLAG="--force-recreate"
    fi

    # Profile-gated compose services of this stack that are RUNNING on this
    # host right now. A plain `up` never selects them, so before this they
    # kept their old image forever (orion-vision-retina-walkway). Never
    # starts a profile service that is not already running. The analytics
    # stack is brought up with its whole profile below, so skip it here.
    SKIP_PROFILE_ARGS=""
    if [ "$svc" = "$ANALYTICS" ]; then
        SKIP_PROFILE_ARGS="--skip-profile analytics"
    fi
    # shellcheck disable=SC2086
    PROFILE_SERVICES=$(python3 "$PY" --running-profile-services "$svc" $SKIP_PROFILE_ARGS || true)
    # Second compose files of this stack (docker-compose.<name>.yml) with a
    # container running from them now. A plain `up` only ever read
    # docker-compose.yml, so the Qwen caption lane (docker-compose.circe-qwen.yml)
    # kept its 2026-10-16 image and env through every pull (2026-10-20). Never
    # starts a lane that is not running; GPU-pool launch files are skipped.
    EXTRA_COMPOSE=$(python3 "$PY" --running-extra-compose "$svc" || true)

    # GPU-pool launch targets of this stack (config/gpu_pool.yaml `launch:`).
    # A STOPPED one was evicted by the pool on purpose: build its image, never
    # start it. Confirmed live 2026-10-16 04:02 UTC: a plain `up -d --build`
    # restarted diffusion-host under agent-gpu2 on gpu2, the lane controller
    # then refused every load/unload (seat_and_evicted_both_running) and no
    # picture rendered for 15 hours. The helper exits non-zero when it cannot
    # tell what the pool owns: that stack is skipped (FAILED), never plain-up'd.
    POOL_RC=0
    POOL_PLAN=$(python3 "$PY" --pool-held-services "$svc") || POOL_RC=$?
    if [ "$POOL_RC" -ne 0 ]; then
        echo "rebuild_services_from_git_diff.sh: FAILED $svc -- cannot read GPU pool ownership (exit $POOL_RC); not starting anything" >&2
        FAILED="$FAILED $svc"
        return 0
    fi
    POOL_HOLD=$(printf '%s\n' "$POOL_PLAN" | awk -F '\t' '$1 == "hold" { print $2 }')

    if [ "$DRY_RUN" = "1" ]; then
        if [ -n "$POOL_PLAN" ]; then
            echo "[dry-run] would build: $svc, then up -d --no-build --no-deps${RECREATE_FLAG:+ $RECREATE_FLAG} on its non-held services"
            if [ -n "$POOL_HOLD" ]; then
                # shellcheck disable=SC2086
                echo "[dry-run] would hold stopped (GPU pool evicted it): $svc" $POOL_HOLD
            fi
        else
            echo "[dry-run] would rebuild: $svc${RECREATE_FLAG:+ ($RECREATE_FLAG)}"
        fi
        if [ -n "$PROFILE_SERVICES" ]; then
            printf '%s\n' "$PROFILE_SERVICES" | while IFS="$(printf '\t')" read -r profile name; do
                [ -n "$name" ] || continue
                echo "[dry-run] would rebuild running profile service: $svc --profile $profile $name"
            done
        fi
        if [ -n "$EXTRA_COMPOSE" ]; then
            printf '%s\n' "$EXTRA_COMPOSE" | while IFS="$(printf '\t')" read -r extra names; do
                [ -n "$extra" ] || continue
                echo "[dry-run] would rebuild running extra compose file: $svc --compose-file $extra $names"
            done
        fi
        REBUILT=$((REBUILT + 1))
        return 0
    fi

    echo ""
    echo "=== rebuild_services_from_git_diff: $svc ==="

    if [ -n "$POOL_PLAN" ]; then
        # Build first, then decide what to start from a FRESH look at docker:
        # the build takes minutes and the pool may evict a target meanwhile.
        run_safe_build "$svc" build
        if [ "$rc" -eq 0 ]; then
            POOL_RC=0
            POOL_PLAN=$(python3 "$PY" --pool-held-services "$svc") || POOL_RC=$?
            if [ "$POOL_RC" -ne 0 ]; then
                echo "rebuild_services_from_git_diff.sh: cannot re-read GPU pool ownership for $svc (exit $POOL_RC); built, not started" >&2
                rc=$POOL_RC
            else
                POOL_HOLD=$(printf '%s\n' "$POOL_PLAN" | awk -F '\t' '$1 == "hold" { print $2 }')
                POOL_UP=$(printf '%s\n' "$POOL_PLAN" | awk -F '\t' '$1 == "running" || $1 == "start" { print $2 }')
                if [ -n "$POOL_HOLD" ]; then
                    # The controller's next load starts the image built above.
                    # shellcheck disable=SC2086
                    echo "rebuild_services_from_git_diff.sh: GPU pool holds stopped:" $POOL_HOLD "-- built, not started"
                fi
                if [ -n "$POOL_UP" ]; then
                    # --no-deps: a depends_on edge must not start a held service.
                    # shellcheck disable=SC2086
                    run_safe_build "$svc" up -d --no-build --no-deps $RECREATE_FLAG $POOL_UP
                fi
            fi
        fi
    elif [ "$svc" = "$CORTEX_EXEC" ]; then
        # Match mesh-utilities/common/cortex_exec_fleet_helpers.sh lane bring-up.
        # shellcheck disable=SC2086
        run_safe_build "$svc" up -d --build $RECREATE_FLAG $CORTEX_LANES
    elif [ "$svc" = "$ANALYTICS" ]; then
        # Every long-running service in orion-analytics is intentionally behind
        # the `analytics` Compose profile (`analytics-dbt` is a separate tools
        # profile). Calling plain `up` selects nothing and exits with
        # "no service selected", which made the first post-merge rebuild of
        # the Curiosity dashboard fail despite a valid compose file.
        # shellcheck disable=SC2086
        run_safe_build "$svc" --profile analytics up -d --build $RECREATE_FLAG
    else
        # shellcheck disable=SC2086
        run_safe_build "$svc" up -d --build $RECREATE_FLAG
    fi

    svc_failed=0
    if [ "$rc" -ne 0 ]; then
        echo "rebuild_services_from_git_diff.sh: FAILED $svc (exit $rc)" >&2
        svc_failed=1
    fi

    if [ -n "$PROFILE_SERVICES" ]; then
        TAB=$(printf '\t')
        while IFS="$TAB" read -r profile name; do
            [ -n "$name" ] || continue
            echo "=== rebuild_services_from_git_diff: $svc (running profile service $name, profile $profile) ==="
            # shellcheck disable=SC2086
            # </dev/null: the loop reads its list from stdin; nothing the build
            # runs may consume the remaining lines.
            run_safe_build "$svc" --profile "$profile" up -d --build $RECREATE_FLAG "$name" </dev/null
            if [ "$rc" -ne 0 ]; then
                echo "rebuild_services_from_git_diff.sh: FAILED $svc/$name (exit $rc)" >&2
                svc_failed=1
            fi
        done <<EOF_PROFILES
$PROFILE_SERVICES
EOF_PROFILES
    fi

    if [ -n "$EXTRA_COMPOSE" ]; then
        TAB=$(printf '\t')
        while IFS="$TAB" read -r extra names; do
            [ -n "$extra" ] && [ -n "$names" ] || continue
            echo "=== rebuild_services_from_git_diff: $svc (running extra compose file $extra: $names) ==="
            # Only the services running now, --no-deps: a stopped sibling in the
            # same file (or a depends_on edge) is never started.
            # shellcheck disable=SC2086
            run_safe_build "$svc" --compose-file "$extra" up -d --build --no-deps $RECREATE_FLAG $names </dev/null
            if [ "$rc" -ne 0 ]; then
                echo "rebuild_services_from_git_diff.sh: FAILED $svc/$extra (exit $rc)" >&2
                svc_failed=1
            fi
        done <<EOF_EXTRA
$EXTRA_COMPOSE
EOF_EXTRA
    fi

    if [ "$svc_failed" -ne 0 ]; then
        FAILED="$FAILED $svc"
    else
        REBUILT=$((REBUILT + 1))
    fi
}

for svc in $SERVICES; do
    rebuild_one "$svc"
done

# Files deferred behind a backup: wait, apply, then rebuild again the
# consumers the deploy gate refused above. Only services that FAILED and are
# named in a deferred file's REQUIRED-BY header are retried, once.
apply_deferred_migrations
if [ "$DEFERRED_RAN" = "1" ] && [ -n "${FAILED//[$'\n\r\t ']}" ]; then
    set +e
    # shellcheck disable=SC2086
    NEEDS=$(cd "$REPO_ROOT" && "$CHECK_PY" "$APPLIER" --required-by $MIGRATIONS </dev/null 2>/dev/null)
    set -e
    RETRY=""
    STILL_FAILED=""
    for svc in $FAILED; do
        if printf '%s\n' $NEEDS | grep -qx "$svc" 2>/dev/null; then
            RETRY="$RETRY $svc"
        else
            STILL_FAILED="$STILL_FAILED $svc"
        fi
    done
    if [ -n "${RETRY// }" ]; then
        echo ""
        echo "rebuild_services_from_git_diff.sh: deferred migration pass done; rebuilding the services the deploy gate refused:$RETRY"
        rebuild_log "RETRY-AFTER-MIGRATION services=${RETRY# }"
        FAILED="$STILL_FAILED"
        for svc in $RETRY; do
            rebuild_one "$svc"
        done
    fi
fi

echo ""
echo "rebuild_services_from_git_diff.sh: rebuilt $REBUILT service(s)"
if [ "$DRY_RUN" != "1" ]; then
    rebuild_log "DONE rebuilt=$REBUILT failed=${FAILED# }"
    echo "rebuild_services_from_git_diff.sh: run log: $REBUILD_LOG"
fi

if [ -n "${FAILED//[$'\n\r\t ']}" ]; then
    echo "rebuild_services_from_git_diff.sh: failures:$FAILED" >&2
    exit 1
fi

exit 0
