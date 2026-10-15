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
    exit "$PY_RC"
fi
SERVICES=$(printf '%s\n' "$PLAN" | awk -F'\t' 'NF {print $1}')
RECREATE=$(printf '%s\n' "$PLAN" | awk -F'\t' '$2 == "recreate" {print $1}')

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

if [ -z "${SERVICES//[$'\n\r\t ']}" ]; then
    echo "rebuild_services_from_git_diff.sh: no affected services to rebuild on this host"
    echo "rebuild_services_from_git_diff.sh: (see rebuild_affected_services stderr above for host filter details)"
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
    set +e
    if [ -n "$ALLOW_SHARED" ]; then
        ORION_ALLOW_SHARED_CHECKOUT_WRITE="$ALLOW_SHARED" "$SAFE_BUILD" "$@"
    else
        "$SAFE_BUILD" "$@"
    fi
    rc=$?
    set -e
}

for svc in $SERVICES; do
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

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] would rebuild: $svc${RECREATE_FLAG:+ ($RECREATE_FLAG)}"
        if [ -n "$PROFILE_SERVICES" ]; then
            printf '%s\n' "$PROFILE_SERVICES" | while IFS="$(printf '\t')" read -r profile name; do
                [ -n "$name" ] || continue
                echo "[dry-run] would rebuild running profile service: $svc --profile $profile $name"
            done
        fi
        REBUILT=$((REBUILT + 1))
        continue
    fi

    echo ""
    echo "=== rebuild_services_from_git_diff: $svc ==="

    if [ "$svc" = "$CORTEX_EXEC" ]; then
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

    if [ "$svc_failed" -ne 0 ]; then
        FAILED="$FAILED $svc"
    else
        REBUILT=$((REBUILT + 1))
    fi
done

echo ""
echo "rebuild_services_from_git_diff.sh: rebuilt $REBUILT service(s)"

if [ -n "${FAILED//[$'\n\r\t ']}" ]; then
    echo "rebuild_services_from_git_diff.sh: failures:$FAILED" >&2
    exit 1
fi

exit 0
