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

# shellcheck disable=SC2086
SERVICES=$(python3 "$PY" --list-only "$@")
PY_RC=$?
if [ "$PY_RC" -ne 0 ]; then
    exit "$PY_RC"
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

FAILED=""
REBUILT=0

for svc in $SERVICES; do
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] would rebuild: $svc"
        REBUILT=$((REBUILT + 1))
        continue
    fi

    echo ""
    echo "=== rebuild_services_from_git_diff: $svc ==="

    if [ "$svc" = "$CORTEX_EXEC" ]; then
        # Match mesh-utilities/common/cortex_exec_fleet_helpers.sh lane bring-up.
        set +e
        if [ -n "$ALLOW_SHARED" ]; then
            ORION_ALLOW_SHARED_CHECKOUT_WRITE="$ALLOW_SHARED" \
                "$SAFE_BUILD" "$svc" up -d --build $CORTEX_LANES
        else
            "$SAFE_BUILD" "$svc" up -d --build $CORTEX_LANES
        fi
        rc=$?
        set -e
    else
        set +e
        if [ -n "$ALLOW_SHARED" ]; then
            ORION_ALLOW_SHARED_CHECKOUT_WRITE="$ALLOW_SHARED" \
                "$SAFE_BUILD" "$svc" up -d --build
        else
            "$SAFE_BUILD" "$svc" up -d --build
        fi
        rc=$?
        set -e
    fi

    if [ "$rc" -ne 0 ]; then
        echo "rebuild_services_from_git_diff.sh: FAILED $svc (exit $rc)" >&2
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
