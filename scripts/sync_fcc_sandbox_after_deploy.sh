#!/usr/bin/env bash
# Refresh Orion's FCC sandbox (/mnt/orion-fcc/repo) to origin/main after a deploy.
#
# Why: Orion's chat turns put the sandbox first on PYTHONPATH
# (orion/fcc/context_budget.py extend_fcc_subprocess_env), so `python -P -m
# orion.dig ...` imports the sandbox's orion/, not the deployed image's. The
# sandbox only refreshed when a Hub browser session connected, so a deploy
# followed by more chat in an open tab left Orion on old code (2026-11-04: the
# sandbox sat at #3107 while #3109 and #3112 were deployed).
#
# How: runs the existing sync (orion/fcc/sandbox_sync.py: rescue dirty work,
# then reset to origin/main, under the cross-container turn lock) inside the
# running Hub container -- the same code and the same root user the
# connect-time sync uses. The host user cannot do it: most sandbox files are
# root-owned (written by turns in the containers).
#
# Called by scripts/rebuild_services_from_git_diff.sh at the end of a
# successful run. Prints ONE line on stdout and always exits 0:
#   ok result=synced head=<sha> ...
#   skipped reason=<why>
#   failed reason=<why>
# Diagnostics go to stderr.
#
# Usage: scripts/sync_fcc_sandbox_after_deploy.sh [--dry-run] [--wait-s N] [--log-to FILE]
#   --wait-s N     how long to wait for in-flight turns to release the sandbox
#                  lock (default 120). The sync never touches the tree mid-turn.
#   --log-to FILE  also append "<ts> pid=.. head=.. SANDBOX_SYNC retry <verdict>"
#                  to FILE (the rebuild log). Used by the detached retry the
#                  rebuild script starts when a long turn outlasted the wait.

set -u

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
DRY_RUN=0
WAIT_S=120
LOG_TO=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --wait-s) shift; WAIT_S="${1:-120}" ;;
        --log-to) shift; LOG_TO="${1:-}" ;;
        *) echo "failed reason=unknown_argument:$1"; exit 0 ;;
    esac
    shift
done
case "$WAIT_S" in
    ""|*[!0-9]*) WAIT_S=120 ;;
esac

# Every verdict leaves through here: stdout, plus the rebuild log for a retry.
verdict() {
    echo "$*"
    if [ -n "$LOG_TO" ]; then
        (
            mkdir -p "$(dirname "$LOG_TO")" &&
                printf '%s pid=%s head=%s SANDBOX_SYNC retry %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
                    "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)" "$*" >>"$LOG_TO"
        ) 2>/dev/null || true
    fi
    exit 0
}

# Same rule as a live deploy (safe_docker_build.sh, apply_post_pull_migrations.py):
# only the clean primary main checkout, synchronized with origin/main.
git_dir=$(git -C "$REPO_ROOT" rev-parse --absolute-git-dir 2>/dev/null || true)
common=$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null || true)
case "$common" in
    ""|/*) ;;
    *) common="$REPO_ROOT/$common" ;;
esac
if [ -z "$git_dir" ] || [ -z "$common" ] ||
    [ "$(cd "$git_dir" && pwd -P)" != "$(cd "$common" 2>/dev/null && pwd -P)" ]; then
    verdict "skipped reason=linked_worktree"
fi
if [ "$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null)" != "main" ]; then
    verdict "skipped reason=not_on_main"
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
    verdict "skipped reason=dirty_checkout"
fi
deploy_head=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)
origin_head=$(git -C "$REPO_ROOT" rev-parse --verify origin/main 2>/dev/null || true)
if [ -z "$deploy_head" ] || [ "$deploy_head" != "$origin_head" ]; then
    verdict "skipped reason=main_not_origin_main"
fi

if ! command -v docker >/dev/null 2>&1; then
    verdict "skipped reason=no_docker"
fi
# The hub stack by compose labels, not by container name: the name carries
# ${PROJECT}, which differs per host. No running hub = no sandbox here (circe).
hub=$(docker ps --filter label=com.docker.compose.project=orion-hub \
    --filter label=com.docker.compose.service=hub-app --format '{{.Names}}' 2>/dev/null | head -n 1)
if [ -z "$hub" ]; then
    verdict "skipped reason=hub_not_running"
fi

if [ "$DRY_RUN" = "1" ]; then
    verdict "skipped reason=dry_run would_run=docker_exec_${hub}_python3_-m_orion.fcc.sandbox_sync"
fi

# The sync's git steps time out at 30 s each; bound the whole call too.
TIMEOUT=""
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT="timeout $((${WAIT_S%.*} + 300))"
fi
# shellcheck disable=SC2086
out=$($TIMEOUT docker exec "$hub" python3 -m orion.fcc.sandbox_sync --wait-s "$WAIT_S" </dev/null)
rc=$?
result=$(printf '%s\n' "$out" | awk 'NF' | tail -n 1)
if [ "$rc" -eq 0 ] && [ -n "$result" ]; then
    verdict "ok $result deploy_head=${deploy_head:0:12}"
elif [ -n "$result" ]; then
    verdict "failed reason=sync_declined rc=$rc $result"
fi
# An image built before the CLI existed runs the module and prints nothing.
verdict "failed reason=no_verdict rc=$rc container=$hub"
