#!/bin/sh
# safe_docker_build.sh — wrapper around `docker compose` that refuses to run
# from the shared/primary git checkout of this repo.
#
# Rationale: a real incident occurred where a concurrent AI coding-agent
# session ran `docker compose build`+`up` directly from the shared/primary
# checkout (not a worktree) against services/orion-cortex-orch, and silently
# reverted another session's already-verified fix. A git hook cannot prevent
# this category of incident — `docker compose` is not a git operation, so
# git hooks have no jurisdiction over it. This wrapper is the weaker,
# convention-based mitigation: it is only effective if callers route their
# docker compose invocations through it instead of calling `docker compose`
# directly.
#
# Usage:
#   scripts/safe_docker_build.sh <service-name> [docker compose args...]
#
# Example:
#   scripts/safe_docker_build.sh orion-cortex-orch build
#   scripts/safe_docker_build.sh orion-cortex-orch up -d --build
#
# Escape hatch (use deliberately, not by default):
#   ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh ...

set -e

# --- 1. Detect whether this is the shared/primary checkout -----------------
# A linked worktree has its own private git-dir (typically under
# <common-dir>/worktrees/<name>) that differs from the common git-dir shared
# by all worktrees. The primary/shared checkout's git-dir IS the common-dir.
#
# NOTE: this detection block is intentionally byte-for-byte duplicated in
# scripts/git_hooks/pre-commit rather than sourced from a shared file --
# that file gets copied verbatim by git into .git/hooks/ and must stay
# self-contained. If you fix a bug here, fix it there too. A THIRD, Python
# reimplementation of this same detection lives in
# scripts/hooks/destructive_git_guard.py's _is_shared_checkout(), and a
# FOURTH lives in scripts/git_hooks/orion-git-shim -- if you fix a bug in
# the detection logic itself, check whether it applies to all three of the
# others too.
_GITDIR=$(cd "$(git rev-parse --git-dir 2>/dev/null)" 2>/dev/null && pwd)
_COMMONDIR=$(cd "$(git rev-parse --git-common-dir 2>/dev/null)" 2>/dev/null && pwd)
IS_SHARED_CHECKOUT=0
if [ -n "$_COMMONDIR" ] && [ "$_GITDIR" = "$_COMMONDIR" ]; then
    IS_SHARED_CHECKOUT=1
fi

if [ "$IS_SHARED_CHECKOUT" = "1" ] && [ "$ORION_ALLOW_SHARED_CHECKOUT_WRITE" != "1" ]; then
    echo "safe_docker_build.sh: REFUSING to run docker compose from the shared/primary git checkout." >&2
    echo "" >&2
    echo "This repo has had a real incident where a concurrent agent session ran" >&2
    echo "'docker compose build'/'up' directly from the shared checkout and silently" >&2
    echo "reverted another session's already-verified fix. Docker compose is not a" >&2
    echo "git operation, so a git hook cannot block it — this wrapper is the only" >&2
    echo "available mitigation, and it only works if you route through it." >&2
    echo "" >&2
    echo "Fix: run this from a git worktree instead of the shared checkout, e.g.:" >&2
    echo "  git worktree add ../Orion-Sapienform-<task-name> -b <type>/<task-name>" >&2
    echo "  cd ../Orion-Sapienform-<task-name>" >&2
    echo "  scripts/safe_docker_build.sh <service-name> ..." >&2
    echo "" >&2
    echo "Escape hatch (only if you are certain this is safe right now):" >&2
    echo "  ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh <service-name> ..." >&2
    exit 1
fi

# --- 2. Validate service argument and its docker-compose.yml ---------------
SERVICE="$1"
if [ -z "$SERVICE" ]; then
    echo "safe_docker_build.sh: missing required <service-name> argument." >&2
    echo "Usage: scripts/safe_docker_build.sh <service-name> [docker compose args...]" >&2
    exit 1
fi
shift

if [ ! -f "services/$SERVICE/docker-compose.yml" ]; then
    echo "safe_docker_build.sh: no such service compose file: services/$SERVICE/docker-compose.yml" >&2
    echo "Expected path does not exist. Check the service name and that you are" >&2
    echo "running this from the repo root." >&2
    exit 1
fi

# --- 2.5. Best-effort collision-visibility heartbeat ------------------------
# Advisory only, never blocks the build: this repo has had a real incident
# of concurrent agents stepping on each other's docker deploys (see header),
# and separately, live sessions have reported real collisions from multiple
# concurrent agents working similar topics. Announce the action on the
# shared host-local agent board (scripts/agent_board.py) before running it,
# so a concurrent session's `agent_board.py checkin` sees it. Every call
# through this wrapper heartbeats, not just build/up -- config/logs/ps calls
# are cheap to announce too and the wrapper has no reliable way to tell
# which subcommands are read-only vs mutating across arbitrary compose args.
#
# Resolve agent_board.py relative to THIS script's own location, not the
# caller's cwd/git-toplevel -- this wrapper can be invoked via a relative
# path or from a subdirectory, and cwd is not guaranteed to be the repo root
# even in normal use. `$0`'s directory is always this repo's scripts/, where
# agent_board.py lives as a sibling file.
_SCRIPT_DIR_FOR_BOARD=$(cd "$(dirname "$0")" && pwd)
if [ -f "$_SCRIPT_DIR_FOR_BOARD/agent_board.py" ] && command -v python3 >/dev/null 2>&1; then
    if [ -n "${CLAUDE_CODE_SESSION_ID:-}" ]; then
        python3 "$_SCRIPT_DIR_FOR_BOARD/agent_board.py" heartbeat \
            --summary "docker compose $* for services/$SERVICE" \
            --task "deploy:$SERVICE" --session-id "$CLAUDE_CODE_SESSION_ID" >/dev/null 2>&1 || true
    else
        python3 "$_SCRIPT_DIR_FOR_BOARD/agent_board.py" heartbeat \
            --summary "docker compose $* for services/$SERVICE" \
            --task "deploy:$SERVICE" >/dev/null 2>&1 || true
    fi
fi

# --- 3. Run docker compose with this repo's mandatory dual --env-file  -----
# AGENTS.md section 8 requires every docker compose invocation in this repo
# to load BOTH the root .env and the service's own .env, root first --
# docker compose only auto-loads a .env from its own working directory, and
# many services reference root-only vars (e.g. ${ORION_BUS_URL}) with no
# per-service duplicate, so cd'ing into the service dir and running plain
# `docker compose` (the original shape of this script) silently drops those
# vars. Stay at repo root and pass both --env-file flags plus -f explicitly,
# matching AGENTS.md's own documented pattern exactly. exec replaces this
# shell process so the wrapper's exit code is exactly docker compose's.
exec docker compose \
    --env-file .env \
    --env-file "services/$SERVICE/.env" \
    -f "services/$SERVICE/docker-compose.yml" \
    "$@"
