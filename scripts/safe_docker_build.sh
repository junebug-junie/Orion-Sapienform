#!/bin/sh
# safe_docker_build.sh — wrapper around `docker compose` that requires live
# deployments to run from the clean, synchronized primary `main` checkout.
#
# Builds and read-only compose commands may run from development worktrees.
# `up`, however, changes the live runtime and therefore has one canonical
# source: the primary checkout on `main`, clean and equal to `origin/main`.
# This prevents a feature-named or stale worktree from becoming an accidental
# deployment source while preserving isolated worktrees for development.
#
# Usage:
#   scripts/safe_docker_build.sh <service-name> [--compose-file docker-compose.<name>.yml] [docker compose args...]
#
# --compose-file targets a second compose file of the same service INSTEAD of
# services/<service>/docker-compose.yml (never merged with it), e.g. the Qwen
# caption lane in services/orion-vision-host/docker-compose.circe-qwen.yml.
# Every check below still applies. Only docker-compose.<name>.yml basenames.
#
# Example:
#   scripts/safe_docker_build.sh orion-cortex-orch build
#   scripts/safe_docker_build.sh orion-cortex-orch up -d --build
#
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

# --- 2. Validate service argument and its docker-compose.yml ---------------
SERVICE="$1"
if [ -z "$SERVICE" ]; then
    echo "safe_docker_build.sh: missing required <service-name> argument." >&2
    echo "Usage: scripts/safe_docker_build.sh <service-name> [--compose-file docker-compose.<name>.yml] [docker compose args...]" >&2
    exit 1
fi
shift

COMPOSE_FILE_NAME="docker-compose.yml"
if [ "$1" = "--compose-file" ]; then
    COMPOSE_FILE_NAME="$2"
    case "$COMPOSE_FILE_NAME" in
        docker-compose.yml) ;;
        docker-compose.*.yml)
            case "$COMPOSE_FILE_NAME" in
                */*|*..*)
                    echo "safe_docker_build.sh: --compose-file takes a basename, got '$COMPOSE_FILE_NAME'" >&2
                    exit 1 ;;
            esac ;;
        *)
            echo "safe_docker_build.sh: --compose-file must be docker-compose.<name>.yml, got '$COMPOSE_FILE_NAME'" >&2
            exit 1 ;;
    esac
    shift 2
fi

# --- 2.1. Require live deployments to originate from canonical main --------
IS_DEPLOY=0
for arg in "$@"; do
    if [ "$arg" = "up" ]; then
        IS_DEPLOY=1
        break
    fi
done

if [ "$IS_DEPLOY" = "1" ]; then
    if [ "$IS_SHARED_CHECKOUT" != "1" ]; then
        PRIMARY_ROOT=$(git worktree list --porcelain 2>/dev/null | awk '
            /^worktree / { sub(/^worktree /, ""); print; exit }
        ')
        echo "safe_docker_build.sh: REFUSING live deployment from a linked worktree." >&2
        echo "Deploy from the clean primary main checkout instead:" >&2
        echo "  cd \"${PRIMARY_ROOT:-<primary-checkout>}\"" >&2
        echo "  scripts/safe_docker_build.sh $SERVICE up -d --build" >&2
        exit 1
    fi

    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || true)
    if [ "$CURRENT_BRANCH" != "main" ]; then
        echo "safe_docker_build.sh: REFUSING live deployment from branch '$CURRENT_BRANCH'; main is required." >&2
        exit 1
    fi

    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        echo "safe_docker_build.sh: REFUSING live deployment from dirty main." >&2
        echo "Commit, move, or remove local changes before deploying." >&2
        exit 1
    fi

    if ! git fetch --quiet origin main; then
        echo "safe_docker_build.sh: REFUSING live deployment because origin/main could not be refreshed." >&2
        echo "Restore access to origin, then retry." >&2
        exit 1
    fi
    if ! ORIGIN_MAIN=$(git rev-parse --verify origin/main 2>/dev/null); then
        echo "safe_docker_build.sh: REFUSING live deployment because origin/main is unavailable after fetch." >&2
        exit 1
    fi
    LOCAL_HEAD=$(git rev-parse HEAD 2>/dev/null)
    if [ "$LOCAL_HEAD" != "$ORIGIN_MAIN" ]; then
        echo "safe_docker_build.sh: REFUSING live deployment because main is not synchronized with origin/main." >&2
        echo "main=$LOCAL_HEAD origin/main=$ORIGIN_MAIN" >&2
        exit 1
    fi
fi

if [ ! -f "services/$SERVICE/$COMPOSE_FILE_NAME" ]; then
    echo "safe_docker_build.sh: no such service compose file: services/$SERVICE/$COMPOSE_FILE_NAME" >&2
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
            --summary "docker compose $* for services/$SERVICE/$COMPOSE_FILE_NAME" \
            --task "deploy:$SERVICE" --session-id "$CLAUDE_CODE_SESSION_ID" >/dev/null 2>&1 || true
    else
        python3 "$_SCRIPT_DIR_FOR_BOARD/agent_board.py" heartbeat \
            --summary "docker compose $* for services/$SERVICE/$COMPOSE_FILE_NAME" \
            --task "deploy:$SERVICE" >/dev/null 2>&1 || true
    fi
fi

# --- 2b. Refuse to ship a service whose live .env is behind its contract ---
# `.env` is gitignored and machine-local, so NO CI gate can ever see it drift
# behind the committed `.env_example`. This is the only chokepoint that can --
# every bring-up already routes through here. Blocks only on structured drift
# (a key present but short entries inside its JSON value), which no other tool
# in this repo can detect and which deploys cleanly and then does nothing:
# 2026-09-05, orion-sql-writer was four channels short and self_concept_history
# sat at 0 rows through two "successful" deploys. Missing whole keys only warn,
# since sync_local_env_from_example.py already fixes those.
# Never blocks on a read-only compose arg it cannot classify -- it runs for any
# invocation, and the cost of a skipped check is a silent outage.
if [ -f "scripts/check_env_template_parity.py" ]; then
    if ! python3 scripts/check_env_template_parity.py "$SERVICE"; then
        echo "" >&2
        echo "REFUSING to deploy $SERVICE with a structured env value behind its contract." >&2
        echo "Deliberate exception: ORION_ALLOW_ENV_DRIFT=1 scripts/safe_docker_build.sh ..." >&2
        exit 1
    fi
fi

# Same blind spot, different symptom: check_service_hostname_refs.py has always
# read .env_example only, because it runs in CI and .env is gitignored. But .env
# is what the container reads, so a hand-edit to http://orion-notify:7140 passes
# CI green, deploys clean, and silently breaks notify -- ~20 PRs of recurring
# repair. Verified from inside a container 2026-09-05: `notify` and
# `orion-athena-notify` resolve, `orion-notify` resolves nowhere. Scoped to the
# service being deployed so an unrelated service's file cannot block this bring-up.
# The hatch is honoured HERE rather than inside the script: that script is also a
# CI gate, and teaching a CI gate to read an env var meant for deploys would give
# CI a bypass it must never have.
if [ -f "scripts/check_service_hostname_refs.py" ] && [ "${ORION_ALLOW_ENV_DRIFT:-}" != "1" ]; then
    if ! python3 scripts/check_service_hostname_refs.py \
            --include-live-env --service "$SERVICE"; then
        echo "" >&2
        echo "REFUSING to deploy $SERVICE: its live .env names a service by its" >&2
        echo "directory name, which resolves on no host. Use the compose service key." >&2
        echo "Deliberate exception: ORION_ALLOW_ENV_DRIFT=1 scripts/safe_docker_build.sh ..." >&2
        exit 1
    fi
fi

# --- 2c. Refuse to bring up a service whose required SQL migration is unapplied
# 2026-10-10 and 10-11: PRs #2594 and #2605 deployed orion-durable-runs before
# their services/orion-sql-db/manual_migration_*.sql had been applied; the
# service started, then failed every step with UndefinedTable until the
# migration was applied by hand. A migration declares who needs it with a
# header line `-- ORION-MIGRATION-REQUIRED-BY: <svc>, <svc>`; this asks the live
# database (read-only) whether each one the service needs is applied.
# Deploy-time, not a container boot guard: a service that refuses to start is
# worse than one that degrades. Only `up` is gated (build/config/logs/ps pass).
# Never auto-applies (a migration can be destructive); the post-pull flow
# applies the safe ones before its rebuilds (scripts/apply_post_pull_migrations.py).
# An unreachable database
# is UNKNOWN and refuses -- no answer is not "applied".
_GATE_UP=0
for _a in "$@"; do
    [ "$_a" = "up" ] && _GATE_UP=1
done
if [ "$_GATE_UP" = "1" ] && [ -f "scripts/check_sql_migrations_applied.py" ]; then
    if [ "${ORION_ALLOW_UNAPPLIED_MIGRATION:-}" = "1" ]; then
        echo "safe_docker_build.sh: ORION_ALLOW_UNAPPLIED_MIGRATION=1 -- SKIPPING the SQL migration gate for $SERVICE." >&2
    else
        # The repo venv has psycopg2; system python3 does not. Linked worktrees have
        # no .venv of their own, so reach the main checkout's via git's common dir
        # (same lookup as the Makefile's METRIC_PYTHON).
        _GATE_PY=python3
        _CD=$(git rev-parse --git-common-dir 2>/dev/null || true)
        if [ -x .venv/bin/python ]; then
            _GATE_PY=.venv/bin/python
        elif [ -n "$_CD" ] && [ -x "$_CD/../.venv/bin/python" ]; then
            _GATE_PY="$_CD/../.venv/bin/python"
        fi
        _GATE_RC=0
        "$_GATE_PY" scripts/check_sql_migrations_applied.py --service "$SERVICE" || _GATE_RC=$?
        if [ "$_GATE_RC" = "1" ]; then
            echo "REFUSING to bring up $SERVICE: a SQL migration it needs is not applied (apply command above)." >&2
            echo "Deliberate exception: ORION_ALLOW_UNAPPLIED_MIGRATION=1 scripts/safe_docker_build.sh $SERVICE ..." >&2
            exit 1
        elif [ "$_GATE_RC" != "0" ]; then
            echo "REFUSING to bring up $SERVICE: whether its required SQL migrations are applied is UNKNOWN (see above)." >&2
            echo "Deliberate exception: ORION_ALLOW_UNAPPLIED_MIGRATION=1 scripts/safe_docker_build.sh $SERVICE ..." >&2
            exit 1
        fi
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
    -f "services/$SERVICE/$COMPOSE_FILE_NAME" \
    "$@"
