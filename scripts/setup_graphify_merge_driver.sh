#!/bin/sh
# Registers graphify's union-merge driver for graphify-out/graph.json in the
# CURRENT clone's local git config. This step is NOT committable by git's
# own design (local git config is never tracked), so every clone/worktree
# that wants conflict-free graph.json merges must run this script once.
#
# Idempotent: safe to run multiple times, safe to run in a repo that already
# has the driver configured.
#
# Usage:
#   scripts/setup_graphify_merge_driver.sh [REPO_PATH]
#
# REPO_PATH defaults to the repo containing this script. Pass a path to
# target a different repo (e.g. for testing against a throwaway repo). Any
# path inside the repo (including a linked worktree) works.

set -eu

EXPECTED_DRIVER='graphify merge-driver %O %A %B'
ATTR_LINE='graphify-out/graph.json merge=graphify'

if [ "${1:-}" != "" ]; then
    start_dir="$1"
else
    start_dir="$(cd "$(dirname "$0")" && pwd)"
fi

if [ ! -d "$start_dir" ] && [ ! -e "$start_dir" ]; then
    echo "error: REPO_PATH '$start_dir' does not exist" >&2
    exit 1
fi

if ! git -C "$start_dir" rev-parse --git-common-dir >/dev/null 2>&1; then
    echo "error: '$start_dir' is not inside a git repository" >&2
    exit 1
fi

REPO_ROOT="$(git -C "$start_dir" rev-parse --show-toplevel)"
# `git config` without -C/--git-dir operates on the repo containing $PWD, so
# resolve everything from here relative to REPO_ROOT explicitly rather than
# depending on the caller's cwd.
cd "$REPO_ROOT"

# For the confirmation message only: where the config this script edits
# actually lives. In a linked worktree, "$REPO_ROOT/.git" is a *file* (a
# gitdir pointer), not a directory -- there is no "$REPO_ROOT/.git/config"
# in that case. `git config` (no --local override) still correctly targets
# the per-worktree-or-shared config that applies here; this just makes sure
# the printed path matches where it was actually written, for anyone who
# goes looking.
CONFIG_PATH="$(git rev-parse --git-common-dir)/config"
case "$CONFIG_PATH" in
    /*) : ;;
    *) CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH" ;;
esac

# 1. Confirm graphify is available.
if ! command -v graphify >/dev/null 2>&1; then
    echo "error: 'graphify' is not on PATH." >&2
    if [ -x "graphify-out/.graphify_python" ]; then
        echo "  This repo has a pinned interpreter at graphify-out/.graphify_python" >&2
        echo "  that can run it as: \$(cat graphify-out/.graphify_python) -m graphify ..." >&2
        echo "  but the plain 'graphify' command needs to be on PATH for the merge" >&2
        echo "  driver git invokes to find it." >&2
    fi
    echo "  Install it with: uv tool install graphifyy   (or: pip install graphifyy" >&2
    echo "  if you're not using uv -- check which one this machine actually has" >&2
    echo "  on PATH before picking a command)." >&2
    exit 1
fi

# 2. Configure the local (repo-scoped, NOT --global) merge driver.
current_driver="$(git config --get merge.graphify.driver 2>/dev/null || true)"
if [ "$current_driver" = "$EXPECTED_DRIVER" ]; then
    echo "merge.graphify.driver already configured in $CONFIG_PATH"
else
    git config merge.graphify.driver "$EXPECTED_DRIVER"
    echo "set merge.graphify.driver in $CONFIG_PATH"
fi

# 3. Fallback: ensure .gitattributes has the mapping line (normally already
# committed via the repo's own .gitattributes; this is only a safety net for
# someone running this script before that change has been pulled).
attrs_file="$REPO_ROOT/.gitattributes"
if [ -f "$attrs_file" ] && grep -qxF "$ATTR_LINE" "$attrs_file"; then
    : # already present
else
    printf '%s\n' "$ATTR_LINE" >> "$attrs_file"
    echo "appended '$ATTR_LINE' to $attrs_file"
fi

# 4. Confirmation.
echo "graphify merge driver ready: merge.graphify.driver='$(git config --get merge.graphify.driver)', .gitattributes maps graphify-out/graph.json -> merge=graphify"
