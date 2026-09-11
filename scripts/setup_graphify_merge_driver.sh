#!/bin/sh
# Registers Orion's three-way union-merge driver for graphify-out/graph.json in the
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

# The LFS-aware wrapper resolves all three pointer inputs, invokes the
# repo's stdlib JSON merger, then cleans its result back to an LFS pointer.
# This handles the repo's >100MB graph and preserves top-level metadata.
SCRIPT_DIR_FOR_DRIVER="$(cd "$(dirname "$0")" && pwd)"
EXPECTED_DRIVER="$SCRIPT_DIR_FOR_DRIVER/graphify_lfs_merge_driver.sh %O %A %B"
ATTR_LINE='graphify-out/graph.json filter=lfs diff=lfs merge=graphify -text'

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

# 1. The merger is stdlib-only; no graphify CLI is needed during a git merge.
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required by the JSON merge helper." >&2
    exit 1
fi

# 1b. Confirm the LFS-aware wrapper this driver command points at actually
# exists and is executable -- a missing/non-executable wrapper would
# otherwise only surface as a cryptic failure the next time git tries to
# merge graphify-out/graph.json.
if [ ! -x "$SCRIPT_DIR_FOR_DRIVER/graphify_lfs_merge_driver.sh" ]; then
    echo "error: expected wrapper not found or not executable: $SCRIPT_DIR_FOR_DRIVER/graphify_lfs_merge_driver.sh" >&2
    exit 1
fi

# 1c. Confirm git-lfs itself is on PATH -- the wrapper depends on
# `git lfs smudge`/`git lfs clean` at merge time. Catching a missing
# git-lfs install here (setup time) is far easier to diagnose than letting
# it surface as a `command not found` buried inside the wrapper's stderr
# during someone's first real merge.
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "error: 'git-lfs' is not on PATH. The merge driver this script configures" >&2
    echo "  depends on it. Install it (e.g. 'apt install git-lfs' or see" >&2
    echo "  https://git-lfs.com), then re-run this script." >&2
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
