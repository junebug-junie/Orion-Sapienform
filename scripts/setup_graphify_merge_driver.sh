#!/bin/sh
# Registers graphify's union-merge driver for graphify-out/graph.json in the
# CURRENT clone's local .git/config. This step is NOT committable by git's
# own design (.git/config is never tracked), so every clone/worktree that
# wants conflict-free graph.json merges must run this script once.
#
# Idempotent: safe to run multiple times, safe to run in a repo that already
# has the driver configured.
#
# Usage:
#   scripts/setup_graphify_merge_driver.sh [REPO_PATH]
#
# REPO_PATH defaults to the repo containing this script. Pass a path to
# target a different repo (e.g. for testing against a throwaway repo).

set -eu

EXPECTED_DRIVER='graphify merge-driver %O %A %B'
ATTR_LINE='graphify-out/graph.json merge=graphify'

# Resolve the target repo root.
#
# If REPO_PATH is given, walk up from there to find .git. Otherwise walk up
# from this script's own location (mirrors the REPO_ROOT pattern used
# elsewhere in this repo, e.g. scripts/collapse_mirror_live_path_truth.sh /
# scripts/git-stash-table.sh) rather than assuming cwd is repo root.
find_repo_root() {
    start_dir="$1"
    dir="$start_dir"
    while [ "$dir" != "/" ]; do
        if [ -e "$dir/.git" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

if [ "${1:-}" != "" ]; then
    start_dir="$(cd "$1" 2>/dev/null && pwd)" || {
        echo "error: REPO_PATH '$1' does not exist" >&2
        exit 1
    }
else
    script_dir="$(cd "$(dirname "$0")" && pwd)"
    start_dir="$script_dir"
fi

REPO_ROOT="$(find_repo_root "$start_dir")" || {
    echo "error: could not find a .git directory above '$start_dir'" >&2
    exit 1
}

cd "$REPO_ROOT"

# 1. Confirm graphify is available.
if ! command -v graphify >/dev/null 2>&1; then
    echo "error: 'graphify' is not on PATH." >&2
    echo "  Install it with: pip install graphifyy" >&2
    if [ -x "graphify-out/.graphify_python" ]; then
        echo "  This repo also has a pinned interpreter at graphify-out/.graphify_python" >&2
        echo "  that can run it as: \$(cat graphify-out/.graphify_python) -m graphify ..." >&2
        echo "  but the plain 'graphify' command should be on PATH for this driver to work." >&2
    fi
    exit 1
fi

# 2. Configure the local (repo-scoped, NOT --global) merge driver.
current_driver="$(git config --get merge.graphify.driver 2>/dev/null || true)"
if [ "$current_driver" = "$EXPECTED_DRIVER" ]; then
    echo "merge.graphify.driver already configured in $REPO_ROOT/.git/config"
else
    git config merge.graphify.driver "$EXPECTED_DRIVER"
    echo "set merge.graphify.driver in $REPO_ROOT/.git/config"
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
