#!/bin/sh
# orion-git-safety-guard installer
#
# Installs scripts/git_hooks/pre-commit (the shared/primary-checkout guard)
# into the correct git hooks directory for a repo -- respecting
# core.hooksPath when it's configured, and resolving linked-worktree hook
# locations correctly (hooks live in the shared/common git dir, not
# per-worktree).
#
# Usage:
#   scripts/install_git_safety_hooks.sh [target-repo-dir]
#
# If target-repo-dir is omitted, the current working directory is used.
# The script walks up from there to find the repo (a directory containing
# a .git entry -- directory or file, so this also works from inside a
# linked worktree).
#
# POSIX sh only -- no bashisms.

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOURCE_HOOK="$SCRIPT_DIR/git_hooks/pre-commit"

if [ ! -f "$SOURCE_HOOK" ]; then
    echo "[install-git-safety-hooks] ERROR: source hook not found at $SOURCE_HOOK" >&2
    exit 1
fi

TARGET_DIR=${1:-.}

if [ ! -d "$TARGET_DIR" ]; then
    echo "[install-git-safety-hooks] ERROR: target directory does not exist: $TARGET_DIR" >&2
    exit 1
fi

# Walk up from TARGET_DIR looking for a .git entry (dir or file -- a file
# means we're in a linked worktree, which is fine).
find_repo_root() {
    dir=$(cd "$1" && pwd)
    while [ "$dir" != "/" ]; do
        if [ -e "$dir/.git" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
        dir=$(dirname "$dir")
    done
    return 1
}

REPO_ROOT=$(find_repo_root "$TARGET_DIR") || {
    echo "[install-git-safety-hooks] ERROR: no .git found walking up from $TARGET_DIR" >&2
    exit 1
}

if ! git -C "$REPO_ROOT" rev-parse --git-common-dir >/dev/null 2>&1; then
    echo "[install-git-safety-hooks] ERROR: '$REPO_ROOT' is not a git repository" >&2
    exit 1
fi

# Resolve the shared/common git dir (this is where hooks live even for a
# linked worktree checkout -- hooks are not per-worktree).
COMMON_DIR_RAW=$(git -C "$REPO_ROOT" rev-parse --git-common-dir)
case "$COMMON_DIR_RAW" in
    /*)
        COMMON_DIR="$COMMON_DIR_RAW"
        ;;
    *)
        COMMON_DIR=$(cd "$REPO_ROOT/$COMMON_DIR_RAW" && pwd)
        ;;
esac

# Respect core.hooksPath if the repo has customized it.
HOOKS_PATH_CFG=$(git -C "$REPO_ROOT" config --get core.hooksPath 2>/dev/null) || HOOKS_PATH_CFG=""

if [ -n "$HOOKS_PATH_CFG" ]; then
    case "$HOOKS_PATH_CFG" in
        /*)
            HOOKS_DIR="$HOOKS_PATH_CFG"
            ;;
        *)
            # A relative core.hooksPath is resolved by git relative to the
            # top level of the working tree, NOT relative to $GIT_DIR.
            TOPLEVEL=$(git -C "$REPO_ROOT" rev-parse --show-toplevel)
            HOOKS_DIR="$TOPLEVEL/$HOOKS_PATH_CFG"
            ;;
    esac
else
    HOOKS_DIR="$COMMON_DIR/hooks"
fi

mkdir -p "$HOOKS_DIR"

DEST_HOOK="$HOOKS_DIR/pre-commit"
MARKER="# orion-git-safety-guard"

if [ -f "$DEST_HOOK" ] && grep -qF "$MARKER" "$DEST_HOOK" 2>/dev/null; then
    echo "[install-git-safety-hooks] already installed at $DEST_HOOK"
    exit 0
fi

if [ -f "$DEST_HOOK" ]; then
    BACKUP="$HOOKS_DIR/pre-commit.pre-orion-safety.bak"
    cp "$DEST_HOOK" "$BACKUP"
    echo "[install-git-safety-hooks] existing pre-commit hook at $DEST_HOOK preserved as $BACKUP -- merge the two by hand if the old hook did something you still need" >&2
fi

cp "$SOURCE_HOOK" "$DEST_HOOK"
chmod +x "$DEST_HOOK"

echo "[install-git-safety-hooks] installed orion git-safety pre-commit hook at $DEST_HOOK"
