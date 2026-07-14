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
# Any path inside the repo (including a subdirectory) works -- resolution
# goes through `git rev-parse`, same as git itself uses.
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

if ! git -C "$TARGET_DIR" rev-parse --git-common-dir >/dev/null 2>&1; then
    echo "[install-git-safety-hooks] ERROR: '$TARGET_DIR' is not inside a git repository" >&2
    exit 1
fi

# Resolve the shared/common git dir (this is where hooks live even for a
# linked worktree checkout -- hooks are not per-worktree).
COMMON_DIR_RAW=$(git -C "$TARGET_DIR" rev-parse --git-common-dir)
case "$COMMON_DIR_RAW" in
    /*)
        COMMON_DIR="$COMMON_DIR_RAW"
        ;;
    *)
        COMMON_DIR=$(cd "$TARGET_DIR/$COMMON_DIR_RAW" && pwd)
        ;;
esac

TOPLEVEL=$(git -C "$TARGET_DIR" rev-parse --show-toplevel)

# Respect core.hooksPath if the repo has customized it.
HOOKS_PATH_CFG=$(git -C "$TARGET_DIR" config --get core.hooksPath 2>/dev/null) || HOOKS_PATH_CFG=""

if [ -n "$HOOKS_PATH_CFG" ]; then
    # Git expands a leading ~ (or ~user) in core.hooksPath to the home
    # directory; do the same here, since a value read out of `git config
    # --get` at runtime never goes through the shell's own tilde expansion.
    case "$HOOKS_PATH_CFG" in
        "~")
            HOOKS_PATH_CFG="$HOME"
            ;;
        "~/"*)
            HOOKS_PATH_CFG="$HOME/${HOOKS_PATH_CFG#\~/}"
            ;;
    esac
    case "$HOOKS_PATH_CFG" in
        /*)
            HOOKS_DIR="$HOOKS_PATH_CFG"
            ;;
        *)
            # A relative core.hooksPath is resolved by git relative to the
            # top level of the working tree, NOT relative to $GIT_DIR.
            HOOKS_DIR="$TOPLEVEL/$HOOKS_PATH_CFG"
            ;;
    esac
else
    HOOKS_DIR="$COMMON_DIR/hooks"
fi

mkdir -p "$HOOKS_DIR"

DEST_HOOK="$HOOKS_DIR/pre-commit"
MARKER="# orion-git-safety-guard"

# Refuse to write through an existing symlink rather than silently following
# it -- a symlinked hooks dir (e.g. dotfiles/chezmoi-managed hooks) would
# otherwise get its *target* file overwritten, which can live outside this
# repo entirely and outside what an operator expects this script to touch.
if [ -L "$DEST_HOOK" ]; then
    echo "[install-git-safety-hooks] ERROR: $DEST_HOOK is a symlink (-> $(readlink "$DEST_HOOK" 2>/dev/null || echo '?'))." >&2
    echo "  Refusing to write through it. Remove the symlink or point it at" >&2
    echo "  $SOURCE_HOOK yourself, then re-run this script." >&2
    exit 1
fi

if [ -f "$DEST_HOOK" ] && grep -qF "$MARKER" "$DEST_HOOK" 2>/dev/null; then
    # Already our hook -- always refresh to the current version rather than
    # skipping, so a later fix to scripts/git_hooks/pre-commit actually
    # reaches an already-installed checkout when this script is re-run.
    cp "$SOURCE_HOOK" "$DEST_HOOK"
    chmod +x "$DEST_HOOK"
    echo "[install-git-safety-hooks] refreshed orion git-safety pre-commit hook at $DEST_HOOK"
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
