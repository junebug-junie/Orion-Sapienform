#!/bin/sh
# Installs scripts/git_hooks/orion-git-shim as `git` in a directory earlier
# in PATH than the real git binary, and opts the target repo in by
# creating a marker file at <shared-checkout>/.git/orion-safety-guard-enabled.
#
# The marker lives in the shared/common git dir (resolved via
# `git rev-parse --git-common-dir`, same as scripts/install_git_safety_hooks.sh
# resolves hook locations), not wherever TARGET_DIR happens to be -- so
# running this from any worktree of the repo correctly protects the one
# true shared checkout, matching that installer's existing behavior.
# A plain marker FILE (not a git-config flag) is deliberate: the shim's own
# gate check is a single `[ -f ... ]`, no subprocess, checked on every
# opted-in-candidate git invocation -- see the shim's own comments for why
# that matters at this repo's scale.
#
# This is a bigger-footprint change than scripts/install_git_safety_hooks.sh:
# that one only touches files inside a single repo's .git/hooks/. This one
# changes what `git` resolves to for the invoking user's whole shell
# environment, system-wide, for every repo -- though the shim itself only
# ever blocks anything for repos that have explicitly opted in via the
# marker file this script creates, so unrelated repos are unaffected either
# way. Companion installer: scripts/install_git_safety_hooks.sh sets up the
# commit-time guard (pre-commit) and the worktree-hygiene nudge
# (post-merge); this one sets up the destructive-command guard (clean -f*,
# reset --hard, checkout/switch --force). Run both for full coverage --
# running only one leaves the other class of command unguarded.
#
# Usage:
#   scripts/install_orion_git_shim.sh [target-repo-dir]
#
# Safe to re-run: refreshes the shim to the current version, re-creates the
# marker file (idempotent), does not duplicate anything.
#
# POSIX sh only -- no bashisms.

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOURCE_SHIM="$SCRIPT_DIR/git_hooks/orion-git-shim"
TARGET_DIR=${1:-.}

if [ ! -f "$SOURCE_SHIM" ]; then
    echo "[install-orion-git-shim] ERROR: source shim not found at $SOURCE_SHIM" >&2
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "[install-orion-git-shim] ERROR: target directory does not exist: $TARGET_DIR" >&2
    exit 1
fi

if ! git -C "$TARGET_DIR" rev-parse --git-common-dir >/dev/null 2>&1; then
    echo "[install-orion-git-shim] ERROR: '$TARGET_DIR' is not inside a git repository" >&2
    exit 1
fi

SHIM_DIR="${ORION_GIT_SHIM_DIR:-$HOME/.local/bin}"
mkdir -p "$SHIM_DIR"
DEST_SHIM="$SHIM_DIR/git"

if [ -L "$DEST_SHIM" ] && [ "$(readlink "$DEST_SHIM" 2>/dev/null)" != "$SOURCE_SHIM" ]; then
    echo "[install-orion-git-shim] ERROR: $DEST_SHIM is a symlink to something else (-> $(readlink "$DEST_SHIM")). Refusing to overwrite it." >&2
    exit 1
fi

if [ -f "$DEST_SHIM" ] && ! grep -qF "orion-git-shim" "$DEST_SHIM" 2>/dev/null; then
    echo "[install-orion-git-shim] ERROR: $DEST_SHIM already exists and is not an orion-git-shim install (no marker found)." >&2
    echo "  Refusing to overwrite an unrelated file. If $SHIM_DIR/git is meant to be" >&2
    echo "  something else on this machine, set ORION_GIT_SHIM_DIR to a different" >&2
    echo "  directory (that is still earlier in PATH than the real git) and re-run." >&2
    exit 1
fi

cp "$SOURCE_SHIM" "$DEST_SHIM"
chmod +x "$DEST_SHIM"
echo "[install-orion-git-shim] installed shim at $DEST_SHIM"

# Verify the shim is actually reachable ahead of the real git on PATH --
# install succeeding at the file level doesn't guarantee PATH ordering.
_RESOLVED=$(command -v git 2>/dev/null || echo "")
_RESOLVED_REAL=$(readlink -f "$_RESOLVED" 2>/dev/null || echo "$_RESOLVED")
_SHIM_REAL=$(readlink -f "$DEST_SHIM" 2>/dev/null || echo "$DEST_SHIM")
if [ "$_RESOLVED_REAL" != "$_SHIM_REAL" ]; then
    echo "[install-orion-git-shim] WARNING: '$SHIM_DIR' does not appear to be ahead of the real git on this shell's current PATH (command -v git resolved to: $_RESOLVED)." >&2
    echo "  The shim is installed but may not actually be used until PATH is fixed" >&2
    echo "  (check your shell rc file's PATH order, or open a new shell)." >&2
fi

# Resolve the shared/common git dir -- NOT TARGET_DIR directly, since
# TARGET_DIR might be a linked worktree. Same relative/absolute handling
# as scripts/install_git_safety_hooks.sh's COMMON_DIR resolution.
_COMMON_DIR_RAW=$(git -C "$TARGET_DIR" rev-parse --git-common-dir)
case "$_COMMON_DIR_RAW" in
    /*)
        _COMMON_DIR="$_COMMON_DIR_RAW"
        ;;
    *)
        _COMMON_DIR=$(cd "$TARGET_DIR/$_COMMON_DIR_RAW" && pwd)
        ;;
esac

touch "$_COMMON_DIR/orion-safety-guard-enabled"
echo "[install-orion-git-shim] opted in: $(git -C "$TARGET_DIR" rev-parse --show-toplevel) (marker: $_COMMON_DIR/orion-safety-guard-enabled)"
