#!/bin/sh
# One obvious front door for creating a worktree, matching the sibling-
# directory convention AGENTS.md section 2 documents (as opposed to the
# other two legitimate-but-undocumented-here patterns this repo also has:
# .worktrees/ and .claude/worktrees/agent-<id>, both driven by other
# tooling this script doesn't touch or replace).
#
# Usage:
#   scripts/new_worktree.sh <type> <name>
#
# Example:
#   scripts/new_worktree.sh feat my-thing
#   -> creates ../Orion-Sapienform-my-thing on branch feat/my-thing
#
# type must be one of: feat fix chore docs test (matches AGENTS.md section 2's
# branch type examples).

set -e

TYPE="$1"
NAME="$2"

if [ -z "$TYPE" ] || [ -z "$NAME" ]; then
    echo "Usage: scripts/new_worktree.sh <type> <name>" >&2
    echo "  type: feat | fix | chore | docs | test" >&2
    exit 1
fi

case "$TYPE" in
    feat|fix|chore|docs|test) ;;
    *)
        echo "error: type must be one of feat, fix, chore, docs, test (got: $TYPE)" >&2
        exit 1
        ;;
esac

case "$NAME" in
    *[!a-zA-Z0-9_-]*)
        echo "error: name must contain only letters, digits, - and _ (got: $NAME)" >&2
        exit 1
        ;;
esac

REPO_ROOT=$(git rev-parse --show-toplevel)
REPO_NAME=$(basename "$REPO_ROOT")
TARGET_DIR="$(dirname "$REPO_ROOT")/${REPO_NAME}-${NAME}"
BRANCH="${TYPE}/${NAME}"

if [ -e "$TARGET_DIR" ]; then
    echo "error: $TARGET_DIR already exists" >&2
    exit 1
fi

# This script only ever creates the sibling-directory convention -- it does
# NOT unify or replace the other two worktree location patterns this repo
# also has (.worktrees/, .claude/worktrees/agent-<id>, both driven by other
# tooling). Warn, not block, if a worktree whose path or branch already
# mentions this name exists under ANY convention, so a second attempt at
# the same task under a different mechanism doesn't happen silently.
_EXISTING=$(git -C "$REPO_ROOT" worktree list --porcelain | grep -iE "(^worktree .*${NAME}|^branch .*${NAME})" || true)
if [ -n "$_EXISTING" ]; then
    echo "warning: a worktree already mentions \"$NAME\" -- check this isn't a duplicate of existing work:" >&2
    echo "$_EXISTING" | sed 's/^/  /' >&2
    echo "" >&2
fi

_STDERR_TMP=$(mktemp)
trap 'rm -f "$_STDERR_TMP"' EXIT

if ! git -C "$REPO_ROOT" worktree add "$TARGET_DIR" -b "$BRANCH" 2>"$_STDERR_TMP"; then
    if grep -q "already exists" "$_STDERR_TMP" 2>/dev/null; then
        echo "error: branch $BRANCH already exists. If its worktree was already pruned" >&2
        echo "  (see scripts/prune_merged_worktrees.py), reuse it with:" >&2
        echo "    git worktree add $TARGET_DIR $BRANCH" >&2
        echo "  or delete the stale branch first with: git branch -d $BRANCH" >&2
    else
        cat "$_STDERR_TMP" >&2
    fi
    exit 1
fi

echo ""
echo "Created. Next:"
echo "  cd $TARGET_DIR"
