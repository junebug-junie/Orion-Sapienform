#!/bin/sh
# orion-git-safety-guard installer
#
# Installs scripts/git_hooks/pre-commit (the shared/primary-checkout guard),
# scripts/git_hooks/post-merge (the worktree-hygiene nudge), and
# scripts/git_hooks/post-commit (the agent-board heartbeat nudge) into the
# correct git hooks directory for a repo -- respecting core.hooksPath when
# it's configured, and resolving linked-worktree hook locations correctly
# (hooks live in the shared/common git dir, not per-worktree).
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

# Validate all source hooks exist BEFORE any mutation (mkdir -p below) --
# failing here must not leave a side effect like a freshly-created, empty
# custom hooks directory behind.
for _hook_name in pre-commit post-merge post-commit; do
    if [ ! -f "$SCRIPT_DIR/git_hooks/$_hook_name" ]; then
        echo "[install-git-safety-hooks] ERROR: source hook not found at $SCRIPT_DIR/git_hooks/$_hook_name" >&2
        exit 1
    fi
done

mkdir -p "$HOOKS_DIR"

install_one_hook() {
    local HOOK_NAME="$1"
    local SOURCE_HOOK="$SCRIPT_DIR/git_hooks/$HOOK_NAME"
    local DEST_HOOK="$HOOKS_DIR/$HOOK_NAME"
    local MARKER="# orion-git-safety-guard"
    local BACKUP

    # Refuse to write through an existing symlink rather than silently
    # following it -- a symlinked hooks dir (e.g. dotfiles/chezmoi-managed
    # hooks) would otherwise get its *target* file overwritten, which can
    # live outside this repo entirely and outside what an operator expects
    # this script to touch.
    if [ -L "$DEST_HOOK" ]; then
        echo "[install-git-safety-hooks] ERROR: $DEST_HOOK is a symlink (-> $(readlink "$DEST_HOOK" 2>/dev/null || echo '?'))." >&2
        echo "  Refusing to write through it. Remove the symlink or point it at" >&2
        echo "  $SOURCE_HOOK yourself, then re-run this script." >&2
        exit 1
    fi

    if [ -f "$DEST_HOOK" ] && grep -qF "$MARKER" "$DEST_HOOK" 2>/dev/null; then
        # Already our hook -- always refresh to the current version rather
        # than skipping, so a later fix to the source hook actually reaches
        # an already-installed checkout when this script is re-run.
        cp "$SOURCE_HOOK" "$DEST_HOOK"
        chmod +x "$DEST_HOOK"
        echo "[install-git-safety-hooks] refreshed orion git-safety $HOOK_NAME hook at $DEST_HOOK"
        return 0
    fi

    if [ -f "$DEST_HOOK" ]; then
        BACKUP="$HOOKS_DIR/$HOOK_NAME.pre-orion-safety.bak"
        cp "$DEST_HOOK" "$BACKUP"
        echo "[install-git-safety-hooks] existing $HOOK_NAME hook at $DEST_HOOK preserved as $BACKUP -- merge the two by hand if the old hook did something you still need" >&2
    fi

    cp "$SOURCE_HOOK" "$DEST_HOOK"
    chmod +x "$DEST_HOOK"

    echo "[install-git-safety-hooks] installed orion git-safety $HOOK_NAME hook at $DEST_HOOK"
}

_emit_post_commit_block() {
    # Emits our post-commit fragment, bounded by exact BEGIN/END marker
    # lines (not just the loose "$MARKER" substring) so a refresh can find
    # BOTH ends of our own block and preserve whatever sits on either side
    # of it -- including content some OTHER tool appended after ours later
    # (e.g. a `graphify hook install` re-run). $1 = source hook file path.
    echo "$MARKER (post-commit) BEGIN"
    echo "("
    tail -n +3 "$1"
    echo ") || true"
    echo "$MARKER (post-commit) END"
}

install_post_commit_hook() {
    # post-commit gets its own install path instead of install_one_hook's
    # overwrite-and-backup behavior: unlike pre-commit/post-merge, post-commit
    # is a hook name real, independently-installed tooling in this repo
    # already uses (`graphify hook install` ships a post-commit hook that
    # rebuilds the knowledge graph after every commit). Running install_one_hook
    # here would silently disable that on first install -- confirmed live
    # against this repo's own primary checkout, not a hypothetical.
    #
    # Appending our block after the foreign one is not enough by itself,
    # confirmed by a second live bug: graphify's hook does its own
    # `exit 0` for any commit made from a linked worktree (its rebuild only
    # runs from the primary checkout) -- and a bare `exit` in a concatenated
    # shell script terminates the WHOLE file, not just the fragment it's in.
    # With the two fragments pasted back to back, that exit silently killed
    # our block on every worktree commit, i.e. the actual common case this
    # repo's own convention requires (AGENTS.md section 2: worktrees only).
    # Each fragment is therefore wrapped in its own `( ... ) || true`
    # subshell: `exit` inside `( ... )` only ends that subshell, so one
    # fragment's early-out can never prevent the other from running,
    # regardless of install order.
    local SOURCE_HOOK="$SCRIPT_DIR/git_hooks/post-commit"
    local DEST_HOOK="$HOOKS_DIR/post-commit"
    local MARKER="# orion-git-safety-guard"

    if [ -L "$DEST_HOOK" ]; then
        echo "[install-git-safety-hooks] ERROR: $DEST_HOOK is a symlink (-> $(readlink "$DEST_HOOK" 2>/dev/null || echo '?'))." >&2
        echo "  Refusing to write through it. Remove the symlink or point it at" >&2
        echo "  $SOURCE_HOOK yourself, then re-run this script." >&2
        exit 1
    fi

    if [ ! -f "$DEST_HOOK" ]; then
        {
            echo "#!/bin/sh"
            echo ""
            _emit_post_commit_block "$SOURCE_HOOK"
        } > "$DEST_HOOK"
        chmod +x "$DEST_HOOK"
        echo "[install-git-safety-hooks] installed orion git-safety post-commit hook at $DEST_HOOK"
        return 0
    fi

    if grep -qF "$MARKER" "$DEST_HOOK" 2>/dev/null; then
        # Our block is already present. Find BOTH its BEGIN and END lines
        # (not just where it starts) so a refresh can preserve content on
        # EITHER side of it -- including anything another tool appended
        # after ours since the last install (confirmed live: re-running
        # `graphify hook install` after ours was already in place appends
        # graphify's content past our old block's end; an earlier version
        # of this refresh logic assumed our block was always the file's
        # tail and silently discarded that on the next refresh).
        local BEGIN_LINE END_LINE
        BEGIN_LINE=$(grep -nF "$MARKER (post-commit) BEGIN" "$DEST_HOOK" | head -1 | cut -d: -f1)
        END_LINE=$(grep -nF "$MARKER (post-commit) END" "$DEST_HOOK" | head -1 | cut -d: -f1)

        if [ -n "$BEGIN_LINE" ] && [ -n "$END_LINE" ]; then
            local PREFIX_FILE SUFFIX_FILE
            PREFIX_FILE=$(mktemp)
            SUFFIX_FILE=$(mktemp)
            if [ "$BEGIN_LINE" -gt 1 ]; then
                head -n "$((BEGIN_LINE - 1))" "$DEST_HOOK" > "$PREFIX_FILE"
            else
                : > "$PREFIX_FILE"
            fi
            tail -n "+$((END_LINE + 1))" "$DEST_HOOK" > "$SUFFIX_FILE"

            # Trim trailing blank lines from the prefix and leading blank
            # lines from the suffix so repeated refreshes don't accumulate
            # blank-line padding on either side of our block every time
            # this script is re-run.
            awk 'BEGIN{n=0} {lines[++n]=$0} END{while (n>0 && lines[n]=="") n--; for (i=1;i<=n;i++) print lines[i]}' "$PREFIX_FILE" > "$PREFIX_FILE.trimmed"
            awk 'BEGIN{started=0} {if (!started && $0=="") next; started=1; print}' "$SUFFIX_FILE" > "$SUFFIX_FILE.trimmed"

            {
                cat "$PREFIX_FILE.trimmed"
                echo ""
                _emit_post_commit_block "$SOURCE_HOOK"
                if [ -s "$SUFFIX_FILE.trimmed" ]; then
                    echo ""
                    cat "$SUFFIX_FILE.trimmed"
                fi
            } > "$DEST_HOOK.tmp"
            rm -f "$PREFIX_FILE" "$PREFIX_FILE.trimmed" "$SUFFIX_FILE" "$SUFFIX_FILE.trimmed"
            mv "$DEST_HOOK.tmp" "$DEST_HOOK"
            chmod +x "$DEST_HOOK"
            echo "[install-git-safety-hooks] refreshed orion git-safety post-commit hook block at $DEST_HOOK (any other hook content, before or after ours, preserved)"
            return 0
        fi
        # "$MARKER" matched as a loose substring (e.g. mentioned in some
        # other tool's comments) but this isn't a well-formed BEGIN/END
        # block of ours -- fall through to the append branch below rather
        # than guessing at boundaries that don't exist.
    fi

    # Foreign hook with no orion block yet -- wrap its existing body in a
    # subshell (isolating any `exit` it calls) and append ours, also
    # subshell-wrapped, rather than overwriting/backing up. Only skip the
    # foreign hook's own first line if it's actually a shebang -- a foreign
    # hook with no shebang would otherwise silently lose its real first
    # line of code.
    local FOREIGN_FIRST_LINE FOREIGN_BODY_START
    FOREIGN_FIRST_LINE=$(head -n 1 "$DEST_HOOK")
    case "$FOREIGN_FIRST_LINE" in
        "#!"*) FOREIGN_BODY_START=2 ;;
        *) FOREIGN_BODY_START=1 ;;
    esac
    {
        echo "("
        tail -n "+$FOREIGN_BODY_START" "$DEST_HOOK"
        echo ") || true"
        echo ""
        _emit_post_commit_block "$SOURCE_HOOK"
    } > "$DEST_HOOK.tmp"
    {
        echo "#!/bin/sh"
        cat "$DEST_HOOK.tmp"
    } > "$DEST_HOOK"
    rm -f "$DEST_HOOK.tmp"
    chmod +x "$DEST_HOOK"
    echo "[install-git-safety-hooks] appended orion git-safety post-commit hook block to existing $DEST_HOOK (existing hook content preserved, not backed up or replaced)"
}

install_one_hook pre-commit
install_one_hook post-merge
install_post_commit_hook
