#!/bin/sh
# safe_graphify_update.sh -- wraps `graphify update .` with a sanity check
# against catastrophic node-count loss.
#
# Real incident (2026-07-14): `graphify update .` silently shrank
# graphify-out/graph.json from 31566 nodes to 1559 (a ~95% loss) despite
# its own log output claiming to have "kept" the old nodes ("fail-closed:
# kept N nodes from M files that left the scan corpus"). That message does
# not accurately describe what actually lands in the rebuilt graph.json.
# The result was committed and merged to main before being caught.
# Reproduced identically from both the main checkout and a worktree -- not
# purely a worktree-scope issue. Root cause not understood. Until it is,
# treat every `graphify update .` result as untrusted until this script's
# sanity check passes -- do not run the raw command directly and trust its
# own log output.
#
# Usage:
#   scripts/safe_graphify_update.sh [graphify update args...]
#
# Exit 0: update applied, node count did not drop more than the threshold
#         (GRAPHIFY_UPDATE_MAX_NODE_LOSS_PCT, default 10) -- safe to commit.
# Exit 1: update was destructive -- every artifact listed in ARTIFACTS is
#         automatically restored to its pre-update state, nothing to
#         commit. Investigate before retrying; do not just re-run this.
#
# Second incident (found 2026-08-12): this script originally backed up only
# graph.json and manifest.json, but `graphify update .` also rewrites
# GRAPH_REPORT.md (and .graphify_labels.json, and creates graph.html). Those
# were never captured and never restored, so a REFUSED run restored the graph
# and left the SHRUNKEN REPORT sitting in the working tree -- while printing
# "there is nothing to commit", which was false every single time it fired.
# Measured in the shared checkout that day: GRAPH_REPORT.md read 2471 nodes /
# 3210 edges next to a graph.json holding 28306 nodes / 81046 links, a 91.3%
# disagreement. CLAUDE.md points every agent at GRAPH_REPORT.md for broad
# architecture review, so the stale copy was actively telling agents that
# whole services, schemas, and bus channels did not exist.
#
# POSIX sh only -- no bashisms.

set -e

GRAPH_FILE="graphify-out/graph.json"
THRESHOLD_PCT="${GRAPHIFY_UPDATE_MAX_NODE_LOSS_PCT:-10}"
SNAPSHOT_GLOB="graphify-out/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"

# Every artifact `graphify update .` may rewrite or create. All of them are
# restored together -- a partial restore is what produced the second incident.
# Basenames must stay unique; they are the backup filenames.
#
# .graphify_labels.json is on this list for a measured reason, not a guessed
# one: during the 2026-08-12 cleanup the leaked top-level copy held 290
# entries -- exactly the community count of the DESTROYED graph, whose report
# read "290 communities" -- while the pre-update copy held 1395, matching the
# real graph's 1395. It was leaking on every refused run alongside the report,
# just less visibly because it is gitignored scratch state.
ARTIFACTS="graphify-out/graph.json
graphify-out/manifest.json
graphify-out/GRAPH_REPORT.md
graphify-out/graph.html
graphify-out/.graphify_labels.json
graphify-out/.graphify_learning.json"

if [ ! -f "$GRAPH_FILE" ]; then
    echo "[safe-graphify-update] ERROR: $GRAPH_FILE not found -- nothing to compare against, refusing to run an unguarded update." >&2
    exit 1
fi

_count_nodes() {
    python3 -c "
import json, sys
try:
    with open('$GRAPH_FILE', encoding='utf-8') as f:
        d = json.load(f)
    print(len(d.get('nodes', [])))
except Exception as exc:
    print(f'ERROR: {exc}', file=sys.stderr)
    sys.exit(1)
"
}

BEFORE=$(_count_nodes) || {
    echo "[safe-graphify-update] ERROR: could not read node count before update -- refusing to run" >&2
    exit 1
}

BACKUP_DIR=$(mktemp -d)
# The backup set grew from 2 files to 6 (~46MB, dominated by graph.json and
# graph.html) and `graphify update` on this repo runs for minutes, so a Ctrl-C
# would otherwise leak that into /tmp. Cleans up on any exit path.
trap 'rm -rf "$BACKUP_DIR"' EXIT INT TERM

_backup_artifacts() {
    for _f in $ARTIFACTS; do
        _base=$(basename "$_f")
        if [ -f "$_f" ]; then
            cp "$_f" "$BACKUP_DIR/$_base"
        else
            # Record that it did NOT exist, so restore removes anything the
            # update created. graph.html is exactly this case: it kept
            # accumulating as untracked junk after every refused run.
            : > "$BACKUP_DIR/$_base.absent"
        fi
    done
}

_restore_artifacts() {
    for _f in $ARTIFACTS; do
        _base=$(basename "$_f")
        if [ -f "$BACKUP_DIR/$_base.absent" ]; then
            rm -f "$_f"
        elif [ -f "$BACKUP_DIR/$_base" ]; then
            cp "$BACKUP_DIR/$_base" "$_f"
        fi
    done
}

# graphify drops a dated snapshot dir (graphify-out/<YYYY-MM-DD>/) holding a
# PRE-update copy of its outputs. That ordering is not an assumption -- it is
# established from tracked git history: in commit eae14a6c, which added
# graphify-out/2026-07-29/, that snapshot's graph.json has 32529 nodes, the
# top-level graph.json in the SAME commit has 28307, and the top-level in the
# PARENT commit has 32529. The snapshot matches the parent, i.e. pre-update.
# It is therefore a genuine recovery source, which is exactly how the
# 2026-08-12 report and label-cache restore was performed.
#
# NOT auto-removed: deleting directories is a destructive op needing explicit
# approval per CLAUDE.md 13, and it is the recovery source. Only dirs CREATED
# BY THIS RUN are named -- an earlier version globbed every dated dir and told
# the operator to delete it, which would have included the git-TRACKED
# graphify-out/2026-07-29/, i.e. advice to delete committed repo content.
_list_snapshot_dirs() {
    for _d in $SNAPSHOT_GLOB; do
        [ -d "$_d" ] && echo "$_d"
    done
    # Explicit success REQUIRED. With no dated dir present the glob stays
    # literal, `[ -d ]` is false, and that false becomes the function's exit
    # status -- which under `set -e` killed the whole script before its first
    # log line, on the common path where no snapshot exists yet.
    return 0
}

_warn_snapshot_dirs() {
    _new=""
    for _d in $(_list_snapshot_dirs); do
        case "
$SNAPSHOT_DIRS_BEFORE
" in
            *"
$_d
"*) continue ;;
        esac
        _new="$_new $_d"
    done
    [ -n "$_new" ] || return 0
    echo "" >&2
    echo "[safe-graphify-update] NOTE: this run created snapshot dir(s):$_new" >&2
    echo "  They hold a PRE-update copy of graphify-out/ (~45MB each) and are" >&2
    echo "  gitignored, so they will NOT appear in git status. Left in place as" >&2
    echo "  the recovery source; remove deliberately once the restore is verified." >&2
}

# Recorded BEFORE the update so _warn_snapshot_dirs reports only what this run
# created, never a pre-existing (possibly git-tracked) dated dir. Must come
# after the function definitions above -- sh resolves functions at call time,
# and under `set -e` an undefined-function call kills the script outright.
SNAPSHOT_DIRS_BEFORE=$(_list_snapshot_dirs)

_backup_artifacts

echo "[safe-graphify-update] before: $BEFORE nodes. running: graphify update . $*"
if ! graphify update . "$@"; then
    echo "[safe-graphify-update] ERROR: graphify update itself failed (nonzero exit) -- restoring backup" >&2
    _restore_artifacts
    _warn_snapshot_dirs
    echo "[safe-graphify-update] Verify with: git status --short graphify-out/" >&2
    exit 1
fi

AFTER=$(_count_nodes) || {
    echo "[safe-graphify-update] ERROR: could not read node count after update -- restoring backup" >&2
    _restore_artifacts
    _warn_snapshot_dirs
    echo "[safe-graphify-update] Verify with: git status --short graphify-out/" >&2
    exit 1
}

# Percent drop via awk -- portable, no bc dependency. Clamped to 0 for an
# increase or an empty/zero baseline (nothing to divide by).
DROP_PCT=$(awk -v b="$BEFORE" -v a="$AFTER" 'BEGIN { if (b <= 0) { print 0 } else { d = (b - a) / b * 100; print (d < 0) ? 0 : d } }')
EXCEEDS=$(awk -v d="$DROP_PCT" -v t="$THRESHOLD_PCT" 'BEGIN { print (d > t) ? 1 : 0 }')

if [ "$EXCEEDS" = "1" ]; then
    echo "" >&2
    echo "[safe-graphify-update] REFUSED: node count dropped from $BEFORE to $AFTER (~${DROP_PCT}%, threshold ${THRESHOLD_PCT}%)." >&2
    echo "  This matches the known destructive-update failure mode from the 2026-07-14" >&2
    echo "  incident. Restoring graph.json, manifest.json, GRAPH_REPORT.md, graph.html," >&2
    echo "  and the .graphify_* caches to their pre-update state." >&2
    echo "  Do not re-run graphify update . directly and trust its own log output." >&2
    echo "  Investigate first, or use a full re-extraction instead of incremental update." >&2
    _restore_artifacts
    _warn_snapshot_dirs
    echo "" >&2
    echo "[safe-graphify-update] Verify with: git status --short graphify-out/" >&2
    echo "  (snapshot dirs are gitignored, so check any named above by hand)" >&2
    exit 1
fi

echo "[safe-graphify-update] OK: node count $BEFORE -> $AFTER (~${DROP_PCT}% change, within ${THRESHOLD_PCT}% threshold)."
# Also fires on the success path: dated snapshots are gitignored now, so a
# healthy run would otherwise add ~45MB to disk with no signal at all.
_warn_snapshot_dirs
exit 0
