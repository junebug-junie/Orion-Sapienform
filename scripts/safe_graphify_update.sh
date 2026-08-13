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
MANIFEST_FILE="graphify-out/manifest.json"
THRESHOLD_PCT="${GRAPHIFY_UPDATE_MAX_NODE_LOSS_PCT:-10}"

# Every artifact `graphify update .` may rewrite or create. All of them are
# restored together -- a partial restore is what produced the second incident.
# Basenames must stay unique; they are the backup filenames.
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

# graphify also drops a dated snapshot dir (graphify-out/<YYYY-MM-DD>/). It is
# NOT auto-removed here: deleting directories is a destructive op needing
# explicit approval per CLAUDE.md 13, and that dir is the recovery source when
# a restore is incomplete. Named so it can be cleaned up deliberately.
_warn_snapshot_dirs() {
    for _d in graphify-out/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]; do
        [ -d "$_d" ] || continue
        echo "" >&2
        echo "[safe-graphify-update] NOTE: snapshot dir $_d is present (left in place," >&2
        echo "  not auto-removed). It duplicates graphify-out/ artifacts; remove it" >&2
        echo "  deliberately once you have confirmed the restore is correct." >&2
    done
}

_backup_artifacts

echo "[safe-graphify-update] before: $BEFORE nodes. running: graphify update . $*"
if ! graphify update . "$@"; then
    echo "[safe-graphify-update] ERROR: graphify update itself failed (nonzero exit) -- restoring backup" >&2
    _restore_artifacts
    rm -rf "$BACKUP_DIR"
    exit 1
fi

AFTER=$(_count_nodes) || {
    echo "[safe-graphify-update] ERROR: could not read node count after update -- restoring backup" >&2
    _restore_artifacts
    rm -rf "$BACKUP_DIR"
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
    rm -rf "$BACKUP_DIR"
    _warn_snapshot_dirs
    echo "" >&2
    echo "[safe-graphify-update] Verify with: git status --short graphify-out/" >&2
    exit 1
fi

echo "[safe-graphify-update] OK: node count $BEFORE -> $AFTER (~${DROP_PCT}% change, within ${THRESHOLD_PCT}% threshold)."
rm -rf "$BACKUP_DIR"
exit 0
