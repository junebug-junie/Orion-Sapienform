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
# Exit 1: update was destructive -- graph.json/manifest.json are
#         automatically restored to their pre-update state, nothing to
#         commit. Investigate before retrying; do not just re-run this.
#
# POSIX sh only -- no bashisms.

set -e

GRAPH_FILE="graphify-out/graph.json"
MANIFEST_FILE="graphify-out/manifest.json"
THRESHOLD_PCT="${GRAPHIFY_UPDATE_MAX_NODE_LOSS_PCT:-10}"

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
cp "$GRAPH_FILE" "$BACKUP_DIR/graph.json"
[ -f "$MANIFEST_FILE" ] && cp "$MANIFEST_FILE" "$BACKUP_DIR/manifest.json"

echo "[safe-graphify-update] before: $BEFORE nodes. running: graphify update . $*"
if ! graphify update . "$@"; then
    echo "[safe-graphify-update] ERROR: graphify update itself failed (nonzero exit) -- restoring backup" >&2
    cp "$BACKUP_DIR/graph.json" "$GRAPH_FILE"
    [ -f "$BACKUP_DIR/manifest.json" ] && cp "$BACKUP_DIR/manifest.json" "$MANIFEST_FILE"
    rm -rf "$BACKUP_DIR"
    exit 1
fi

AFTER=$(_count_nodes) || {
    echo "[safe-graphify-update] ERROR: could not read node count after update -- restoring backup" >&2
    cp "$BACKUP_DIR/graph.json" "$GRAPH_FILE"
    [ -f "$BACKUP_DIR/manifest.json" ] && cp "$BACKUP_DIR/manifest.json" "$MANIFEST_FILE"
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
    echo "  incident. Restoring graphify-out/graph.json and manifest.json to their" >&2
    echo "  pre-update state -- there is nothing to commit." >&2
    echo "  Do not re-run graphify update . directly and trust its own log output." >&2
    echo "  Investigate first, or use a full re-extraction instead of incremental update." >&2
    cp "$BACKUP_DIR/graph.json" "$GRAPH_FILE"
    [ -f "$BACKUP_DIR/manifest.json" ] && cp "$BACKUP_DIR/manifest.json" "$MANIFEST_FILE"
    rm -rf "$BACKUP_DIR"
    exit 1
fi

echo "[safe-graphify-update] OK: node count $BEFORE -> $AFTER (~${DROP_PCT}% change, within ${THRESHOLD_PCT}% threshold)."
rm -rf "$BACKUP_DIR"
exit 0
