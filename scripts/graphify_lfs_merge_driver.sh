#!/bin/sh
# graphify_lfs_merge_driver.sh -- LFS-aware wrapper around `graphify merge-driver`.
#
# Why this exists: git feeds a merge driver the RAW blob content of %O/%A/%B.
# For an LFS-tracked path that is always the ~130-byte pointer stub text
# (`version https://git-lfs.github.com/spec/v1\noid sha256:...`), never the
# smudged real file -- git only smudges LFS content on checkout, it does not
# smudge blobs handed to a merge driver. Confirmed by direct repro in a
# throwaway repo on 2026-09-08. Without this wrapper, once graphify-out/graph.json
# is LFS-tracked, `graphify merge-driver` would union-merge three pointer
# stubs instead of real graph JSON -- silently producing garbage or a bogus
# "clean" merge the instant two branches both touch the graph.
#
# Registered as merge.graphify.driver by scripts/setup_graphify_merge_driver.sh.
# Git invokes it as:  graphify_lfs_merge_driver.sh %O %A %B
#   %O = common ancestor (base)
#   %A = current branch's version -- git reads THIS PATH back as the result
#   %B = other branch's version being merged in
#
# For each of the three paths: if it looks like an LFS pointer stub, resolve
# it to real content via `git lfs smudge` (this fetches the object on demand
# if it is not already in the local LFS cache -- this is exactly why smudge,
# not a local object-store read, is used: it works for %O/%B even when their
# objects have not been fetched yet). If a path is not a pointer (plain-JSON
# history from before this migration, or a checkout with LFS uninstalled),
# it is used as-is.
#
# `graphify merge-driver` only actually reads %A/%B (not %O, confirmed against
# the installed package: cli.py's merge-driver branch loads "_current_path"
# and "_other_path" only) and writes its merged result onto the %A path it
# was given. Because we hand it *temporary, resolved* paths rather than the
# originals, its result lands in a temp file -- that result is copied back
# onto the ORIGINAL %A path here, since that's what git reads back as the
# merge result.
#
# POSIX sh only -- no bashisms.

set -eu

if [ $# -ne 3 ]; then
    echo "graphify_lfs_merge_driver: usage: $0 %O %A %B" >&2
    exit 2
fi

ORIG_O=$1
ORIG_A=$2
ORIG_B=$3

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT INT TERM

_is_lfs_pointer() {
    # A real LFS pointer file's first line is exactly this. Plain JSON always
    # starts with '{' (or whitespace before it), never this string.
    head -c 60 "$1" 2>/dev/null | grep -q '^version https://git-lfs\.github\.com/spec/v1'
}

# Resolves $1 (an original %O/%A/%B path) into $2 (a fresh temp path holding
# real content). Leaves $2 absent if $1 does not exist (e.g. the file was
# added on only one side of the merge) -- callers fall back to the original
# path in that case, matching the no-LFS behavior exactly.
_resolve() {
    _src=$1
    _dst=$2
    if [ ! -f "$_src" ]; then
        return 0
    fi
    if _is_lfs_pointer "$_src"; then
        if ! git lfs smudge -- "$_src" < "$_src" > "$_dst" 2>"$WORKDIR/smudge.err"; then
            echo "graphify_lfs_merge_driver: git lfs smudge failed for $_src:" >&2
            cat "$WORKDIR/smudge.err" >&2
            exit 1
        fi
    else
        cp "$_src" "$_dst"
    fi
}

RESOLVED_O="$WORKDIR/O.json"
RESOLVED_A="$WORKDIR/A.json"
RESOLVED_B="$WORKDIR/B.json"

_resolve "$ORIG_O" "$RESOLVED_O"
_resolve "$ORIG_A" "$RESOLVED_A"
_resolve "$ORIG_B" "$RESOLVED_B"

[ -f "$RESOLVED_O" ] || RESOLVED_O=$ORIG_O
[ -f "$RESOLVED_A" ] || RESOLVED_A=$ORIG_A
[ -f "$RESOLVED_B" ] || RESOLVED_B=$ORIG_B

set +e
graphify merge-driver "$RESOLVED_O" "$RESOLVED_A" "$RESOLVED_B"
STATUS=$?
set -e

if [ "$STATUS" -eq 0 ] && [ -f "$RESOLVED_A" ]; then
    cp "$RESOLVED_A" "$ORIG_A"
fi

exit "$STATUS"
