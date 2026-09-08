#!/usr/bin/env python3
"""Refuse a commit that shrinks graphify-out/graph.json past a threshold.

Why this exists, and why it is a *commit* gate rather than another wrapper:

`graphify update .` can silently discard ~95% of the graph while its own log
output claims to have "kept" the old nodes. `scripts/safe_graphify_update.sh`
already guards the command -- it compares node counts and auto-restores -- and
it works. But it only works when someone chooses to call it, and the incident
that started all this (2026-07-14) got **committed and merged to main** before
anyone noticed. As of 2026-08-14 the destructive update still recurs several
times a day.

So this gate deliberately guards the *artifact*, not the command. It fires no
matter how the damage happened: a bare `graphify update`, a different tool, a
human's own terminal, an agent not running under Claude Code, or a wrapper
that was bypassed. Command-level guards are prevention; this is the backstop
that keeps a gutted graph out of history.

NOT a fix for the underlying graphify bug, which is still unroot-caused (see
CLAUDE.md's graphify section). This contains the damage.

Pure stdlib on purpose: `scripts/git_hooks/pre-commit` has a whole interpreter
fallback chain because some gates need pydantic and a bare system python3
usually lacks it. This one must run under any python3 so it never silently
skips.

Usage:
    python3 scripts/check_graph_node_loss.py            # git mode (the hook)
    python3 scripts/check_graph_node_loss.py --before a.json --after b.json
    python3 scripts/check_graph_node_loss.py --json

Exit codes: 0 = acceptable (grew, unchanged, or shrank within threshold, or
                nothing to compare against).
            1 = shrank past the threshold. BLOCK.
            2 = could not run the check (unreadable/unparseable input). Also
                blocking: a graph.json we cannot parse is not a graph.json
                worth committing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

GRAPH_PATH = "graphify-out/graph.json"
# Same default as safe_graphify_update.sh's GRAPHIFY_UPDATE_MAX_NODE_LOSS_PCT,
# and the same env-var shape, so the two guards cannot drift into disagreeing
# about what counts as destructive.
DEFAULT_MAX_LOSS_PCT = 10.0
ENV_THRESHOLD = "GRAPHIFY_COMMIT_MAX_NODE_LOSS_PCT"
ENV_ESCAPE = "ORION_ALLOW_GRAPH_SHRINK"

# networkx node-link format, as emitted by graphify: nodes + links (NOT
# "edges" -- confirmed against the live 43MB artifact, which has 28306 nodes,
# an empty "edges" key, and its real edges under "links"). Counting the wrong
# key would make this gate always read zero and never fire.
COUNTED_KEYS = ("nodes", "links", "hyperedges")


def _counts(doc: Any, *, source: str) -> dict[str, int]:
    if not isinstance(doc, dict):
        raise ValueError(f"{source}: expected a JSON object, got {type(doc).__name__}")
    out: dict[str, int] = {}
    for key in COUNTED_KEYS:
        value = doc.get(key)
        out[key] = len(value) if isinstance(value, list) else 0
    return out


def _load_path(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# graphify-out/graph.json became git-LFS-tracked in fix/graph-json-lfs
# (2026-09-08, GitHub's 100MB per-blob push cap). `git show <ref>:<path>`
# returns whatever git actually stores for that blob -- for an LFS-tracked
# path that is the ~130-byte pointer stub text (first line
# "version https://git-lfs.github.com/spec/v1"), NOT the real file content.
# LFS only smudges on checkout; `git show`/the index are raw blob reads, same
# as a merge driver receiving raw blobs (see scripts/graphify_lfs_merge_driver.sh).
# Without this, both HEAD's committed graph.json and the staged graph.json
# would read as ~130-byte pointer stubs the moment either side is a real LFS
# commit, and this gate would compare pointer-stub "graphs" (0 nodes either
# side) instead of real content.
_LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


def _is_lfs_pointer(raw: str) -> bool:
    return raw.startswith(_LFS_POINTER_PREFIX)


# This gate runs on every commit (via the pre-commit hook), so a hang here
# hangs every commit, not just ones touching graph.json. `git lfs smudge` can
# legitimately hit the network (fetching an object not yet in the local LFS
# cache -- exactly why smudge, not a local object-store read, is used).
# Override with GRAPHIFY_LFS_SMUDGE_TIMEOUT (seconds).
_LFS_SMUDGE_TIMEOUT = float(os.environ.get("GRAPHIFY_LFS_SMUDGE_TIMEOUT", "60"))


def _lfs_smudge(raw: str, path_hint: str) -> str:
    """Resolve LFS pointer-stub text to real file content via `git lfs smudge`.

    This fetches the object on demand if it is not already in the local LFS
    cache -- exactly what's needed for HEAD's version, which may not have
    been pulled yet in a shallow or partial LFS checkout.
    """
    # GIT_LFS_SKIP_SMUDGE unset explicitly: if set to 1 in the invoking
    # environment (a common CI/large-repo speed optimization), `git lfs
    # smudge` would pass the pointer text through unchanged instead of
    # fetching real content -- this gate needs real content unconditionally.
    env = dict(os.environ)
    env.pop("GIT_LFS_SKIP_SMUDGE", None)
    try:
        proc = subprocess.run(
            ["git", "lfs", "smudge", "--", path_hint],
            input=raw,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=_LFS_SMUDGE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"git lfs smudge timed out after {_LFS_SMUDGE_TIMEOUT}s for {path_hint}"
        ) from exc
    if proc.returncode != 0:
        raise ValueError(f"git lfs smudge failed for {path_hint}: {proc.stderr.strip()}")
    return proc.stdout


def _git_show(ref: str) -> str | None:
    """Blob contents at `ref`, smudged if it is an LFS pointer stub, or None
    if it does not exist there."""
    proc = subprocess.run(
        ["git", "show", ref],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    raw = proc.stdout
    if _is_lfs_pointer(raw):
        raw = _lfs_smudge(raw, GRAPH_PATH)
    return raw


def _staged_paths() -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def loss_pct(before: int, after: int) -> float:
    """Percent of nodes lost. 0.0 when before is 0 (nothing to lose) or when
    the graph grew -- a gate that fired on growth would be noise."""
    if before <= 0:
        return 0.0
    if after >= before:
        return 0.0
    return (before - after) * 100.0 / before


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", help="baseline graph.json (default: HEAD's version via git)")
    parser.add_argument("--after", help="candidate graph.json (default: the staged version via git)")
    parser.add_argument("--threshold", type=float, default=None, help=f"max %% node loss (default {DEFAULT_MAX_LOSS_PCT})")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    threshold = args.threshold
    if threshold is None:
        raw = os.environ.get(ENV_THRESHOLD, "").strip()
        try:
            threshold = float(raw) if raw else DEFAULT_MAX_LOSS_PCT
        except ValueError:
            print(
                f"check_graph_node_loss: {ENV_THRESHOLD}={raw!r} is not a number; "
                f"using default {DEFAULT_MAX_LOSS_PCT}",
                file=sys.stderr,
            )
            threshold = DEFAULT_MAX_LOSS_PCT

    explicit = args.before is not None or args.after is not None
    try:
        if explicit:
            if not (args.before and args.after):
                print("check_graph_node_loss: --before and --after must be given together", file=sys.stderr)
                return 2
            before_doc = _load_path(args.before)
            after_doc = _load_path(args.after)
        else:
            if GRAPH_PATH not in _staged_paths():
                # Nothing to check. Silent: this runs on every commit.
                return 0
            head_raw = _git_show(f"HEAD:{GRAPH_PATH}")
            if head_raw is None:
                print(f"check_graph_node_loss: {GRAPH_PATH} is new in this commit -- nothing to compare. OK.")
                return 0
            staged_raw = _git_show(f":{GRAPH_PATH}")
            if staged_raw is None:
                print(f"check_graph_node_loss: could not read staged {GRAPH_PATH}", file=sys.stderr)
                return 2
            before_doc = json.loads(head_raw)
            after_doc = json.loads(staged_raw)

        before = _counts(before_doc, source="before")
        after = _counts(after_doc, source="after")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"check_graph_node_loss: cannot compare graphs: {exc}", file=sys.stderr)
        return 2

    pct = loss_pct(before["nodes"], after["nodes"])
    blocked = pct > threshold

    if args.json:
        print(json.dumps({
            "before": before, "after": after,
            "node_loss_pct": round(pct, 4),
            "threshold_pct": threshold,
            "blocked": blocked,
        }, indent=2))
    else:
        detail = " ".join(f"{k}={before[k]}->{after[k]}" for k in COUNTED_KEYS)
        verdict = "BLOCK" if blocked else "OK"
        print(f"check_graph_node_loss: {verdict} node_loss={pct:.2f}% (threshold {threshold:.2f}%) {detail}")

    if not blocked:
        return 0

    if os.environ.get(ENV_ESCAPE) == "1":
        print(
            f"check_graph_node_loss: {pct:.2f}% node loss ALLOWED via {ENV_ESCAPE}=1 "
            "(intentional re-extraction)",
            file=sys.stderr,
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
