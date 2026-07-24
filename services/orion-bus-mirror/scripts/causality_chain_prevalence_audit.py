#!/usr/bin/env python3
"""Read-only audit: how often does a real envelope populate ``causality_chain``
vs. relying purely on ``correlation_id``?

Idea G of docs/superpowers/specs/2026-07-24-bus-synaptic-graph-wider-net-brainstorm.md.
Resolves a Missing Question carried since the parent brainstorm doc: whether
``CAUSALLY_FOLLOWED_BY`` edges (built from ``correlation_id`` co-occurrence,
see services/orion-bus-mirror/app/graph_writer.py) need a confidence property
distinguishing weak co-occurrence-only evidence from the stronger, explicit
``causality_chain`` signal (populated only via ``BaseEnvelope.derive_child()``).

Reads the live, actively-growing MIRROR_SQLITE_PATH database in read-only mode
(``mode=ro``) -- never writes, never touches the graph. Samples via direct
rowid seeks at evenly-spaced checkpoints across the full table range (cheap:
each checkpoint is an indexed seek + small LIMIT, not a table scan) rather
than ``COUNT(*)``/``MAX(rowid)`` aggregates, which were observed live to hang
against this table (root cause not investigated -- not blocking for a
one-off read-only audit, worth knowing if this script is reused elsewhere).

Usage (run inside the orion-bus-mirror container, where the sqlite file
actually lives -- MIRROR_SQLITE_PATH is a container-local volume path):

    docker exec orion-athena-bus-mirror python3 \
        services/orion-bus-mirror/scripts/causality_chain_prevalence_audit.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import Counter


def _first_and_last_rowid(cur: sqlite3.Cursor) -> tuple[int, int]:
    cur.execute("SELECT rowid FROM bus_events ORDER BY rowid ASC LIMIT 1")
    first = cur.fetchone()
    cur.execute("SELECT rowid FROM bus_events ORDER BY rowid DESC LIMIT 1")
    last = cur.fetchone()
    if not first or not last:
        raise RuntimeError("bus_events table is empty")
    return int(first[0]), int(last[0])


def audit(
    sqlite_path: str,
    *,
    checkpoints: int = 12,
    rows_per_checkpoint: int = 300,
) -> dict:
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True, timeout=5)
    cur = conn.cursor()
    min_rowid, max_rowid = _first_and_last_rowid(cur)
    span = max_rowid - min_rowid

    total = 0
    decode_fail = 0
    has_causality_chain = 0
    has_correlation_id = 0
    kind_with_chain: Counter[str] = Counter()

    # Stride-based checkpoints can land short of max_rowid by up to
    # span/checkpoints due to integer rounding -- always add an explicit
    # tail checkpoint anchored to the true end of the table so the most
    # recent traffic is never a sampling blind spot (found while testing:
    # a naive stride-only sample missed the last row entirely).
    checkpoint_starts = {min_rowid + int(span * i / checkpoints) for i in range(checkpoints)}
    checkpoint_starts.add(max(max_rowid - rows_per_checkpoint + 1, min_rowid))

    for start in sorted(checkpoint_starts):
        cur.execute(
            "SELECT envelope_json FROM bus_events WHERE rowid >= ? ORDER BY rowid ASC LIMIT ?",
            (start, rows_per_checkpoint),
        )
        for (envelope_json,) in cur.fetchall():
            total += 1
            try:
                env = json.loads(envelope_json)
            except Exception:
                decode_fail += 1
                continue
            if env.get("causality_chain"):
                has_causality_chain += 1
                kind_with_chain[env.get("kind", "?")] += 1
            if env.get("correlation_id"):
                has_correlation_id += 1

    return {
        "rowid_range": (min_rowid, max_rowid),
        "total_sampled": total,
        "decode_fail": decode_fail,
        "has_causality_chain": has_causality_chain,
        "has_correlation_id": has_correlation_id,
        "kinds_with_chain": kind_with_chain.most_common(15),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite-path",
        default=os.getenv("MIRROR_SQLITE_PATH", "/data/bus_mirror.sqlite"),
    )
    parser.add_argument("--checkpoints", type=int, default=12)
    parser.add_argument("--rows-per-checkpoint", type=int, default=300)
    args = parser.parse_args()

    t0 = time.time()
    result = audit(
        args.sqlite_path,
        checkpoints=args.checkpoints,
        rows_per_checkpoint=args.rows_per_checkpoint,
    )
    elapsed = time.time() - t0

    total = result["total_sampled"]
    print(f"sqlite_path: {args.sqlite_path}")
    print(f"rowid_range: {result['rowid_range']}")
    print(f"total_sampled: {total}  (elapsed {elapsed:.2f}s)")
    print(f"decode_fail: {result['decode_fail']}")
    pct_chain = 100 * result["has_causality_chain"] / max(total, 1)
    pct_corr = 100 * result["has_correlation_id"] / max(total, 1)
    print(f"has_causality_chain: {result['has_causality_chain']} ({pct_chain:.2f}%)")
    print(f"has_correlation_id: {result['has_correlation_id']} ({pct_corr:.2f}%)")
    print("kinds with populated causality_chain (top 15):")
    for kind, count in result["kinds_with_chain"]:
        print(f"  {kind}: {count}")


if __name__ == "__main__":
    main()
