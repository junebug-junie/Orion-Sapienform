#!/usr/bin/env python3
"""Read-only: do two evidence streams on the same personality node fire together?

## Why this exists

Promotion in ``orion/personality/graph.py`` counts distinct SOURCES (``source_key``). Two lanes
that narrate the same upstream signal must share a source, or one signal counts twice. On
2026-10-20 this measurement showed reverie and the metacog journal were one source (16 of 27
metacog hits on the 3 active prediction-error nodes had a reverie hit on the same node within
10 minutes; 1 of 27 at a 6 h or 3 h offset). Re-run it when a lane or journal kind is added.

## How it measures

For each node and each ordered pair of streams (a, b) on it, a stream being the lane, or
``journal:<source_kind>`` for journal: of a's hits, how many have a b hit on the same node within
``--window-min``, against the same count with a's times shifted by each ``--shift-hours`` (the
baseline: the same streams, same node, no shared moment). ``lift`` = near rate / baseline rate.
A high lift with enough hits means the pair is driven by one cause. ``same_source`` says whether
``source_key`` already merges the pair.

## What it reads and writes

``personality_evidence`` rows with a node (read-only session). Writes nothing to any DB; prints
the table and writes ``summary.txt`` to ``--out-dir``.

    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
        python3 scripts/analysis/measure_personality_lane_independence.py
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orion.personality.graph import source_key  # noqa: E402

SQL = """
SELECT disposition_id, lane, meta, occurred_at FROM personality_evidence
WHERE disposition_id IS NOT NULL AND occurred_at >= now() - make_interval(days => %s)
"""


def _dsn() -> str:
    for key in ("POSTGRES_URI", "DATABASE_URL", "ORION_SQL_DSN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return "postgresql://postgres:postgres@localhost:55432/conjourney"


# ------------------------------------------------------------------ pure


def _meta(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, Mapping) else {}


def stream(lane: str, meta: Mapping[str, Any]) -> str:
    kind = meta.get("source_kind") if lane == "journal" else None
    return f"journal:{kind}" if kind else lane


def _any_within(times: Sequence[datetime], t: datetime, window: timedelta) -> bool:
    i = bisect.bisect_left(times, t - window)
    return i < len(times) and times[i] <= t + window


def cooccurrence(rows: Iterable[Mapping[str, Any]], *, window: timedelta,
                 shifts: Sequence[timedelta]) -> List[Dict[str, Any]]:
    """One row per ordered stream pair (a, b), summed over nodes."""
    by_node: Dict[str, Dict[str, List[datetime]]] = {}
    keys: Dict[str, str] = {}
    for r in rows:
        meta = _meta(r.get("meta"))
        s = stream(str(r["lane"]), meta)
        keys[s] = source_key(str(r["lane"]), meta)
        by_node.setdefault(str(r["disposition_id"]), {}).setdefault(s, []).append(r["occurred_at"])
    pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for streams in by_node.values():
        for times in streams.values():
            times.sort()
        for a, ta in streams.items():
            for b, tb in streams.items():
                if a == b:
                    continue
                p = pairs.setdefault((a, b), {"a": a, "b": b, "nodes": 0, "n": 0, "near": 0,
                                              "shifted": [0] * len(shifts)})
                p["nodes"] += 1
                p["n"] += len(ta)
                p["near"] += sum(_any_within(tb, t, window) for t in ta)
                for i, sh in enumerate(shifts):
                    p["shifted"][i] += sum(_any_within(tb, t + sh, window) for t in ta)
    out = []
    for p in pairs.values():
        base = max(max(p["shifted"], default=0), 1) / p["n"]  # floor: one hit, so lift stays finite
        p["near_rate"] = p["near"] / p["n"]
        p["lift"] = p["near_rate"] / base
        p["same_source"] = keys[p["a"]] == keys[p["b"]]
        out.append(p)
    return sorted(out, key=lambda p: (-p["lift"], -p["n"], p["a"], p["b"]))


def render(pairs: Sequence[Mapping[str, Any]], *, min_hits: int, window_min: float,
           shift_hours: Sequence[float]) -> str:
    lines = [f"stream pairs with >= {min_hits} hits; near = b within {window_min:g} min of a; "
             f"baseline = a shifted by {', '.join(f'{h:+g}h' for h in shift_hours)}",
             f"{'a':24} {'b':24} {'nodes':>5} {'n':>5} {'near':>5} {'base':>9} {'lift':>6}  same_source"]
    for p in pairs:
        if p["n"] < min_hits:
            continue
        lines.append(f"{p['a']:24} {p['b']:24} {p['nodes']:5d} {p['n']:5d} {p['near']:5d} "
                     f"{'/'.join(map(str, p['shifted'])):>9} {p['lift']:6.1f}  {p['same_source']}")
    return "\n".join(lines) + "\n"


def _fetch(conn, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--window-min", type=float, default=10.0)
    ap.add_argument("--shift-hours", type=float, nargs="+", default=[6.0, -3.0])
    ap.add_argument("--min-hits", type=int, default=5)
    ap.add_argument("--out-dir", default="/tmp/personality-lane-independence/")
    args = ap.parse_args(argv)

    from orion.db_readonly import open_readonly_connection

    conn = open_readonly_connection(_dsn(), connect_timeout=10, statement_timeout_ms=60_000)
    if conn is None:
        print("UNKNOWN: could not open a confirmed read-only DB session", file=sys.stderr)
        return 2
    try:
        rows = _fetch(conn, SQL, (args.window_days,))
    finally:
        conn.close()
    pairs = cooccurrence(rows, window=timedelta(minutes=args.window_min),
                         shifts=[timedelta(hours=h) for h in args.shift_hours])
    summary = render(pairs, min_hits=args.min_hits, window_min=args.window_min, shift_hours=args.shift_hours)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(summary)
    print(summary, end="")
    print(f"wrote {out / 'summary.txt'} ({len(rows)} evidence rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
