#!/usr/bin/env python3
"""Read-only: does a personality node show up in particular situations more than chance?

## Why this exists

Handoff step 4 (docs/superpowers/specs/2026-10-20-personality-graph-handoff.md): before the
reducer grows CAPS if-then subnodes ("in situation S, theme T"), prove that a node's situation
mix differs from what its lanes would give anyway. Every evidence row carries
``meta.situation`` (arousal, day_phase, company) stamped at its own time (#2739).

## How it measures

- Unit: an EPISODE, not a row. A node's rows in one stream (the lane) within ``--burst-min`` of
  each other are one episode with its first row's situation (reverie writes several thoughts on
  one theme within minutes, all in one situation).
- Statistic, per node and dimension: total variation distance (TVD) between the node's
  situation counts and its EXPECTED counts, where each episode contributes the situation mix of
  all episodes in its stratum. Stratum = (lane, UTC date): lanes run in different situations,
  and a theme seen only on a strained day is not a strained-day theme.
- Null: shuffle situation values among episodes within each stratum (``--permutations``), the
  p-value is the share of shuffles with TVD >= observed (+1 smoothing). Benjamini-Hochberg q
  over every (node, dimension) tested.

A node needs ``--min-episodes`` situated episodes in strata that actually vary; otherwise it is
reported as untestable, never as "no effect".

## What it reads and writes

``personality_evidence`` joined to ``personality_disposition`` (read-only session). Writes nothing
to any DB; prints the table and writes ``summary.txt`` to ``--out-dir``.

    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
        python3 scripts/analysis/measure_personality_situation_specificity.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DIMENSIONS = ("arousal", "day_phase", "company")

SQL = """
SELECT e.disposition_id, d.status, d.label, e.lane, e.meta, e.occurred_at
FROM personality_evidence e JOIN personality_disposition d USING (disposition_id)
WHERE e.disposition_id IS NOT NULL AND e.meta ? 'situation'
"""


def _dsn() -> str:
    for key in ("POSTGRES_URI", "DATABASE_URL", "ORION_SQL_DSN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return "postgresql://postgres:postgres@localhost:55432/conjourney"


# ------------------------------------------------------------------ pure


def _situation(raw: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    sit = raw.get("situation") if isinstance(raw, Mapping) else None
    return sit if isinstance(sit, Mapping) else None


def episodes(rows: Iterable[Mapping[str, Any]], *, burst: timedelta) -> List[Dict[str, Any]]:
    """Collapse each node's same-lane rows within ``burst`` of the previous row into one episode."""
    by_key: Dict[Tuple[str, str], List[Tuple[datetime, Mapping[str, Any]]]] = defaultdict(list)
    for r in rows:
        sit = _situation(r.get("meta"))
        if sit is not None:
            by_key[(str(r["disposition_id"]), str(r["lane"]))].append((r["occurred_at"], sit))
    out = []
    for (node, lane), hits in by_key.items():
        hits.sort(key=lambda h: h[0])
        last = None
        for at, sit in hits:
            if last is None or at - last > burst:
                out.append({"node": node, "lane": lane, "at": at,
                            "stratum": (lane, at.date().isoformat()),
                            **{d: str(sit.get(d)) if sit.get(d) is not None else None for d in DIMENSIONS}})
            last = at
    return out


def _tvd(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b)
    na, nb = sum(a.values()), sum(b.values())
    if na <= 0 or nb <= 0:
        return 0.0
    return 0.5 * sum(abs(a.get(k, 0.0) / na - b.get(k, 0.0) / nb) for k in keys)


def _observed_vs_expected(eps: Sequence[Mapping[str, Any]], values: Sequence[Optional[str]], dim_mix):
    obs: Counter = Counter()
    exp: Dict[str, float] = defaultdict(float)
    for e, v in zip(eps, values):
        obs[v] += 1
        for k, share in dim_mix[e["stratum"]].items():
            exp[k] += share
    return obs, exp


def specificity(eps: Sequence[Mapping[str, Any]], *, min_episodes: int, permutations: int,
                seed: int = 0) -> List[Dict[str, Any]]:
    """One row per (node, dimension): observed vs lane-and-day expected mix, TVD, p, BH q."""
    rng = random.Random(seed)
    results = []
    for dim in DIMENSIONS:
        situated = [e for e in eps if e[dim] is not None]
        strata: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for i, e in enumerate(situated):
            strata[e["stratum"]].append(i)
        mix = {}
        for s, idx in strata.items():
            c = Counter(situated[i][dim] for i in idx)
            mix[s] = {k: v / len(idx) for k, v in c.items()}
        varying = {s for s, m in mix.items() if len(m) > 1}
        by_node: Dict[str, List[int]] = defaultdict(list)
        for i, e in enumerate(situated):
            by_node[e["node"]].append(i)
        values = [e[dim] for e in situated]
        for node, idx in by_node.items():
            informative = [i for i in idx if situated[i]["stratum"] in varying]
            row = {"node": node, "dimension": dim, "episodes": len(idx), "informative": len(informative)}
            if len(informative) < min_episodes:
                row.update(testable=False)
                results.append(row)
                continue
            node_eps = [situated[i] for i in informative]
            obs, exp = _observed_vs_expected(node_eps, [values[i] for i in informative], mix)
            stat = _tvd(obs, exp)
            hits = 0
            for _ in range(permutations):
                shuffled = list(values)
                for s in varying:
                    pool = [shuffled[i] for i in strata[s]]
                    rng.shuffle(pool)
                    for i, v in zip(strata[s], pool):
                        shuffled[i] = v
                o, _e = _observed_vs_expected(node_eps, [shuffled[i] for i in informative], mix)
                hits += _tvd(o, exp) >= stat - 1e-12
            row.update(testable=True, tvd=stat, p=(hits + 1) / (permutations + 1),
                       observed=dict(obs), expected={k: round(v, 2) for k, v in exp.items()},
                       over=max(obs, key=lambda k: obs[k] - exp.get(k, 0.0)))
            results.append(row)
    tested = sorted((r for r in results if r["testable"]), key=lambda r: r["p"])
    m = len(tested)
    q_prev = 1.0
    for rank in range(m, 0, -1):  # Benjamini-Hochberg, monotone from the top
        r = tested[rank - 1]
        q_prev = min(q_prev, r["p"] * m / rank)
        r["q"] = q_prev
    return results


def render(results: Sequence[Mapping[str, Any]], labels: Mapping[str, Tuple[str, str]], *,
           alpha: float) -> str:
    tested = sorted((r for r in results if r["testable"]), key=lambda r: (r["q"], r["p"]))
    lines = [f"tested {len(tested)} (node, dimension) pairs; untestable "
             f"{sum(1 for r in results if not r['testable'])} (too few episodes in strata that vary)",
             f"{'q':>6} {'p':>6} {'tvd':>5} {'eps':>4} {'dim':10} {'over':10} status     label"]
    for r in tested:
        status, label = labels.get(r["node"], ("?", r["node"]))
        lines.append(f"{r['q']:6.3f} {r['p']:6.3f} {r['tvd']:5.2f} {r['informative']:4d} {r['dimension']:10} "
                     f"{str(r['over']):10} {status:10} {label[:60]}")
        if r["q"] <= alpha:
            lines.append(f"{'':37}observed {r['observed']} vs expected {r['expected']}")
    sig = sum(1 for r in tested if r["q"] <= alpha)
    lines.append(f"significant at q <= {alpha}: {sig} of {len(tested)}")
    return "\n".join(lines) + "\n"


def _fetch(conn, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--burst-min", type=float, default=30.0)
    ap.add_argument("--min-episodes", type=int, default=8)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out-dir", default="/tmp/personality-situation-specificity/")
    args = ap.parse_args(argv)

    _repo = Path(__file__).resolve().parents[2]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from orion.db_readonly import open_readonly_connection

    conn = open_readonly_connection(_dsn(), connect_timeout=10, statement_timeout_ms=60_000)
    if conn is None:
        print("UNKNOWN: could not open a confirmed read-only DB session", file=sys.stderr)
        return 2
    try:
        rows = _fetch(conn, SQL)
    finally:
        conn.close()
    labels = {str(r["disposition_id"]): (str(r["status"]), str(r["label"])) for r in rows}
    eps = episodes(rows, burst=timedelta(minutes=args.burst_min))
    results = specificity(eps, min_episodes=args.min_episodes, permutations=args.permutations)
    summary = render(results, labels, alpha=args.alpha)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(summary)
    print(summary, end="")
    print(f"wrote {out / 'summary.txt'} ({len(rows)} rows, {len(eps)} episodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
