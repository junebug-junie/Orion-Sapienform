#!/usr/bin/env python3
"""Live eval of `dig attention` (issue #3600). Read-only.

Reads the window the way `python -P -m orion.dig attention` does (ReadOnlyPsql, READ ONLY transaction) and checks:

  E1  it reads in under 5 s (rollup coverage + rollup counts + events; it never reads frame_json);
  E2  every printed line is a rate with its numerator and denominator, or a ref (attention.bare_counts);
  E3  it is not empty-shell: frames were read, at least one target was eligible, and the rest share is a real
      rate (0 < rest < 1: a pinned 0 or 1 is a broken contest, not calm);
  E4  the rollup's last whole hour equals an independent SQL aggregate of that hour's raw frame_json (the
      PR #3654 semantics: frames, rest, contested, and per target eligible and ranked first). About 1,700
      frames, read once. Skipped when that hour is the rollup's partial first hour or older than the
      frames' 72 h retention;
  E5  each event source's event count equals a plain count of its temporal_self_event rows;
  E6  at most 1% of not-attended frames fall in the "other" cause (a new verdict shape classify() misses);
  E7  each event source's lit seconds (SQL, overlap-merged) equal lit_seconds() over the raw event times.

    PYTHONPATH=. python orion/dig/evals/run_attention_live_eval.py [--window 24h] [--end <ISO>] [--role orion_readonly]

``--role``: the read-only role (default orion_readonly, Orion's own; ``postgres`` still runs in a READ ONLY
transaction, for use before the grant lands). Exit 0 = every check passed; 1 = a check failed; 2 = the database
could not be read.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.body_atlas.sources import psql_rows  # noqa: E402
from orion.dig import attention as A  # noqa: E402
from orion.evals.model_replay.sandbox import ReadOnlyPsql  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default=A.DEFAULT_WINDOW)
    ap.add_argument("--end", default=None)
    ap.add_argument("--role", default="orion_readonly")
    ns = ap.parse_args(argv)
    run = ReadOnlyPsql(role=ns.role, timeout_sec=30.0)
    q = lambda sql: psql_rows(run, sql)  # noqa: E731
    w = A.parse_window(ns.window, ns.end)
    t0 = time.monotonic()
    try:
        r = A.read(w, q)
    except RuntimeError as exc:
        print(f"UNREAD: {exc}")
        return 2
    r.took_sec = took = time.monotonic() - t0
    lines = A.render(r)
    print("\n".join(lines))
    fails = []
    if took >= 5:
        fails.append(f"E1 took {took:.1f} s")
    bad = [ln for ln in lines if A.bare_counts(ln)]
    if bad:
        fails.append(f"E2 bare counts on {len(bad)} line(s): {bad[0]!r}")
    c = r.cur
    if not c.frames or not c.present or not (0 < c.rest < c.frames):
        fails.append(f"E3 empty shell: frames={c.frames} eligible targets={len(c.present)} rest={c.rest}")
    hour_end = A.parse_end(r.end)
    hour = A._ts(hour_end - timedelta(hours=1))
    origin = q("SELECT min(hour) AS h FROM substrate_attention_hour_rollup")[0]["h"]
    if origin and A.parse_end(origin) < hour_end - timedelta(hours=1):
        rolled = {(x["part"], x["target"]): (int(x["a"]), int(x["b"]), int(x["c"])) for x in q(
            f"""SELECT part, target, sum(a) AS a, sum(b) AS b, sum(c) AS c FROM substrate_attention_hour_rollup
                 WHERE hour = '{hour}' AND part IN ('frame', 'won') GROUP BY 1, 2""")}
        wf = f"'{A.WORLD_FIRST_PREFIX}%'"
        raw = {(x["part"], x["target"]): (int(x["a"]), int(x["b"]), int(x["c"])) for x in q(f"""
SET LOCAL statement_timeout = '25s';
WITH f AS (SELECT frame_json->'dominant_targets' AS d,
                  (SELECT count(*) FROM jsonb_array_elements(frame_json->'dominant_targets') e
                    WHERE e->'reasons'->>0 LIKE {wf}) AS n
             FROM substrate_attention_frames WHERE generated_at >= '{hour}' AND generated_at < '{r.end}')
SELECT 'frame' AS part, '' AS target, count(*) AS a, count(*) FILTER (WHERE n = 0) AS b,
       count(*) FILTER (WHERE n > 1) AS c FROM f
UNION ALL
SELECT 'won', t->>'target_id', count(*), count(*) FILTER (WHERE o = 1), count(*) FILTER (WHERE f.n > 1)
  FROM f, jsonb_array_elements(f.d) WITH ORDINALITY AS x(t, o) WHERE t->'reasons'->>0 LIKE {wf} GROUP BY 2""")}
        if raw.get(("frame", ""), (0,))[0] and rolled != raw:
            diff = {k: (rolled.get(k), raw.get(k)) for k in set(rolled) | set(raw) if rolled.get(k) != raw.get(k)}
            fails.append(f"E4 hour {hour}: rollup differs from raw frames: {dict(list(diff.items())[:3])}")
    for target, src in A.EVENT_SOURCES.items():
        cond = f"source_kind = '{src.kind}'"
        if src.payload_key:
            cond += f" AND payload_json->>'{src.payload_key}' = '{src.payload_value}'"
        n = int(q(f"SELECT count(*) AS n FROM temporal_self_event WHERE {cond} AND occurred_at >= '{r.start}' "
                  f"AND occurred_at < '{r.end}'")[0]["n"])
        if n != c.events.get(src.kind, 0):
            fails.append(f"E5 {target}: {c.events.get(src.kind, 0)} events != count {n}")
    judged = sum(sum(v.values()) for v in c.causes.values())
    other = sum(v.get(A.OTHER_CAUSE, 0) for v in c.causes.values())
    if not judged or other > 0.01 * judged:
        fails.append(f"E6 'other' cause on {other} of {judged} not-attended frames")
    start, end = A.parse_end(r.start), A.parse_end(r.end)
    for target, src in A.EVENT_SOURCES.items():
        cond = f"source_kind = '{src.kind}'"
        if src.payload_key:
            cond += f" AND payload_json->>'{src.payload_key}' = '{src.payload_value}'"
        lo = start - timedelta(seconds=src.window_sec)
        times = [A.parse_end(x["t"]) for x in q(f"SELECT occurred_at AS t FROM temporal_self_event WHERE {cond} "
                                                 f"AND occurred_at >= '{A._ts(lo)}' AND occurred_at < '{r.end}'")]
        want = A.lit_seconds(times, src.window_sec, start, end)
        if abs(want - c.lit.get(src.kind, 0.0)) > 1.0:
            fails.append(f"E7 {target}: lit {c.lit.get(src.kind, 0.0):.0f} s != reference {want:.0f} s")
    print(f"\neval: read in {took:.1f} s; " + ("PASS (E1-E7)" if not fails else "FAIL: " + "; ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
