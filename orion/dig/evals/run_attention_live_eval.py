#!/usr/bin/env python3
"""Live eval of `dig attention` (issue #3600). Read-only.

Reads the window the way `python -P -m orion.dig attention` does (ReadOnlyPsql, READ ONLY transaction) and checks:

  E1  it reads in under 30 s (coverage + one frame pass + events);
  E2  every printed line is a rate with its numerator and denominator, or a ref (attention.bare_counts);
  E3  it is not empty-shell: frames were read, at least one target was eligible, and the rest share is a real
      rate (0 < rest < 1: a pinned 0 or 1 is a broken contest, not calm);
  E4  the frame total equals a plain count(*) of the same window (the sampled pass covers every frame when k = 1);
  E5  each event source's event count equals a plain count of its temporal_self_event rows;
  E6  the SQL cause CASE and Python classify() agree on up to 2,000 real not-attended reasons of the window;
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
    if took >= 30:
        fails.append(f"E1 took {took:.1f} s")
    bad = [ln for ln in lines if A.bare_counts(ln)]
    if bad:
        fails.append(f"E2 bare counts on {len(bad)} line(s): {bad[0]!r}")
    c = r.cur
    if not c.frames or not c.present or not (0 < c.rest < c.frames):
        fails.append(f"E3 empty shell: frames={c.frames} eligible targets={len(c.present)} rest={c.rest}")
    if c.k == 1:
        n = int(q(f"SELECT count(*) AS n FROM substrate_attention_frames WHERE generated_at >= '{r.start}' "
                  f"AND generated_at < '{r.end}'")[0]["n"])
        if n != c.frames:
            fails.append(f"E4 frames {c.frames} != count(*) {n}")
    for target, src in A.EVENT_SOURCES.items():
        cond = f"source_kind = '{src.kind}'"
        if src.payload_key:
            cond += f" AND payload_json->>'{src.payload_key}' = '{src.payload_value}'"
        n = int(q(f"SELECT count(*) AS n FROM temporal_self_event WHERE {cond} AND occurred_at >= '{r.start}' "
                  f"AND occurred_at < '{r.end}'")[0]["n"])
        if n != c.events.get(src.kind, 0):
            fails.append(f"E5 {target}: {c.events.get(src.kind, 0)} events != count {n}")
    reasons = q(f"""SELECT t->'reasons'->>0 AS reason, {A._cause_case("t->'reasons'->>0")} AS cause
  FROM (SELECT frame_json FROM substrate_attention_frames WHERE generated_at >= '{r.start}' AND generated_at < '{r.end}'
        ORDER BY generated_at DESC LIMIT 200) f, jsonb_array_elements(f.frame_json->'suppressed_targets') t
 WHERE t->'reasons'->>0 LIKE '{A.WORLD_FIRST_PREFIX}%' LIMIT 2000""")
    differ = [x for x in reasons if A.classify(x["reason"]) != x["cause"]]
    if not reasons or differ:
        fails.append(f"E6 {len(differ)} of {len(reasons)} reasons classified differently: {differ[:1]}")
    from datetime import timedelta
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
