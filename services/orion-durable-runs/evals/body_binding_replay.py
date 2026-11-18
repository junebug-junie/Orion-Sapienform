#!/usr/bin/env python3
"""Eval: what the felt-state producer writes, and the binding arcs it folds into. READ-ONLY.

The producer is the physiology node's ``_binding_step`` (``app.temporal_self_graph``, rules in
``app.body_binding``); the arcs come from the real chronicle adapter and reducer
(``orion.temporal_self``). Both run here exactly as shipped, with a fake writer that keeps rows.

* ``--fixture`` (default; deterministic, no database): scripted bind-state streams. Gates: a single
  bound reading or a bound/slack flap never opens an arc; three bound readings (240 s) open one; a
  no_reading stretch never releases it and the stale timeout closes it at its last row; a release
  ends it at its first slack reading; two constraints bound together give two open arcs and the
  stance cue names both with their citations.
* ``--live``: the last ``--days`` of REAL ``physiology_state`` rows (``docker exec <container> psql``,
  read-only), replayed tick by tick. Prints rows by kind and arcs per constraint, the bound share
  and naive onsets (a bound tick after a non-bound one) for comparison, and the stance cue clause
  at the end. Gates: every arc has evidence and ends at or after it began; rows within one episode
  are at least ``BINDING_CHANGE_MIN_SEC`` (600 s) apart; no constraint writes more than ``--max-rows-per-day`` rows a day.

    python -P services/orion-durable-runs/evals/body_binding_replay.py [--live --days 7]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

SVC = Path(__file__).resolve().parents[1]
ROOT = SVC.parents[1]
sys.path[:0] = [str(ROOT), str(SVC)]

from app.body_binding import BINDING_CHANGE_MIN_SEC, BOUND_STATES  # noqa: E402
from app.temporal_self_graph import _binding_step  # noqa: E402
from orion.schemas.physiology import PhysiologyConstraintReadingV1, PhysiologyPriceV1, PhysiologyStateV1  # noqa: E402
from orion.temporal_self import sources as S  # noqa: E402
from orion.temporal_self.arcs import advance_clock, drain_closed_days, fold, initial_state  # noqa: E402
from orion.temporal_self.day import day_id_for  # noqa: E402
from orion.temporal_self.frame import build_frame  # noqa: E402
from orion.temporal_self.stance_cue import project_stance_cue  # noqa: E402

T0 = datetime(2026, 10, 23, 18, 0, tzinfo=timezone.utc)   # 12:00 MDT
TICK = timedelta(seconds=120)


def replay(ticks: list[tuple[datetime, PhysiologyStateV1]]) -> tuple[list[dict], dict]:
    """Run the shipped producer step over ``ticks`` in order. Returns (rows written, final carry)."""
    rows: list[dict] = []

    async def record(row):
        rows.append(row)
        return True

    deps = SimpleNamespace(record_binding=record, binding_max_gap_sec=360.0)
    carry: dict = {}

    async def go():
        nonlocal carry
        for now, model in ticks:
            carry = await _binding_step(deps, model, {"binding_episodes": carry, "day_id": day_id_for(now)}, now, [])

    asyncio.run(go())
    return rows, carry


def fold_rows(rows: list[dict], end: datetime):
    events = sorted(S.adapt("body_binding", [dict(r, payload_json=r["payload"]) for r in rows]),
                    key=lambda e: (e.occurred_at, e.event_id))
    state = advance_clock(fold(initial_state(), events=events), end)
    state, days = drain_closed_days(state)
    arcs = [a for d in days for a in d.arcs if a.kind == "binding"] + [a for a in state.arcs.values() if a.kind == "binding"]
    return state, arcs


def _state(now: datetime, states: dict[str, str]) -> PhysiologyStateV1:
    cons = []
    for c, st in states.items():
        if st == "absent":
            continue
        cons.append(PhysiologyConstraintReadingV1(
            constraint=c, unit="c" if c == "heat" else "slot_s", level=32.4 if c == "heat" else 4000.0,
            budget=31.0 if c == "heat" else 3600.0,
            price=PhysiologyPriceV1(value=0.0 if st == "slack" else 0.6, ceiling=1.0, bind_state=st, why=st)))
    return PhysiologyStateV1(tick_id=f"physiology:{now.isoformat()}", generated_at=now, constraints=cons)


def _stream(seq: list[dict[str, str]]) -> list[tuple[datetime, PhysiologyStateV1]]:
    return [(T0 + i * TICK, _state(T0 + i * TICK, s)) for i, s in enumerate(seq)]


def fixture() -> list[str]:
    fails: list[str] = []
    H, L = "heat", "lane:agent"
    one = _stream([{H: "slack"}, {H: "binding"}, {H: "slack"}, {H: "binding"}, {H: "slack"}] * 4)
    rows, _ = replay(one)
    if rows:
        fails.append(f"a flapping constraint wrote rows: {[json.loads(r['payload'])['row'] for r in rows]}")
    seq = [{H: "binding"}] * 3 + [{H: "no_reading"}] * 40
    rows, _ = replay(_stream(seq))
    end = T0 + len(seq) * TICK
    _, arcs = fold_rows(rows, end)
    kinds = [json.loads(r["payload"])["row"] for r in rows]
    if kinds != ["onset"]:
        fails.append(f"no_reading must write nothing after the onset: {kinds}")
    if len(arcs) != 1 or arcs[0].closed_reason != "source_stale" or arcs[0].ended_at != T0 + 2 * TICK:
        fails.append(f"no_reading must close the arc by the stale timeout at its last row: {arcs}")
    seq = [{H: "binding", L: "slack"}] * 3 + [{H: "binding", L: "binding"}] * 3 + [{H: "slack", L: "binding"}] * 3
    rows, _ = replay(_stream(seq))
    end = T0 + len(seq) * TICK
    state, arcs = fold_rows(rows, end)
    heat = [a for a in arcs if a.subject_label == H]
    if len(heat) != 1 or heat[0].closed_reason != "process_ended" or heat[0].ended_at != T0 + 6 * TICK:
        fails.append(f"heat should close process_ended at its first slack reading: {heat}")
    seq2 = [{H: "binding", L: "slack"}] * 3 + [{H: "binding", L: "binding"}] * 4
    rows2, _ = replay(_stream(seq2))
    end2 = T0 + len(seq2) * TICK
    state2, arcs2 = fold_rows(rows2, end2)
    if sum(a.status == "open" for a in arcs2) != 2:
        fails.append(f"two bound constraints must give two open arcs: {[(a.subject_label, a.status) for a in arcs2]}")
    cue = project_stance_cue(build_frame(state2, end2), end2, max_chars=2000)
    if not ("Binding now: " in cue.text and "cabinet running hot" in cue.text and "my agent GPU lane full" in cue.text
            and all(f"[arc:{a.arc_id}]" in cue.text for a in arcs2)):
        fails.append(f"stance cue should name both open constraints with citations: {cue.text!r}")
    return fails


# --------------------------------------------------------------------------- live

LIVE_SQL = """
SELECT row_to_json(r) FROM (
  SELECT tick_id, generated_at, constraints FROM physiology_state
  WHERE generated_at >= now() - interval '{days} days' ORDER BY generated_at
) r;
"""


def live(container: str, days: float, max_rows_per_day: float) -> tuple[list[str], dict]:
    out = subprocess.run(["docker", "exec", container, "psql", "-U", "postgres", "-d", "conjourney", "-At",
                          "-v", "ON_ERROR_STOP=1", "-c", "SET default_transaction_read_only = on;",
                          "-c", LIVE_SQL.format(days=float(days))],
                         check=True, capture_output=True, text=True).stdout
    ticks = []
    bound_ticks: Counter = Counter()
    all_ticks: Counter = Counter()
    naive_onsets: Counter = Counter()
    prev: dict = {}
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        model = PhysiologyStateV1.model_validate({"tick_id": r["tick_id"], "generated_at": r["generated_at"],
                                                  "constraints": r["constraints"]})
        ticks.append((model.generated_at, model))
        for c in model.constraints:
            st = c.price.bind_state if c.price is not None else None
            if st is None:
                continue
            all_ticks[c.constraint] += 1
            b = st in BOUND_STATES
            bound_ticks[c.constraint] += b
            if b and not prev.get(c.constraint):
                naive_onsets[c.constraint] += 1
            prev[c.constraint] = b
    if not ticks:
        return ["no physiology_state rows in the window"], {}
    rows, _ = replay(ticks)
    end = ticks[-1][0] + timedelta(seconds=1)
    state, arcs = fold_rows(rows, end)
    span_days = max((ticks[-1][0] - ticks[0][0]).total_seconds() / 86400.0, 1e-9)

    by_kind: dict = defaultdict(Counter)
    keep_at: dict = defaultdict(list)
    for r in rows:
        p = json.loads(r["payload"])
        by_kind[p["constraint"]][p["row"]] += 1
        if p["row"] in ("onset", "change", "keepalive"):
            keep_at[(p["constraint"], p["began_at"])].append(r["occurred_at"])
    episodes = [a for a in arcs if not a.carried_from_previous_day]
    report = {
        "window": {"from": ticks[0][0].isoformat(), "to": ticks[-1][0].isoformat(), "ticks": len(ticks)},
        "rows_total": len(rows),
        "constraints": {},
        "open_at_end": sorted(a.subject_label for a in state.arcs.values() if a.kind == "binding" and a.status == "open"),
        "cue_at_end": project_stance_cue(build_frame(state, end), end, max_chars=2000).text,
    }
    for c in sorted(set(all_ticks) | set(by_kind)):
        mine = [a for a in episodes if a.subject_label == c]
        closed = Counter(a.closed_reason or "open" for a in arcs if a.subject_label == c and a.closed_reason != "day_boundary")
        report["constraints"][c] = {
            "bound_share": round(bound_ticks[c] / all_ticks[c], 3) if all_ticks[c] else None,
            "naive_onsets": naive_onsets[c], "rows": dict(by_kind[c]), "arcs": len(mine),
            "closed": dict(closed),
        }
    fails = []
    for a in arcs:
        if not a.evidence_refs and not a.carried_from_previous_day:
            fails.append(f"arc without evidence: {a.arc_id}")
        if a.ended_at is not None and a.ended_at < a.began_at:
            fails.append(f"arc ends before it began: {a.arc_id}")
    for (c, began), times in keep_at.items():
        times.sort()
        for t0, t1 in zip(times, times[1:]):
            if (t1 - t0).total_seconds() < BINDING_CHANGE_MIN_SEC - 1:
                fails.append(f"{c} episode {began}: rows {t0} and {t1} under BINDING_CHANGE_MIN_SEC apart")
    for c, k in by_kind.items():
        if sum(k.values()) / span_days > max_rows_per_day:
            fails.append(f"{c}: {sum(k.values()) / span_days:.0f} rows/day over {max_rows_per_day}")
    return fails, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--container", default="orion-athena-sql-db")
    ap.add_argument("--max-rows-per-day", type=float, default=200.0)
    a = ap.parse_args()
    fails = fixture()
    print(json.dumps({"fixture": {"passed": not fails, "fails": fails}}, indent=2))
    if a.live:
        live_fails, report = live(a.container, a.days, a.max_rows_per_day)
        print(json.dumps({"live": {"passed": not live_fails, "fails": live_fails[:20], **report}}, indent=2, default=str))
        fails += live_fails
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
