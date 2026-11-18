"""Street attention eval: does the gate flood, does the node rest, does a ring win?

Two modes, both read-only.

``--synthetic`` (default; also run by test_street_attention_eval.py): a
simulated week of graded handoffs with CALIBRATED p-values (uniform, the
"ordinary street" case) plus a few rings, pushed through the real gate and the
real every-tick node value, then the real world-first ranking. Pass criteria:

- the hourly cap is never exceeded by graded events (rings always pass, so
  rings can add on top; reported as worst_trailing_hour_passes_with_rings);
- the non-ring pass rate is ~5% of graded events (p < 0.05 by chance), never
  "most of them";
- a quiet hour (no events) yields no passes and the node reads exactly 0.0;
- the node is nonzero on at most 10% of ticks (top-decile edge), so a fresh
  ring still lands in the "high"/"unusual" band and wins world-first, and a
  0.0 reading is judged quiet (no winner).

``--live``: reads ``substrate_node_prediction_error_history`` and
``substrate_attention_broadcast_log`` for ``node:substrate.street`` over the
last ``--hours`` and reports readings, nonzero share, distinct values and
street wins. UNVERIFIED until real street events reach the bus.

Run:
  python services/orion-substrate-runtime/evals/run_street_attention_eval.py
  POSTGRES_URI=postgresql://... python .../run_street_attention_eval.py --live --hours 48
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.attention.world_first import node_candidate, rank_candidates  # noqa: E402
from orion.substrate.prediction_error_magnitude import compute_prediction_error_magnitude  # noqa: E402
from orion.substrate.street_attention import STREET_NODE_ID, StreetAttentionState  # noqa: E402

TICK_SEC = 30.0


def _event(kind: str, p: float | None, i: int) -> SimpleNamespace:
    return SimpleNamespace(
        kind=kind,
        p_value=p,
        stream_ids=["walkway"] if kind != "doorbell_ring" else ["doorbell"],
        salience=1.0 if kind == "doorbell_ring" else 0.4,
        event_id=f"e{i}",
        correlation_id=f"c{i}",
        person_guess=None,
    )


def synthetic(
    *,
    days: int = 7,
    handoffs_per_hour: int = 12,
    rings_per_day: int = 3,
    max_per_hour: int = 4,
    hold_sec: float = 90.0,
    quiet_hour: int = 3,
    seed: int = 11,
    p_scale: float = 1.0,
) -> dict:
    """``p_scale`` < 1 simulates a miscalibrated (flooding) model: every p is
    drawn from U(0, p_scale), so most graded events clear p < 0.05."""
    rng = random.Random(seed)
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
    st = StreetAttentionState()
    events: list[tuple[float, SimpleNamespace]] = []
    i = 0
    for h in range(days * 24):
        if h % 24 == quiet_hour:
            continue
        for _ in range(handoffs_per_hour):
            events.append((t0 + h * 3600 + rng.random() * 3600, _event("handoff_met", rng.random() * p_scale, i)))
            i += 1
    for d in range(days):
        for _ in range(rings_per_day):
            hour = rng.choice([hh for hh in range(8, 22) if hh != quiet_hour])
            events.append((t0 + (d * 24 + hour) * 3600 + rng.random() * 3600, _event("doorbell_ring", None, i)))
            i += 1
    events.sort(key=lambda e: e[0])

    end = t0 + days * 86400
    ticks: list[tuple[float, float]] = []
    passes: list[tuple[float, str]] = []
    graded = graded_pass = 0
    j = 0
    t = t0
    while t < end:
        while j < len(events) and events[j][0] <= t:
            ts, ev = events[j]
            st.frame_seen("walkway", ts)
            v = st.judge(ev, now_epoch=ts, max_per_hour=max_per_hour, camera_up_max_age_sec=600.0)
            if ev.kind != "doorbell_ring":
                graded += 1
                graded_pass += int(v.passed)
            if v.passed:
                passes.append((ts, ev.kind))
            j += 1
        ticks.append((t, st.held_reading(t, hold_sec)[0]))
        t += TICK_SEC

    # Trailing-hour cap, checked at every pass time (not clock hours). A ring
    # always passes, even over the cap (spec section 6), so the cap binds the
    # graded passes; rings are reported beside it.
    def worst(kinds_counted) -> int:
        times = [ts for ts, k in passes if k in kinds_counted]
        return max((sum(1 for q in times if p - 3600 < q <= p) for p in times), default=0)

    worst_graded = worst({"handoff_met", "handoff_late", "handoff_missed"})
    worst_all = worst({"handoff_met", "handoff_late", "handoff_missed", "doorbell_ring"})
    quiet_vals = [v for ts, v in ticks if int((ts - t0) // 3600) % 24 == quiet_hour and (ts - t0) % 3600 > hold_sec]
    nonzero = sum(1 for _, v in ticks if v > 0.0)
    now = datetime.fromtimestamp(end, tz=timezone.utc)
    hist = [(datetime.fromtimestamp(ts, tz=timezone.utc), v) for ts, v in ticks]

    def wins(value: float) -> bool:
        mag = compute_prediction_error_magnitude(value=value, observed_at=now, history=hist, now=now)
        cand = node_candidate(
            node_id=STREET_NODE_ID, label="street", magnitude=mag, observed_at=now, now=now,
            history_values=[v for _, v in hist],
        )
        return rank_candidates([cand]).winner is not None

    return {
        "ticks": len(ticks),
        "graded_events": graded,
        "graded_pass_rate": round(graded_pass / graded, 4) if graded else None,
        "rings": sum(1 for _, e in events if e.kind == "doorbell_ring"),
        "worst_trailing_hour_graded_passes": worst_graded,
        "worst_trailing_hour_passes_with_rings": worst_all,
        "max_per_hour": max_per_hour,
        "quiet_hour_max_value": max(quiet_vals) if quiet_vals else None,
        "nonzero_tick_share": round(nonzero / len(ticks), 4),
        "ring_wins": wins(1.0),
        "handoff_wins": wins(0.4),
        "calm_wins": wins(0.0),
        "blocked_by_reason": st.blocked_by_reason,
    }


def verdict(r: dict) -> list[str]:
    problems = []
    if r["worst_trailing_hour_graded_passes"] > r["max_per_hour"]:
        problems.append("hourly cap exceeded by graded events")
    if r["graded_pass_rate"] is None or not 0.01 <= r["graded_pass_rate"] <= 0.10:
        problems.append(f"graded pass rate {r['graded_pass_rate']} not ~5%")
    if r["quiet_hour_max_value"] != 0.0:
        problems.append("quiet hour not at rest")
    if r["nonzero_tick_share"] > 0.10:
        problems.append("node nonzero on >10% of ticks")
    if not r["ring_wins"]:
        problems.append("a ring does not win")
    if r["calm_wins"]:
        problems.append("a calm 0.0 wins")
    return problems


def live(hours: float) -> dict:
    from sqlalchemy import create_engine, text

    eng = create_engine(os.environ["POSTGRES_URI"])
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    with eng.connect() as c:
        rows = c.execute(
            text(
                "SELECT value FROM substrate_node_prediction_error_history "
                "WHERE node_id = :n AND observed_at >= :s"
            ),
            {"n": STREET_NODE_ID, "s": since},
        ).fetchall()
        won = c.execute(
            text(
                "SELECT count(*) FROM substrate_attention_broadcast_log WHERE generated_at >= :s "
                "AND projection_json->'attended_node_ids' ? :n"
            ),
            {"n": STREET_NODE_ID, "s": since},
        ).scalar()
        cycles = c.execute(
            text("SELECT count(*) FROM substrate_attention_broadcast_log WHERE generated_at >= :s"),
            {"s": since},
        ).scalar()
    values = [float(r[0]) for r in rows]
    return {
        "hours": hours,
        "readings": len(values),
        "nonzero_share": round(sum(1 for v in values if v > 0) / len(values), 4) if values else None,
        "distinct_values": sorted(set(values))[:10],
        "attention_cycles": int(cycles or 0),
        "street_wins": int(won or 0),
        "status": "UNVERIFIED (no readings yet)" if not values else "measured",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--hours", type=float, default=48.0)
    args = ap.parse_args()
    if args.live:
        print(json.dumps(live(args.hours), indent=2, default=str))
        return 0
    r = synthetic()
    problems = verdict(r)
    print(json.dumps({**r, "problems": problems}, indent=2, default=str))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
