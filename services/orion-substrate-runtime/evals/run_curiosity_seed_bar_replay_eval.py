"""Replay curiosity's prediction-error seeds per node, old single bar (0.55) vs per-node bars (#3821).

Read-only. Live only (needs POSTGRES_URI; from the host use the published port, e.g.
``postgresql://postgres:<pw>@localhost:55432/conjourney``).

    PYTHONDONTWRITEBYTECODE=1 POSTGRES_URI=... python services/orion-substrate-runtime/evals/run_curiosity_seed_bar_replay_eval.py --days 7

What it does:

1. Node writes. For every seedable node, the stored values in ``substrate_node_prediction_error_history``
   over the window (the table is being retired; while it lives it is the node's write log). For
   ``node:prediction.room_joint`` / ``.body_joint`` (live only since 2026-11-24 04:35Z) it also
   rebuilds a 7-day PROXY from ``prediction_joint_surprise``: the blended ``family_a`` adaptive rows
   (same conformal p and excess, never published) folded through the real
   ``orion/substrate/prediction_joint_surprise.py`` window at each row's ``scored_at``.
2. Curiosity ticks. Every 60 s (``ORION_ENDOGENOUS_CURIOSITY_TICK_INTERVAL_SEC``), the newest write at
   or before the tick, staleness-decayed linearly over 1800 s, exactly as
   ``endogenous_curiosity._prediction_error_candidates`` reads it. A tick seeds when the decayed value
   is >= the bar. (Not modelled: the budget cap, System One admission, the evaluator.)
3. Prints per node: share of writes at or above each bar, seed ticks per day (before / after), the
   days with no seed, and the share of fresh hours with no seed. A forecast node writes ~290 times a
   day, so even at a calibrated 5 % it seeds on most days; rest shows as most HOURS seeding nothing
   and the value sitting at exactly 0 about half the time.
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    """Stdlib modules by path (orion.substrate's package init is heavy)."""
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


csb = _load("curiosity_seed_bar", "orion/substrate/curiosity_seed_bar.py")
pj = _load("prediction_joint_surprise", "orion/substrate/prediction_joint_surprise.py")

OLD_BAR = 0.55
TICK = timedelta(seconds=60)
HORIZON_S = 1800.0
NODES = (
    csb.BODY_NODE_ID, csb.BODY_JOINT_NODE_ID, csb.ROOM_NODE_ID, csb.ROOM_JOINT_NODE_ID,
    csb.CHAT_TURN_NODE_ID, csb.PERCEPTION_NODE_ID, csb.PERCEPTION_WALKWAY_NODE_ID,
)
HISTORY_SQL = ("SELECT node_id, observed_at, value FROM substrate_node_prediction_error_history "
               "WHERE node_id = ANY(%s) AND observed_at >= %s ORDER BY observed_at")
JOINT_SQL = ("SELECT family, horizon_min, origin_ts, target_ts, scored_at, excess, n_observed, joint_id "
             "FROM prediction_joint_surprise WHERE status = 'scored' AND calibration = 'adaptive' "
             "AND scored_at >= %s ORDER BY scored_at")


def tick_seeds(writes: list[tuple[datetime, float]], bar: float, start: datetime, end: datetime):
    """(seed ticks per UTC day, UTC hours with a fresh reading, UTC hours with a seed)."""
    times = [w[0] for w in writes]
    per_day: dict[str, int] = defaultdict(int)
    fresh_hours: set[datetime] = set()
    seed_hours: set[datetime] = set()
    t = start
    while t <= end:
        i = bisect.bisect_right(times, t) - 1
        if i >= 0:
            age = (t - times[i]).total_seconds()
            if age < HORIZON_S:
                hour = t.replace(minute=0)
                fresh_hours.add(hour)
                if writes[i][1] * (1.0 - age / HORIZON_S) >= bar:
                    per_day[t.date().isoformat()] += 1
                    seed_hours.add(hour)
        t += TICK
    return per_day, fresh_hours, seed_hours


def joint_proxy(rows, family: str, n_cal: int = 1000) -> list[tuple[datetime, float]]:
    """Fold one family's scored adaptive rows through the real joint-surprise window."""
    st = pj.JointSurpriseState()
    node = pj.ROOM_JOINT_NODE_ID
    out = []
    for fam, hm, origin, target, scored, excess, _n, jid in rows:
        if fam != family or excess is None or scored is None:
            continue
        ev = {"group": "room", "horizon_min": hm, "origin_ts": origin, "target_ts": target,
              "excess": float(excess), "n_cal": n_cal, "joint_id": jid, "top_shares": {}}
        _, why = st.add(ev, now=scored)
        if why == "accepted":
            r = st.reading(node, now=scored)
            if r is not None:
                out.append((scored, r.value))
    return out


def report(label: str, writes, bar: float, start: datetime, end: datetime) -> None:
    days = [(start + timedelta(days=k)).date().isoformat() for k in range((end - start).days + 1)]
    if not writes:
        print(f"{label}: no writes")
        return
    vals = [v for _, v in writes]
    old, fresh_h, old_h = tick_seeds(writes, OLD_BAR, start, end)
    new, _, new_h = tick_seeds(writes, bar, start, end)
    live_days = sorted({w[0].date().isoformat() for w in writes})
    print(f"{label}  bar {OLD_BAR:.3f} -> {bar:.3f}  writes={len(vals)} "
          f"exact0={sum(v == 0 for v in vals) / len(vals):.1%} "
          f"write>=old {sum(v >= OLD_BAR for v in vals) / len(vals):.2%} "
          f"write>=new {sum(v >= bar for v in vals) / len(vals):.2%} max={max(vals):.3f}")
    print("  day         seeds_old seeds_new")
    for d in days:
        if d in live_days or old.get(d) or new.get(d):
            print(f"  {d}  {old.get(d, 0):9d} {new.get(d, 0):9d}")
    calm_old = sum(1 for d in live_days if not old.get(d))
    calm_new = sum(1 for d in live_days if not new.get(d))
    print(f"  total seed ticks {sum(old.values())} -> {sum(new.values())};  "
          f"days with writes {len(live_days)}, of them with no seed: {calm_old} -> {calm_new}")
    if fresh_h:
        print(f"  hours with a fresh reading {len(fresh_h)}; resting (no seed) "
              f"{1 - len(old_h) / len(fresh_h):.1%} -> {1 - len(new_h) / len(fresh_h):.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    uri = os.environ.get("POSTGRES_URI")
    if not uri:
        print("POSTGRES_URI not set (live-only eval)", file=sys.stderr)
        return 2
    import psycopg2

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    with psycopg2.connect(uri) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(HISTORY_SQL, (list(NODES), start))
        hist = cur.fetchall()
        cur.execute(JOINT_SQL, (start - timedelta(minutes=30),))
        joint = cur.fetchall()
    by_node: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for nid, ts, v in hist:
        by_node[nid].append((ts, float(v)))
    print(f"window {start.isoformat()} .. {end.isoformat()}  tick {TICK.seconds}s  decay horizon {HORIZON_S:.0f}s")
    for nid in NODES:
        report(nid, by_node.get(nid, []), csb.seed_bar(nid, OLD_BAR), start, end)
    for fam, label in (("family_a", "PROXY room/body_joint scale: blended family_a"),
                       ("family_a_room", "REBUILT node:prediction.room_joint (family_a_room)"),
                       ("family_a_body", "REBUILT node:prediction.body_joint (family_a_body)")):
        report(label, joint_proxy(joint, fam), csb.FORECAST_Z_CUT / csb.Z_FULL_SCALE, start, end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
