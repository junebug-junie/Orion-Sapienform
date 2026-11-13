"""Replay chat-turn surprise over settled stance bets: the metric quality gate for node:prediction.chat_turn.

Read-only. Prints, in order:

1. Held-out skill per bet kind (chat, and every caller pooled when live): log-loss gap of the
   recalibrated stated p over the caller's own trailing base rate, SE (max of iid and by-day
   cluster), P(better). This is the evidence behind ``app/turn_surprise.py USED_FACULTIES``.
2. Per-bet calibration of the used kinds after the held-out map: mean excess (calm theory 0) +- SE.
3. Per turn: z distribution, share that rests at exactly 0, share >= curiosity's 0.55 bar.
4. The node (orion/substrate/prediction_turn_surprise.py fold, written at each settle): value
   distribution, exact-0 share, share of the time it is fresh vs absent.
5. Live only: independence from node:substrate.chat (the retired signal) at the same moments.

    # live, inside the prediction image (worktree mounted read-only):
    cd services/orion-prediction && POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_turn_surprise_replay_eval.py --days 21
    # CI, on the committed real fixture (chat bets 2026-10-24 .. 2026-11-13, ids hashed):
    cd services/orion-prediction && PYTHONPATH=../..:. python evals/run_turn_surprise_replay_eval.py --fixture evals/fixtures/turn_self_bets_chat_2026-11-13.json --check

``--check`` exits 1 unless: every used kind has a positive held-out gap, the per-turn rest share is
>= 50 %, and the used kinds' mean excess sits within 3 SE of 0 (calm at 0 by theory, not tuning).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE), str(SERVICE.parents[1])]

import app.turn_surprise as ts  # noqa: E402

PROB_FACULTIES = ("turn.self.asks_question", "turn.self.fetches_outside",
                  "turn.self.more_steps_than_usual", "turn.self.longer_reply_than_usual")
SQL = ("SELECT correlation_id, faculty, caller, placed_at, settled_at, status, stated_p, hit "
       "FROM prediction_turn_self_bet WHERE claim_kind = 'probability' AND placed_at >= %s ORDER BY placed_at")
CHAT_NODE_SQL = ("SELECT observed_at, value FROM substrate_node_prediction_error_history "
                 "WHERE node_id = 'node:substrate.chat' AND observed_at >= %s ORDER BY observed_at")


def _load_fold():
    """orion/substrate/prediction_turn_surprise.py by path: the orion.substrate package __init__ pulls in
    the graph store, which this image does not ship. The module itself is stdlib."""
    import importlib.util

    path = SERVICE.parents[1] / "orion" / "substrate" / "prediction_turn_surprise.py"
    spec = importlib.util.spec_from_file_location("prediction_turn_surprise", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ts(v) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(v))


def load_fixture(path: str) -> list[dict]:
    rows = json.loads(Path(path).read_text())["rows"]
    for r in rows:
        r["placed_at"], r["settled_at"] = _ts(r["placed_at"]), _ts(r["settled_at"])
    return rows


def export_fixture(rows: list[dict], path: str) -> None:
    """Chat rows only; correlation ids hashed (no turn can be looked up from the fixture)."""
    out = [{"correlation_id": hashlib.sha256(r["correlation_id"].encode()).hexdigest()[:12],
            "faculty": r["faculty"], "caller": r["caller"], "placed_at": r["placed_at"].isoformat(),
            "settled_at": r["settled_at"].isoformat(), "status": r["status"], "stated_p": r["stated_p"],
            "hit": r["hit"]} for r in rows if r["caller"] == ts.SCOPE_CALLER]
    Path(path).write_text(json.dumps({"source": "prediction_turn_self_bet, claim_kind=probability, caller=chat",
                                      "exported_at": datetime.now(timezone.utc).isoformat(), "rows": out},
                                     indent=0) + "\n")
    print(f"wrote {len(out)} rows -> {path}")


def _q(vals: list[float], q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


def skill_table(rows: list[dict], live: bool) -> dict:
    print("== 1. held-out skill per bet kind (log-loss gap over the caller's trailing base rate) ==")
    out = {}
    scopes = [("chat", ts.SCOPE_CALLER)] + ([("all callers", None)] if live else [])
    for label, caller in scopes:
        for fac in PROB_FACULTIES:
            for recal in (False, True):
                r = ts.heldout_skill(rows, fac, caller=caller, recalibrate=recal)
                if label == "chat" and recal:
                    out[fac] = r
                if r["gap"] is None:
                    print(f"  {label:11s} {fac:34s} {'recal' if recal else 'raw  '} n={r['n']} (too few)")
                    continue
                print(f"  {label:11s} {fac:34s} {'recal' if recal else 'raw  '} n={r['n']:5d} days={r['days']:2d} "
                      f"gap={r['gap']:+.4f} SE={r['se']:.4f} (iid {r['se_iid']:.4f}, day {r['se_day']:.4f}) "
                      f"P(better)={r['prob_better']:.3f} rate={r['rate']:.3f} "
                      f"{'USED' if fac in ts.USED_FACULTIES and caller else ''}")
    return out


def replay_turns(rows: list[dict]) -> list[ts.TurnSurprise]:
    chat = [r for r in rows if r["caller"] == ts.SCOPE_CALLER]
    history = ts.CalibrationHistory(chat)
    by_turn: dict[str, list[dict]] = defaultdict(list)
    for r in chat:
        by_turn[r["correlation_id"]].append(r)
    turns, reasons = [], defaultdict(int)
    for cid, trs in sorted(by_turn.items(), key=lambda kv: kv[1][0]["placed_at"]):
        out, why = ts.score_turn(trs, history)
        reasons[why] += 1
        if out is not None:
            turns.append((trs[0]["settled_at"], out))
    print(f"  turns: {dict(reasons)}")
    return turns


def bet_calibration(turns) -> dict:
    print("== 2. used kinds after the held-out map: mean excess per bet (calm theory 0) ==")
    by = defaultdict(list)
    for _, t in turns:
        for b in t.bets:
            by[b.faculty].append(b)
    out = {}
    for fac, bs in sorted(by.items()):
        ex = [b.excess for b in bs]
        m, se = statistics.fmean(ex), statistics.stdev(ex) / math.sqrt(len(ex))
        sd_theory = math.sqrt(statistics.fmean(b.calm_var for b in bs))
        out[fac] = (m, se)
        print(f"  {fac:34s} n={len(bs):4d} mean_excess={m:+.4f} SE={se:.4f} ({m / se:+.2f} SE) "
              f"sd={statistics.pstdev(ex):.3f} (calm theory {sd_theory:.3f}) "
              f"mean q={statistics.fmean(b.calibrated_p for b in bs):.3f} happened={statistics.fmean(b.happened for b in bs):.3f}")
    return out


def turn_stats(turns) -> float:
    print("== 3. per turn ==")
    rest = _turn_stats(turns, "all")
    last = max(t for t, _ in turns)
    _turn_stats([x for x in turns if x[0] >= last - timedelta(days=7)], "last 7 days")
    return rest


def _turn_stats(turns, label: str) -> float:
    zs = [t.z for _, t in turns]
    print(f"  [{label}]")
    vals = [ts.node_value(z) for z in zs]
    rest = sum(1 for v in vals if v == 0.0) / len(vals)
    print(f"  turns={len(zs)}  z mean {statistics.fmean(zs):+.3f} sd {statistics.pstdev(zs):.3f}  "
          f"p10 {_q(zs, .1):+.2f} p50 {_q(zs, .5):+.2f} p90 {_q(zs, .9):+.2f} max {max(zs):+.2f}")
    print(f"  value: exact 0 (rest) {rest:.1%}  p50 {_q(vals, .5):.3f} p90 {_q(vals, .9):.3f} "
          f"p99 {_q(vals, .99):.3f}  share >= 0.55 {sum(v >= 0.55 for v in vals) / len(vals):.1%}  "
          f"share == 1.0 {sum(v >= 1.0 for v in vals) / len(vals):.1%}")
    return rest


def node_replay(turns, fold) -> list[tuple[datetime, float, float]]:
    print("== 4. node:prediction.chat_turn replay (written at each settle) ==")
    st = fold.TurnSurpriseState()
    writes = []
    for settled, t in sorted(turns, key=lambda x: x[0]):
        ev = {"caller": "chat", "correlation_id": t.correlation_id, "settled_at": settled, "excess": t.excess,
              "calm_var": t.calm_var, "bets": t.bets}
        node, verdict = st.add(ev, now=settled)
        r = st.reading(node, now=settled) if verdict == "accepted" else None
        if r is not None:
            writes.append((settled, r.value, r.z, r.n))
    vals = [w[1] for w in writes]
    span = (writes[-1][0] - writes[0][0]).total_seconds() if len(writes) > 1 else 0.0
    fresh = 0.0
    for (a, *_), (b, *_) in zip(writes, writes[1:]):
        fresh += min((b - a).total_seconds(), 1800.0)
    print(f"  writes={len(writes)} turns/window median {statistics.median(w[3] for w in writes):.0f}  "
          f"exact 0 {sum(v == 0 for v in vals) / len(vals):.1%}  p50 {_q(vals, .5):.3f} p90 {_q(vals, .9):.3f} "
          f"max {max(vals):.3f}  share >= 0.55 {sum(v >= 0.55 for v in vals) / len(vals):.1%}")
    print(f"  fresh (a write in the last 1800 s) {fresh / span:.1%} of the {span / 86400:.1f} days; absent otherwise")
    return writes


def independence(writes, chat_rows) -> None:
    print("== 5. independence from node:substrate.chat (its newest value <= 30 min before each write) ==")
    times = [r["observed_at"] for r in chat_rows]
    import bisect

    pairs = []
    for t, v, z, _ in writes:
        i = bisect.bisect_right(times, t) - 1
        if i >= 0 and (t - times[i]) <= timedelta(minutes=30):
            pairs.append((v, float(chat_rows[i]["value"])))
    if len(pairs) < 10:
        print(f"  too few overlapping writes ({len(pairs)})")
        return
    a, b = zip(*pairs)
    r = statistics.correlation(a, b) if statistics.pstdev(b) > 0 else float("nan")
    ra = [sorted(a).index(x) for x in a]
    rb = [sorted(b).index(x) for x in b]
    rs = statistics.correlation(ra, rb) if statistics.pstdev(rb) > 0 else float("nan")
    print(f"  pairs={len(pairs)}  pearson {r:+.3f}  spearman {rs:+.3f}  "
          f"old chat exact 0 at these moments {sum(x == 0 for x in b) / len(b):.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=21.0)
    ap.add_argument("--fixture")
    ap.add_argument("--export")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    fold = _load_fold()
    chat_rows = None
    if args.fixture:
        rows, live = load_fixture(args.fixture), False
    else:
        from app.store import PostgresStore

        store = PostgresStore(os.environ["POSTGRES_URI"])
        since = datetime.now(timezone.utc) - timedelta(days=args.days)
        rows = [dict(r, placed_at=_ts(r["placed_at"]), settled_at=_ts(r["settled_at"]),
                     stated_p=None if r["stated_p"] is None else float(r["stated_p"]))
                for r in store._all(SQL, (since,))]
        chat_rows = store._all(CHAT_NODE_SQL, (since,))
        live = True
        if args.export:
            export_fixture(rows, args.export)
    print(f"rows={len(rows)} ({'live' if live else 'fixture'})  used={ts.USED_FACULTIES} caller={ts.SCOPE_CALLER}")
    skill = skill_table(rows, live)
    turns = replay_turns(rows)
    cal = bet_calibration(turns)
    rest = turn_stats(turns)
    writes = node_replay(turns, fold)
    if chat_rows is not None:
        independence(writes, chat_rows)
    if args.check:
        fails = [f"{f}: gap {skill[f]['gap']}" for f in ts.USED_FACULTIES if not (skill[f]["gap"] or 0) > 0]
        fails += [f"{f}: mean excess {m:+.3f} > 3 SE" for f, (m, se) in cal.items() if abs(m) > 3 * se]
        if rest < 0.5:
            fails.append(f"rest share {rest:.1%} < 50%")
        print("CHECK", "FAIL " + "; ".join(fails) if fails else "PASS")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
