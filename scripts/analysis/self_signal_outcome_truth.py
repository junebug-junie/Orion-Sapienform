#!/usr/bin/env python3
"""Truth-to-rows checker + live replay for the self-signal -> outcome association (spec 2026-10-07
section E). Read-only.

Independent of ``orion/attention/self_signal_outcomes.py``'s ``associate``: it recounts every
association from the hour rows with its own loop and its own Katz interval, and parses every
rendered "tends to" clause back to the association row its ``[assoc:<id>]`` names. A number that
does not resolve, or does not match, is a failure. Used by:

- services/orion-equilibrium-service/evals/test_self_signal_outcome_replay_eval.py (stored history)
- services/orion-thought/evals/self_readings_truth.py (a clause inside a self-reading line)
- this script's ``main`` (live table, before and after deploy)

Usage:
  python scripts/analysis/self_signal_outcome_truth.py --live            # replay live PE history now
  python scripts/analysis/self_signal_outcome_truth.py --persisted       # check persisted rows
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HOUR = timedelta(hours=1)
Z95 = 1.959963984540054

CLAUSE_RE = re.compile(
    r"when this is high, my next hour (?:never had (?P<never>.+?)|had (?P<what>.+?) "
    r"(?:(?P<ratio>\d+\.\d)x as often at equal activity|more often)) "
    r"\((?P<rh>\d+)% of (?P<nh>\d+) high hours vs (?P<rn>\d+)% of (?P<nn>\d+) normal\) "
    r"\[assoc:(?P<id>ssa-[0-9a-f]{10})\]$"
)
_WHAT = {
    "turn_failed": "a failed turn",
    "response_repair": "a reply I had to repair",
    "rpc_timeout": "an RPC timeout",
}


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _hour(ts: Any) -> datetime:
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _strata_and_series(hour_rows: Iterable[Any], signal_id: str, outcome: str, min_history: float):
    cond: dict[datetime, str] = {}
    hist: dict[datetime, float] = {}
    out: dict[datetime, tuple[int, int]] = {}
    for row in hour_rows:
        h = _hour(_get(row, "hour_start"))
        hist[h] = float(_get(row, "history_hours") or 0.0)
        for s in _get(row, "signals") or []:
            if _get(s, "signal_id") == signal_id:
                cond[h] = str(_get(s, "condition"))
        for o in _get(row, "outcomes") or []:
            if _get(o, "outcome") == outcome:
                out[h] = (int(_get(o, "exposure") or 0), int(_get(o, "events") or 0))
    strata: dict[int, list[int]] = {}
    used: dict[datetime, tuple[int, int]] = {}
    for h, cd in cond.items():
        nxt = out.get(h + HOUR)
        if cd not in ("high", "normal") or nxt is None or nxt[0] <= 0 or hist.get(h, 0.0) < min_history:
            continue
        hit = 1 if nxt[1] > 0 else 0
        b = min(16, int(math.floor(math.log2(nxt[0]))))
        st = strata.setdefault(b, [0, 0, 0, 0])
        if cd == "high":
            st[0] += hit
            st[1] += 1
        else:
            st[2] += hit
            st[3] += 1
        used[h] = (1 if cd == "high" else 0, hit)
    return strata, used


def recount(hour_rows: Iterable[Any], signal_id: str, outcome: str, min_history: float = 0.0) -> tuple[int, int, int, int]:
    """(n_high, events_high, n_normal, events_normal): hour H's condition vs hour H+1's outcome."""
    strata, _ = _strata_and_series(hour_rows, signal_id, outcome, min_history)
    return (sum(v[1] for v in strata.values()), sum(v[0] for v in strata.values()),
            sum(v[3] for v in strata.values()), sum(v[2] for v in strata.values()))


def _corr(pairs: list[tuple[int, int]]) -> float:
    if len(pairs) < 3:
        return 0.0
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx, vy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def adjusted_interval(strata, used, z: float) -> tuple[float | None, float | None, float | None, float]:
    """(ratio, low, high, inflation), re-derived here: MH + Greenland-Robins, AR(1) inflation."""
    prev = [(used[h - HOUR], v) for h, v in used.items() if h - HOUR in used]
    r = _corr([(p[0], q[0]) for p, q in prev]) * _corr([(p[1], q[1]) for p, q in prev])
    r = min(0.9, max(0.0, r))
    infl = math.sqrt((1 + r) / (1 - r))
    R = S = V = 0.0
    for a, n1, c, n0 in strata.values():
        n = n1 + n0
        if n1 and n0:
            R += a * n0 / n
            S += c * n1 / n
            V += (n1 * n0 * (a + c) - a * c * n) / n ** 2
    a = sum(v[0] for v in strata.values())
    n1 = sum(v[1] for v in strata.values())
    c = sum(v[2] for v in strata.values())
    n0 = sum(v[3] for v in strata.values())
    if R > 0 and S > 0 and V > 0:
        aa, cc, nn1, nn0 = (a + 0.5, c + 0.5, n1 + 1, n0 + 1) if (a == 0 or c == 0) else (a, c, n1, n0)
        crude = math.sqrt(1 / aa - 1 / nn1 + 1 / cc - 1 / nn0)
        lr, se = math.log(R / S), max(math.sqrt(V / (R * S)), crude) * infl
        return R / S, math.exp(lr - z * se), math.exp(lr + z * se), infl
    lo, hi = katz(a, n1, c, n0, z * infl)
    return (R / S if S > 0 else None), lo, hi, infl


def katz(a: int, n1: int, c: int, n0: int, z: float = Z95) -> tuple[float | None, float | None]:
    if n1 <= 0 or n0 <= 0 or (a == 0 and c == 0):
        return None, None
    if a == 0 or c == 0:
        a, c, n1, n0 = a + 0.5, c + 0.5, n1 + 1, n0 + 1
    lr = math.log((a / n1) / (c / n0))
    se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    return math.exp(lr - z * se), math.exp(lr + z * se)


def check_association(assoc: Any, hour_rows: Sequence[Any]) -> list[str]:
    from statistics import NormalDist

    sid, outcome = str(_get(assoc, "signal_id")), str(_get(assoc, "outcome"))
    tag = f"{sid}/{outcome} [{_get(assoc, 'association_id')}]"
    fails: list[str] = []
    strata, used = _strata_and_series(hour_rows, sid, outcome, float(_get(assoc, "min_history_hours") or 0.0))
    n1, a = sum(v[1] for v in strata.values()), sum(v[0] for v in strata.values())
    n0, c = sum(v[3] for v in strata.values()), sum(v[2] for v in strata.values())
    got = (_get(assoc, "n_high"), _get(assoc, "events_high"), _get(assoc, "n_normal"), _get(assoc, "events_normal"))
    if got != (n1, a, n0, c):
        return [f"{tag}: counts {got} but hour rows give {(n1, a, n0, c)}"]
    m = int(_get(assoc, "n_comparisons") or 1)
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * m))
    if abs(z - float(_get(assoc, "ci_z") or 0.0)) > 1e-4:
        fails.append(f"{tag}: ci_z {_get(assoc, 'ci_z')} but Bonferroni for m={m} is {z:.6f}")
    ratio, lo, hi, infl = adjusted_interval(strata, used, z)
    if abs(infl - float(_get(assoc, "se_inflation") or 0.0)) > 1e-4:
        fails.append(f"{tag}: se_inflation {_get(assoc, 'se_inflation')} but rows give {infl:.6f}")
    for name, mine in (("ratio", ratio), ("ci_low", lo), ("ci_high", hi)):
        theirs = _get(assoc, name)
        if (mine is None) != (theirs is None) or (mine is not None and abs(mine - float(theirs)) > 1e-3):
            fails.append(f"{tag}: {name} {theirs} but rows give {mine}")
    clause = str(_get(assoc, "clause") or "")
    verdict = str(_get(assoc, "verdict"))
    if clause and verdict not in ("tends_more", "tends_less"):
        fails.append(f"{tag}: clause present with verdict {verdict}")
    if verdict in ("tends_more", "tends_less"):
        if lo is None or not (lo > 1.0 or hi < 1.0):
            fails.append(f"{tag}: verdict {verdict} but the interval ({lo}, {hi}) contains 1")
        mh = int(_get(assoc, "min_hours") or 0)
        if n1 < mh or n0 < mh:
            fails.append(f"{tag}: verdict {verdict} below min_hours {mh} ({n1}/{n0})")
    return fails


def check_clause(text: str, associations: Mapping[str, Any]) -> list[str]:
    """``text`` is one clause; ``associations`` maps association_id -> row."""
    m = CLAUSE_RE.search(text)
    if m is None:
        return [f"unparseable tends-to clause: {text!r}"]
    row = associations.get(m["id"])
    if row is None:
        return [f"clause cites {m['id']}, which is not a persisted association row"]
    fails = []
    outcome = str(_get(row, "outcome"))
    what = m["what"] or m["never"]
    if _WHAT.get(outcome) != what:
        fails.append(f"{m['id']}: says '{what}' but the row is {outcome}")
    n1, a, n0, c = (int(_get(row, k)) for k in ("n_high", "events_high", "n_normal", "events_normal"))
    if (int(m["nh"]), int(m["nn"])) != (n1, n0):
        fails.append(f"{m['id']}: shows n {m['nh']}/{m['nn']}, row holds {n1}/{n0}")
    if n1 and int(m["rh"]) != round(100 * a / n1):
        fails.append(f"{m['id']}: shows {m['rh']}% high, row gives {100 * a / n1:.1f}%")
    if n0 and int(m["rn"]) != round(100 * c / n0):
        fails.append(f"{m['id']}: shows {m['rn']}% normal, row gives {100 * c / n0:.1f}%")
    if m["never"] is not None and a != 0:
        fails.append(f"{m['id']}: says 'never' but the row has {a} high-hour events")
    if m["ratio"] is not None:
        rr = _get(row, "ratio")
        if rr is None or abs(float(m["ratio"]) - float(rr)) > 0.051:
            fails.append(f"{m['id']}: shows {m['ratio']}x, row holds {rr}")
    if str(_get(row, "verdict")) not in ("tends_more", "tends_less"):
        fails.append(f"{m['id']}: rendered but verdict is {_get(row, 'verdict')}")
    return fails


# ---------------------------------------------------------------------------------------------
# Live replay (read-only)
# ---------------------------------------------------------------------------------------------


def _connect():
    import psycopg2

    dsn = os.environ.get("ORION_PG_DSN", "host=localhost port=55432 user=postgres password=postgres dbname=conjourney")
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on;")
    return conn


def _table(assocs: Sequence[Any]) -> str:
    lines = [f"{'signal':13} {'outcome':16} {'high hrs':>8} {'normal hrs':>10} {'rate hi':>7} {'rate no':>7} "
             f"{'MH ratio':>8} {'adj. interval':>15} {'infl':>5}  verdict"]
    for a in assocs:
        rh = "-" if a.rate_high is None else f"{a.rate_high:.0%}"
        rn = "-" if a.rate_normal is None else f"{a.rate_normal:.0%}"
        ratio = "-" if a.ratio is None else f"{a.ratio:.2f}"
        ci = "-" if a.ci_low is None else f"({a.ci_low:.2f}, {a.ci_high:.2f})"
        lines.append(
            f"{a.signal_id.removeprefix('node:substrate.'):13} {a.outcome:16} {f'{a.events_high}/{a.n_high}':>8} "
            f"{f'{a.events_normal}/{a.n_normal}':>10} {rh:>7} {rn:>7} {ratio:>8} {ci:>15} {a.se_inflation:>5.2f}  {a.verdict}"
        )
    return "\n".join(lines)


def live(min_hours: int, min_history: float | None) -> int:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "services" / "orion-equilibrium-service"))
    import orion.attention.self_signal_outcomes as sso
    from app.self_signal_outcome_rollup import body_signal_ids  # type: ignore[import-not-found]

    now = datetime.now(timezone.utc)
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT node_id, observed_at, value FROM substrate_node_prediction_error_history ORDER BY observed_at")
    hist: dict[str, list] = {}
    for n, ts, v in cur.fetchall():
        hist.setdefault(str(n), []).append((ts, float(v)))
    first = min(r[0][0] for r in hist.values())
    last = sso.hour_floor(now - timedelta(minutes=10)) - HOUR
    hours, h = [], sso.hour_floor(first) + HOUR
    while h <= last:
        hours.append(h)
        h += HOUR
    cur.execute(
        "SELECT created_at, run_artifact->>'compliance_verdict', CASE WHEN run_artifact ? 'response_repair_ran' "
        "THEN (run_artifact->>'response_repair_ran')::boolean END FROM harness_turn_trace "
        "WHERE created_at >= %s AND created_at < %s AND run_artifact IS NOT NULL",
        (hours[0].replace(tzinfo=None), (last + HOUR).replace(tzinfo=None)),
    )
    turns = sso.turn_outcome_hours((r[0].replace(tzinfo=timezone.utc), r[1], r[2]) for r in cur.fetchall())
    cur.execute(
        "SELECT hour_start, key, success_count, timeout_count FROM transport_baseline_hourly "
        "WHERE hour_start >= %s AND hour_start < %s", (hours[0], last + HOUR),
    )
    rpc = sso.rpc_outcome_hours(cur.fetchall())
    signals = body_signal_ids(hist)
    per = {s: sso.classify_signal_hours(s, hist[s], hours) for s in signals}
    rows = [
        sso.build_hour_row(h, signals=[per[s][h] for s in signals],
                           outcomes=list(turns.get(h, [])) + ([rpc[h]] if h in rpc else []),
                           computed_at=now, history_hours=(h - first).total_seconds() / 3600)
        for h in hours
    ]
    floor = sso.MIN_HISTORY_HOURS if min_history is None else float(min_history)
    assocs = sso.associate_all(rows, computed_at=now, min_hours=min_hours, min_history_hours=floor)
    print(f"replayed {len(rows)} hours {hours[0]:%Y-%m-%d %H}Z..{hours[-1]:%Y-%m-%d %H}Z, min_hours={min_hours}, "
          f"min_history_hours={floor:g} ({sum(1 for r in rows if r.history_hours >= floor)} hours eligible), "
          f"Bonferroni z={assocs[0].ci_z if assocs else '-'}")
    for s in signals:
        conds = [per[s][h].condition for h in hours]
        print(f"  {s.removeprefix('node:substrate.'):13} high={conds.count('high')} normal={conds.count('normal')} "
              f"unknown={conds.count('unknown')}")
    print(_table(assocs))
    fails = [f for a in assocs for f in check_association(a, rows)]
    by_id = {a.association_id: a for a in assocs}
    rendered = [a for a in assocs if a.clause]
    for a in rendered:
        fails += check_clause(a.clause, by_id)
    print(f"would render today: {len(rendered)}")
    for a in rendered:
        print(f"  {a.signal_id}: {a.clause}")
    print("truth-to-rows:", "PASS" if not fails else f"FAIL ({len(fails)})")
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


def persisted() -> int:
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT hour_start, signals, outcomes, history_hours FROM self_signal_hour ORDER BY hour_start")
    hours = [{"hour_start": r[0], "signals": r[1], "outcomes": r[2], "history_hours": r[3]} for r in cur.fetchall()]
    cols = ("association_id", "signal_id", "outcome", "n_high", "events_high", "n_normal", "events_normal",
            "ratio", "ci_low", "ci_high", "ci_z", "se_inflation", "n_comparisons", "min_history_hours",
            "verdict", "min_hours", "clause", "computed_at")
    cur.execute(
        f"SELECT DISTINCT ON (signal_id, outcome) {', '.join(cols)} FROM self_signal_outcome_association "
        "ORDER BY signal_id, outcome, computed_at DESC"
    )
    assocs = [dict(zip(cols, r)) for r in cur.fetchall()]
    # Only the hours the producer had when it computed (its rows are older than computed_at).
    fails = []
    for a in assocs:
        # A row for hour H exists only after H ends (+ grace), so H + 1 h <= computed_at.
        hs = [h for h in hours if _hour(h["hour_start"]) + HOUR <= a["computed_at"]]
        fails += check_association(a, hs)
    by_id = {a["association_id"]: a for a in assocs}
    for a in assocs:
        if a["clause"]:
            fails += check_clause(a["clause"], by_id)
    # Renderable = what both readers show: the newest row's clause-bearing verdict held on the last
    # DEFAULT_CONSECUTIVE_HOLDS consecutive hourly computations (held_latest_by_pair).
    from orion.attention.self_signal_outcomes import (
        ASSOCIATION_COLUMNS, DEFAULT_CONSECUTIVE_HOLDS, association_history_limit, association_history_sql,
        ASSOCIATION_MAX_AGE, hourly_runs, consecutive_holds, held_latest_by_pair,
    )
    from orion.schemas.self_signal_outcome import SelfSignalOutcomeAssociationV1

    holds = DEFAULT_CONSECUTIVE_HOLDS
    cur.execute(association_history_sql("now() - %s::interval", "%s"),
                (f"{holds + 4} hours", association_history_limit(holds)))
    hist = [SelfSignalOutcomeAssociationV1.model_validate(dict(zip(ASSOCIATION_COLUMNS, r))) for r in cur.fetchall()]
    runs = {k: consecutive_holds(v) for k, v in hourly_runs(hist).items()}
    held = held_latest_by_pair(hist, holds=holds,
                               fresh_since=datetime.now(timezone.utc) - ASSOCIATION_MAX_AGE)
    print(f"{len(hours)} persisted hours, {len(assocs)} latest associations, "
          f"{sum(1 for a in assocs if a['clause'])} with a clause this hour, "
          f"{len(held)} renderable (held {holds} consecutive hours)")
    for (sid, outcome), n in sorted(runs.items()):
        if n:
            print(f"  run {sid}/{outcome}: {n}/{holds} consecutive hourly holds")
    print("truth-to-rows:", "PASS" if not fails else f"FAIL ({len(fails)})")
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--live", action="store_true", help="replay live PE history + outcomes now")
    ap.add_argument("--persisted", action="store_true", help="check persisted rows against each other")
    ap.add_argument("--min-hours", type=int, default=24)
    ap.add_argument("--min-history", type=float, default=None,
                    help="hours of PE history an hour needs to count (default: the reducer's floor; 0 = diagnostic)")
    args = ap.parse_args(argv)
    if args.persisted:
        return persisted()
    return live(args.min_hours, args.min_history)


if __name__ == "__main__":
    raise SystemExit(main())
