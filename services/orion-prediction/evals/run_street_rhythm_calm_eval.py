"""Calm check for node:prediction.street_rhythm (walkway rhythm forecast surprise): the go-live gate.

Read-only. Design: docs/superpowers/specs/2026-11-24-pe-retire-perception-and-alarms-design.md rev 4,
item 5 ("Gate (7 days, shadow)"). The node ships in SHADOW (orion/schemas/attention_candidate.py
``STREET_RHYTHM_SHADOW``); it competes only after this check passes on 7 days of live outcomes.

Prints, in order:

1. Source: closed walkway expectations, why each was or was not scored, the refusal rate
   (unscorable + stale key + unknown key over all closed rows). Refusals are never misses.
2. Per-outcome excess (nats, raw stated p): n, mean with a 95 % CI (SE = max of iid and by-day
   cluster), p90 / p99 / max, misses at stated p >= 0.9.
3. Calibration: n, mean stated p and hit rate per stated-p bucket.
4. Identity (#3388): excess and miss rate by key age; distinct individual keys and how many lived
   under 10 days (sighting life, and forecast span inside the window); miss rate of each key's last 3 expectations against the rest (a re-key looks like
   a run of misses at a key's end).
5. The node, replayed at its event writes (orion/substrate/prediction_street_rhythm.py, 30-min window,
   one write per scoring time): exact-0 share, p90 / p99, share of time it is fresh (<= 600 s since a
   write), history rows per 7 days against the attention contests' 200-readings floor
   (DEFAULT_MIN_READINGS; #3917 replaces that floor with calm-theory bands for forecast-surprise
   nodes), and the curiosity bar's calm (exact) and observed crossing rates.
6. Live only: independence. Pearson r of the node (at its writes) against walkway scene change and the
   street alarm (substrate_node_signal_history), and of each outcome's excess against the walkway
   value and street fires inside its window.

    # live (read-only; any host with the DB):
    cd services/orion-prediction && POSTGRES_URI=postgresql://... PYTHONPATH=../..:. \\
        python evals/run_street_rhythm_calm_eval.py --days 7 --check
    # CI, on the committed real fixture (ids hashed, no names):
    cd services/orion-prediction && PYTHONPATH=../..:. \\
        python evals/run_street_rhythm_calm_eval.py --fixture evals/fixtures/street_rhythm_2026-11-25.json --check
    # refresh the fixture from live:
    ... --days 14 --export evals/fixtures/street_rhythm_<date>.json

``--check`` exits 1 unless (the design's gate, on the tail): the mean-excess 95 % CI covers 0;
p99 <= 1.2 nats; max <= 4.0 nats; at most 2 misses at stated p >= 0.9; at least MIN_SCORED outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

import app.street_rhythm as sr  # noqa: E402

GATE_P99 = 1.2
GATE_MAX = 4.0
GATE_CONFIDENT_MISSES = 2       # misses at stated p >= CONFIDENT_P
CONFIDENT_P = 0.9
MIN_SCORED = 100                # below this the tail numbers say nothing
MIN_READINGS = 200              # orion/substrate/prediction_error_magnitude.py DEFAULT_MIN_READINGS
SEED_BAR = 1.8 / 3.0            # orion/substrate/curiosity_seed_bar.py STREET_RHYTHM_Z_CUT / Z_FULL_SCALE (test)
BUCKETS = ((0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))
KEY_AGE_BUCKETS = ((0.0, 14.0), (14.0, 28.0), (28.0, 1e9))   # days since the key's first sighting
WALKWAY = "node:substrate.perception_walkway"
STREET = "node:substrate.street"
SIGNAL_SQL = ("SELECT node_id, observed_at, value FROM substrate_node_signal_history "
              "WHERE node_id = ANY(%s) AND observed_at >= %s AND observed_at < %s ORDER BY observed_at")
SIGHTING_SQL = ("SELECT individual_id, min(started_at) AS first_at, max(started_at) AS last_at "
                "FROM vision_individual_sighting WHERE stream_id = 'walkway' GROUP BY 1")


def _load_fold():
    """orion/substrate/prediction_street_rhythm.py by path: the orion.substrate package __init__ pulls in
    the graph store, which the prediction image does not ship. The module itself is stdlib."""
    path = SERVICE.parents[1] / "orion" / "substrate" / "prediction_street_rhythm.py"
    spec = importlib.util.spec_from_file_location("prediction_street_rhythm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ts(v):
    if v is None or isinstance(v, datetime):
        return v
    return datetime.fromisoformat(v)


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def load_live(days: float, until: datetime | None):
    import psycopg
    from psycopg.rows import dict_row

    from app.store import PostgresStore

    until = until or datetime.now(timezone.utc)
    since = until - timedelta(days=days)
    store = PostgresStore(os.environ["POSTGRES_URI"])
    rows = store.street_rhythm_outcomes(since, until)
    with psycopg.connect(os.environ["POSTGRES_URI"], row_factory=dict_row, autocommit=True) as conn:
        conn.execute("SET default_transaction_read_only = on")
        signals = list(conn.execute(SIGNAL_SQL, ([WALKWAY, STREET], since, until)).fetchall())
        sightings = {r["individual_id"]: (r["first_at"], r["last_at"]) for r in conn.execute(SIGHTING_SQL)}
    store.close()
    return rows, signals, sightings, since, until


def export_fixture(path: str, rows, sightings, since, until) -> None:
    """Real outcomes, ids hashed, no names (subject labels are never read). Signals are not exported."""
    out = []
    for r in rows:
        key = str(r["subject_key"])
        kind, _, ident = key.partition(":")
        hk = f"individual:{_hash(ident)}" if kind == "individual" else key
        span = sightings.get(ident) if kind == "individual" else None
        out.append({
            "expectation_id": _hash(str(r["expectation_id"])), "stream_id": r["stream_id"], "subject_key": hk,
            "window_start": r["window_start"].isoformat(), "window_end": r["window_end"].isoformat(),
            "confidence": float(r["confidence"]), "status": r["status"],
            "scored_at": r["scored_at"].isoformat() if r["scored_at"] else None,
            "last_seen_before": r["last_seen_before"].isoformat() if r["last_seen_before"] else None,
            "first_seen_before": r["first_seen_before"].isoformat() if r["first_seen_before"] else None,
            "key_last_seen_at": span[1].isoformat() if span else None,
        })
    body = ",\n".join(json.dumps(r, separators=(",", ":")) for r in out)
    Path(path).write_text(f'{{"since":"{since.isoformat()}","until":"{until.isoformat()}","rows":[\n{body}\n]}}\n')
    print(f"wrote {path}: {len(out)} rows")


def load_fixture(path: str):
    doc = json.loads(Path(path).read_text())
    rows = []
    for r in doc["rows"]:
        r = dict(r)
        for k in ("window_start", "window_end", "scored_at", "last_seen_before", "first_seen_before",
                  "key_last_seen_at"):
            r[k] = _ts(r.get(k))
        rows.append(r)
    return rows, [], None, _ts(doc["since"]), _ts(doc["until"])


def _se(outs) -> float:
    vals = [o.terms.excess for o in outs]
    n = len(vals)
    if n < 2:
        return float("nan")
    m = statistics.fmean(vals)
    se_iid = statistics.stdev(vals) / math.sqrt(n)
    days: dict = defaultdict(float)
    for o in outs:
        days[o.scored_at.date()] += o.terms.excess - m
    g = len(days)
    se_day = math.sqrt(sum(v * v for v in days.values()) * g / (g - 1)) / n if g > 1 else se_iid
    return max(se_iid, se_day)


def _pearson(xs, ys) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def calm_crossing_rate(ps: list[float], bar: float, *, rng=None, max_exact: int = 14, draws: int = 4000) -> float:
    """P(node value >= bar) for one window if every outcome were drawn from its own stated p (calm).
    Exact enumeration up to ``max_exact`` outcomes, Monte Carlo above."""
    import itertools
    import random

    terms = [(sr.outcome_terms(p, True), sr.outcome_terms(p, False)) for p in ps]
    var = sum(t[0].calm_var for t in terms)
    if not ps or var <= 1e-12:
        return 0.0
    sd = math.sqrt(var)
    if len(ps) <= max_exact:
        total = 0.0
        for combo in itertools.product((True, False), repeat=len(ps)):
            z = sum((h.excess if hit else m.excess) for (h, m), hit in zip(terms, combo)) / sd
            if z / 3.0 >= bar:
                total += math.prod(t[0].p if hit else 1.0 - t[0].p for t, hit in zip(terms, combo))
        return total
    rng = rng or random.Random(1)
    hits = 0
    for _ in range(draws):
        z = sum((h.excess if rng.random() < h.p else m.excess) for h, m in terms) / sd
        hits += z / 3.0 >= bar
    return hits / draws


def _f(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def run(rows, signals, sightings, since, until) -> dict:
    fold = _load_fold()
    outs, reasons = sr.score_rows(rows)
    closed = sum(reasons.values()) - reasons.get("not_closed", 0) - reasons.get("not_walkway", 0)
    refused = sum(reasons.get(k, 0) for k in sr.REFUSAL_REASONS)
    res: dict = {"since": since.isoformat(), "until": until.isoformat(), "closed": closed,
                 "reasons": dict(sorted(reasons.items())), "refusal_rate": refused / closed if closed else None}
    print(f"window {since:%Y-%m-%d %H:%MZ} .. {until:%Y-%m-%d %H:%MZ}")
    print(f"1. closed {closed}  reasons {res['reasons']}  refusal rate {_f(res['refusal_rate'])}")

    ex = [o.terms.excess for o in outs]
    n = len(ex)
    mean = statistics.fmean(ex) if ex else None
    se = _se(outs) if n >= 2 else None
    ci = (mean - 1.96 * se, mean + 1.96 * se) if se is not None and not math.isnan(se) else None
    conf_miss = [o for o in outs if o.stated_p >= CONFIDENT_P and not o.terms.happened]
    res.update(n=n, mean_excess=mean, se=se, ci95=ci, p90=sr.quantile(ex, 0.9), p99=sr.quantile(ex, 0.99),
               max=max(ex) if ex else None, sd=statistics.pstdev(ex) if n > 1 else None,
               confident_misses=len(conf_miss), share_le_0=(sum(1 for e in ex if e <= 0) / n) if n else None,
               log_loss=statistics.fmean(o.terms.surprisal for o in outs) if outs else None)
    print(f"2. scored {n}  mean excess {_f(mean, 4)}  SE {_f(se, 4)}  95% CI "
          f"[{_f(ci[0] if ci else None, 4)}, {_f(ci[1] if ci else None, 4)}]  sd {_f(res['sd'])}")
    print(f"   p90 {_f(res['p90'])}  p99 {_f(res['p99'])}  max {_f(res['max'])}  share <= 0 {_f(res['share_le_0'])}"
          f"  log loss {_f(res['log_loss'], 4)}  misses at stated >= {CONFIDENT_P}: {len(conf_miss)}")

    print("3. calibration (stated bucket: n, mean stated, hit rate)")
    res["buckets"] = []
    for lo, hi in BUCKETS:
        b = [o for o in outs if lo <= o.stated_p < hi]
        row = {"bucket": [lo, min(hi, 1.0)], "n": len(b),
               "stated": statistics.fmean(o.stated_p for o in b) if b else None,
               "hit": statistics.fmean(1.0 if o.terms.happened else 0.0 for o in b) if b else None,
               "mean_excess": statistics.fmean(o.terms.excess for o in b) if b else None}
        res["buckets"].append(row)
        print(f"   [{lo:.1f}, {min(hi, 1.0):.1f}): n {row['n']:4d}  stated {_f(row['stated'])}  hit {_f(row['hit'])}"
              f"  mean excess {_f(row['mean_excess'], 4)}")

    print("4. identity (#3388)")
    res["key_age"] = []
    groups = [(f"individual, key age [{lo:g}, {hi:g}) d", [o for o in outs if o.subject_kind == "individual"
                                                          and o.key_age_days is not None
                                                          and lo <= o.key_age_days < hi])
              for lo, hi in KEY_AGE_BUCKETS]
    groups.append(("label (a class, no identity)", [o for o in outs if o.subject_kind == "label"]))
    for name, g in groups:
        row = {"group": name, "n": len(g),
               "mean_excess": statistics.fmean(o.terms.excess for o in g) if g else None,
               "miss_rate": statistics.fmean(0.0 if o.terms.happened else 1.0 for o in g) if g else None}
        res["key_age"].append(row)
        print(f"   {name:36s} n {row['n']:4d}  mean excess {_f(row['mean_excess'], 4)}  miss {_f(row['miss_rate'])}")
    by_key: dict[str, list] = defaultdict(list)
    for o in outs:
        if o.subject_kind == "individual":
            by_key[o.subject_key].append(o)
    lifetimes = []
    for key, g in by_key.items():
        g.sort(key=lambda o: o.window_start)
        span = None
        if sightings is not None:
            span = sightings.get(key.split(":", 1)[1])
        else:
            r0 = next((r for r in rows if r["subject_key"] == key and r.get("key_last_seen_at")), None)
            if r0 is not None and r0.get("first_seen_before") is not None:
                span = (r0["first_seen_before"], r0["key_last_seen_at"])
        if span:
            lifetimes.append((span[1] - span[0]).total_seconds() / 86400.0)
    # How long each key was FORECAST inside this window (first to last expectation window): a short span
    # with a long sighting life is a key that stopped being predicted, the re-key / departure case.
    spans = [(g[-1].window_start - g[0].window_start).total_seconds() / 86400.0 for g in by_key.values()]
    tail = [o for g in by_key.values() for o in g[-3:]]
    head = [o for g in by_key.values() for o in g[:-3]]
    res["identity"] = {
        "individual_keys": len(by_key), "keys_with_lifetime": len(lifetimes),
        "lived_under_10d": sum(1 for d in lifetimes if d < 10.0),
        "forecast_span_under_10d": sum(1 for d in spans if d < 10.0),
        "last3_miss_rate": statistics.fmean(0.0 if o.terms.happened else 1.0 for o in tail) if tail else None,
        "rest_miss_rate": statistics.fmean(0.0 if o.terms.happened else 1.0 for o in head) if head else None,
        "last3_mean_excess": statistics.fmean(o.terms.excess for o in tail) if tail else None,
        "rest_mean_excess": statistics.fmean(o.terms.excess for o in head) if head else None,
    }
    i = res["identity"]
    print(f"   individual keys {i['individual_keys']}  sighting life < 10 d {i['lived_under_10d']} of {i['keys_with_lifetime']}"
          f"  forecast span in window < 10 d {i['forecast_span_under_10d']}"
          f"  last-3 miss {_f(i['last3_miss_rate'])} (excess {_f(i['last3_mean_excess'], 4)})"
          f"  rest miss {_f(i['rest_miss_rate'])} (excess {_f(i['rest_mean_excess'], 4)})")

    samples = [fold.RhythmSample(expectation_id=o.expectation_id, scored_at=o.scored_at, excess=o.terms.excess,
                                 calm_var=o.terms.calm_var, happened=o.terms.happened, p=o.terms.p) for o in outs]
    grid = fold.replay_node_values(samples)
    vals = [v for _, v in grid]
    span_days = max((until - since).total_seconds() / 86400.0, 1e-9)
    fresh_sec = 0.0
    for k, (t, _) in enumerate(grid):
        nxt = grid[k + 1][0] if k + 1 < len(grid) else until
        fresh_sec += max(0.0, min(600.0, (nxt - t).total_seconds()))
    res["node"] = {
        "writes": len(grid), "fresh_share": fresh_sec / max((until - since).total_seconds(), 1e-9),
        "exact0_share": (sum(1 for v in vals if v == 0.0) / len(vals)) if vals else None,
        "p90": sr.quantile(vals, 0.9), "p99": sr.quantile(vals, 0.99), "max": max(vals) if vals else None,
        "writes_per_7d": len(vals) * 7.0 / span_days,
    }
    rates = [calm_crossing_rate([o.terms.p for o in outs if t - fold.WINDOW <= o.scored_at <= t], SEED_BAR)
             for t, _ in grid]
    res["node"]["calm_crossing_at_seed_bar"] = statistics.fmean(rates) if rates else None
    res["node"]["observed_crossing_at_seed_bar"] = (sum(1 for v in vals if v >= SEED_BAR) / len(vals)) if vals else None
    nd = res["node"]
    print(f"5. node (event writes, 30 min window): writes {nd['writes']}  fresh {_f(nd['fresh_share'])}  exact 0 {_f(nd['exact0_share'])}  p90 {_f(nd['p90'])}"
          f"  p99 {_f(nd['p99'])}  max {_f(nd['max'])}  writes / 7 d {nd['writes_per_7d']:.0f} (floor {MIN_READINGS})")
    print(f"   curiosity bar {SEED_BAR:.3f}: calm crossing (exact, per window) {_f(nd['calm_crossing_at_seed_bar'], 4)}"
          f"  observed {_f(nd['observed_crossing_at_seed_bar'], 4)}")

    if signals:
        by_node: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        for r in signals:
            by_node[r["node_id"]].append((r["observed_at"], float(r["value"])))

        def window_stat(node: str, a: datetime, b: datetime, agg):
            xs = [v for t, v in by_node.get(node, []) if a <= t < b]
            return agg(xs) if xs else None

        res["independence"] = {}
        for node, agg, label in ((WALKWAY, statistics.fmean, "walkway scene change (mean)"),
                                 (STREET, max, "street alarm (fired)")):
            xs, ys = [], []
            for t, v in grid:
                s = window_stat(node, t - fold.WINDOW, t, agg)
                if s is not None:
                    xs.append(v)
                    ys.append(s)
            ox, oy = [], []
            for o in outs:
                s = window_stat(node, o.window_start, o.window_end, agg)
                if s is not None:
                    ox.append(o.terms.excess)
                    oy.append(s)
            res["independence"][node] = {"node_r": _pearson(xs, ys), "node_n": len(xs),
                                         "outcome_r": _pearson(ox, oy), "outcome_n": len(ox)}
            d = res["independence"][node]
            print(f"6. vs {label:30s} node r {_f(d['node_r'])} (n {d['node_n']})  outcome r {_f(d['outcome_r'])}"
                  f" (n {d['outcome_n']})")

    fails = []
    if n < MIN_SCORED:
        fails.append(f"only {n} scored outcomes (< {MIN_SCORED})")
    if ci is None or not (ci[0] <= 0.0 <= ci[1]):
        fails.append(f"mean excess CI {ci} does not cover 0")
    if res["p99"] is None or res["p99"] > GATE_P99:
        fails.append(f"p99 {res['p99']} > {GATE_P99}")
    if res["max"] is None or res["max"] > GATE_MAX:
        fails.append(f"max {res['max']} > {GATE_MAX}")
    if len(conf_miss) > GATE_CONFIDENT_MISSES:
        fails.append(f"{len(conf_miss)} misses at stated >= {CONFIDENT_P} (> {GATE_CONFIDENT_MISSES})")
    res["gate_failures"] = fails
    print("GATE " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--until", type=str, default=None, help="ISO end of the window (default now)")
    ap.add_argument("--fixture", type=str, default=None)
    ap.add_argument("--export", type=str, default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.fixture:
        rows, signals, sightings, since, until = load_fixture(a.fixture)
    else:
        rows, signals, sightings, since, until = load_live(a.days, _ts(a.until))
        if a.export:
            export_fixture(a.export, rows, sightings, since, until)
    res = run(rows, signals, sightings, since, until)
    if a.json:
        print(json.dumps(res, default=str, indent=1))
    return 1 if a.check and res["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
