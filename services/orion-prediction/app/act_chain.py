"""Act chain: what will Orion do in the next 15 minutes? (prediction-error suite, family B).

Spec: docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md, "B. Markov chains"
(act-after-act sequences; the honest replacement for the retired ``route`` signal and the bus signal's
"service rhythm"). Measurement and eval only: no table, node, bus event or consumer yet.

The stream is Orion's own acts, ``grammar_acts`` (written by orion-sql-writer through
``orion/grammar/act_ledger.py`` from every ``act_opened`` event). Time is cut into 15-minute buckets.
A bucket's STATE is its main verb (most acts; a tie goes to the verb acted last), or ``idle`` when Orion
did nothing. Idle is a real outcome: silence where acts were forecast is scorable, so an outage shows up.

Two views of each bucket, so a replay sees exactly what a live forecaster would have seen:
    asof   acts that had LANDED (``created_at``) by the bucket's end -- what is known when the next
           bucket's forecast is issued. Conditions the chain.
    final  acts landed by bucket end + SETTLE_GRACE (world-pulse reads land up to ~11 min late,
           p99, 2026-11-13). What the bucket is scored against. Rows that land later than that
           (backfills: every ``vision.ask`` act is days late) never count, in training or scoring.

Every model is fit from scratch at each origin, from transitions whose target bucket had settled
before the origin. Weight = 0.5 ** (age / HALF_LIFE_DAYS). Additive (Jeffreys) smoothing on the base,
then each finer level is a Dirichlet posterior mean around the coarser one,
``(count + K * parent) / (n + K)``. K is chosen per level and per origin from the TRAINING transitions
alone, by leave-one-out likelihood over K_GRID (empirical Bayes, MacKay & Peto 1995). A fixed K = 8 was
tried first (replay 2026-11-13): with hundreds of transitions per context it trusted zero counts, the
shrinkage compounded across levels, and a rare act got p ~ 1e-6 (excess surprise +0.21 nats).

    act_marginal       baseline. P(state).
    act_clock_hour     baseline. P(state | local hour), backed off to the marginal.
    act_clock_block    baseline. P(state | Juniper's clock block), backed off to the marginal.
    act_markov         P(state | previous bucket's state), backed off to the marginal.
    act_markov_block   SERVED. P(state | previous state, clock block), backed off to act_markov.

Clock blocks are Juniper's own rules (America/Denver; the hosts run UTC): asleep 03-08, weekend
morning (Sat/Sun 08-12), day 08-18, evening 18-03. They are the structure of the served model, so
thin data moves the per-block numbers but never removes the rule.

Surprise = -log P(observed state). Excess surprise = surprise minus the forecast's own entropy, whose
expected value is exactly 0 when the forecast is calibrated: calm rests at 0 by theory, not tuning.

Grade: log loss per bucket, local DAY as the unit (buckets inside a day are not independent),
overlap-aware paired gap (backtest.paired_gap). Reported as gap, standard error and the chance the
model is really better. No alpha, no pass/fail.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence, get_args
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

from app.backtest import paired_gap
from orion.schemas.grammar import ActVerb

IDLE = "idle"
STATES: tuple[str, ...] = (IDLE, *get_args(ActVerb))
K_STATES = len(STATES)
MODEL = "act_markov_block"
MODELS = ("act_marginal", "act_clock_hour", "act_clock_block", "act_markov", MODEL)
BASELINES = ("act_marginal", "act_clock_hour", "act_clock_block")
BLOCKS = ("asleep", "weekend_morning", "day", "evening")
LOCAL_TZ = ZoneInfo("America/Denver")
GRAIN = timedelta(minutes=15)
SETTLE_GRACE = timedelta(minutes=20)
LOOKBACK = timedelta(days=28)
HALF_LIFE_DAYS = 14.0
ALPHA = 0.5          # additive smoothing on the marginal (Jeffreys)
K_GRID = tuple(2.0 ** np.arange(0, 13))   # backoff strengths tried by LOO: 1 .. 4096
MIN_DAYS = 7         # settled local days before a gap is reported
TOP_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)
_G = int(GRAIN.total_seconds() * 1e9)
_GRACE = int(SETTLE_GRACE.total_seconds() * 1e9)
_DAY = 86_400_000_000_000
_IDX = {s: i for i, s in enumerate(STATES)}


def _ns(ts: datetime) -> int:
    return int(ts.astimezone(timezone.utc).timestamp() * 1e9)


def bucket_floor(ts: datetime) -> datetime:
    n = _ns(ts)
    return datetime.fromtimestamp((n - n % _G) / 1e9, tz=timezone.utc)


def clock(starts_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(local hour, clock block index) per bucket start, DST-aware."""
    t = pd.DatetimeIndex(pd.to_datetime(starts_ns, utc=True)).tz_convert(LOCAL_TZ)
    hr, wd = t.hour.to_numpy(), t.dayofweek.to_numpy()
    blk = np.where((hr >= 3) & (hr < 8), 0,
                   np.where((wd >= 5) & (hr >= 8) & (hr < 12), 1, np.where((hr >= 8) & (hr < 18), 2, 3)))
    return hr, blk


def bucket_states(acts: Sequence[tuple[datetime, datetime, str]], b0_ns: int, n: int,
                  cutoff_ns: np.ndarray) -> np.ndarray:
    """State index for buckets ``b0 + i * GRAIN``, i < n, counting only acts landed by cutoff_ns[i].

    ``acts``: (opened_at, created_at, verb). An unknown verb is dropped (the vocabulary is the grammar's
    closed ActVerb list)."""
    counts = np.zeros((n, K_STATES), dtype=np.int64)
    latest = np.full((n, K_STATES), -1, dtype=np.int64)
    for opened, created, verb in acts:
        j = _IDX.get(verb)
        if j is None or opened is None or created is None:
            continue
        o = _ns(opened)
        i = (o - b0_ns) // _G
        if 0 <= i < n and _ns(created) <= cutoff_ns[i]:
            counts[i, j] += 1
            latest[i, j] = max(latest[i, j], o)
    top = counts.max(axis=1)
    best = np.argmax(np.where(counts == top[:, None], latest, -2), axis=1)
    return np.where(top == 0, _IDX[IDLE], best)


@dataclass
class Stream:
    """Both views of every bucket from b0, plus each bucket's clock."""
    b0_ns: int
    asof: np.ndarray
    final: np.ndarray
    hour: np.ndarray
    block: np.ndarray

    @classmethod
    def build(cls, acts: Sequence[tuple[datetime, datetime, str]], start: datetime, end: datetime) -> "Stream":
        b0 = _ns(bucket_floor(start))
        n = max(0, (_ns(end) - b0) // _G)
        starts = b0 + _G * np.arange(n, dtype=np.int64)
        hr, blk = clock(starts)
        return cls(b0, bucket_states(acts, b0, n, starts + _G), bucket_states(acts, b0, n, starts + _G + _GRACE),
                   hr, blk)

    def index(self, ts: datetime) -> int:
        return (_ns(ts) - self.b0_ns) // _G


def _level(key: np.ndarray, n_keys: int, y: np.ndarray, w: np.ndarray, parent: np.ndarray) -> tuple[np.ndarray, float]:
    """One backoff level: (n_keys x K matrix of P(state | key), chosen K).

    ``parent``: P(state) of the coarser level for each transition (len(y) x K). K maximises the
    weighted leave-one-out log likelihood of the training transitions (each one left out of its own
    count); no held-out data is used."""
    K = K_STATES
    C = np.bincount(key * K + y, w, n_keys * K).reshape(n_keys, K)
    N = C.sum(axis=1)
    c_i, n_i, par_i = C[key, y], N[key], parent[np.arange(len(y)), y]
    best_k, best_ll = K_GRID[0], -np.inf
    for k in K_GRID:
        ll = float(w @ np.log((c_i - w + k * par_i) / (n_i - w + k)))
        if ll > best_ll:
            best_k, best_ll = k, ll
    return C, best_k


def predict(s: Stream, o: int) -> dict:
    """Every model's distribution for bucket ``o`` (an index into ``s``), from data known at its start:
    the condition is asof[o - 1]; training targets t settled by then (t_end + grace <= o_start)."""
    o_ns = s.b0_ns + o * _G
    t_hi = (o_ns - _G - _GRACE - s.b0_ns) // _G + 1           # t < t_hi
    t_lo = max(1, (o_ns - int(LOOKBACK.total_seconds() * 1e9) - s.b0_ns) // _G)
    t = np.arange(t_lo, max(t_lo, t_hi))
    prev_state = int(s.asof[o - 1]) if o >= 1 else _IDX[IDLE]
    blk, hr = int(s.block[o]), int(s.hour[o])
    K = K_STATES
    w = 0.5 ** (((o - t) * _G / _DAY) / HALF_LIFE_DAYS)
    a, y, h, b = s.asof[t - 1], s.final[t], s.hour[t], s.block[t]
    marg = (np.bincount(y, w, K) + ALPHA) / (w.sum() + ALPHA * K)
    if not len(t):
        return {"origin": datetime.fromtimestamp(o_ns / 1e9, tz=timezone.utc), "prev_state": STATES[prev_state],
                "block": BLOCKS[blk], "n_train": 0, "k": {}, "probs": {m: marg for m in MODELS}}
    flat = np.broadcast_to(marg, (len(t), K))
    post = lambda C, k, row, par: (C[row] + k * par) / (C[row].sum() + k)  # noqa: E731
    C_h, k_h = _level(h, 24, y, w, flat)
    C_b, k_b = _level(b, len(BLOCKS), y, w, flat)
    C_a, k_a = _level(a, K, y, w, flat)
    P_a = (C_a + k_a * marg) / (C_a.sum(axis=1, keepdims=True) + k_a)     # P(state | prev), every prev
    C_ab, k_ab = _level(b * K + a, len(BLOCKS) * K, y, w, P_a[a])
    probs = {
        "act_marginal": marg,
        "act_clock_hour": post(C_h, k_h, hr, marg),
        "act_clock_block": post(C_b, k_b, blk, marg),
        "act_markov": P_a[prev_state],
        MODEL: post(C_ab, k_ab, blk * K + prev_state, P_a[prev_state]),
    }
    return {"origin": datetime.fromtimestamp(o_ns / 1e9, tz=timezone.utc), "prev_state": STATES[prev_state],
            "block": BLOCKS[blk], "n_train": int(len(t)),
            "k": {"act_clock_hour": k_h, "act_clock_block": k_b, "act_markov": k_a, MODEL: k_ab}, "probs": probs}


def score(row: dict, observed: int) -> dict:
    """Surprise (nats) and excess surprise (surprise - entropy) per model for one settled bucket."""
    out = {}
    for m, p in row["probs"].items():
        h = float(-(p * np.log(p)).sum())
        sur = float(-np.log(p[observed]))
        out[m] = {"surprise": sur, "entropy": h, "excess": sur - h}
    return out


def forecast(acts: Sequence[tuple[datetime, datetime, str]], origin: datetime) -> dict:
    """Live-shaped forecast for the bucket starting at ``bucket_floor(origin)`` from acts landed before it."""
    origin = bucket_floor(origin)
    s = Stream.build(acts, origin - LOOKBACK - GRAIN, origin + GRAIN)
    row = predict(s, s.index(origin))
    return {**row, "probs": {m: dict(zip(STATES, map(float, p))) for m, p in row["probs"].items()}}


def replay(acts: Sequence[tuple[datetime, datetime, str]], start: datetime, end: datetime,
           warmup: timedelta = timedelta(days=7)) -> list[dict]:
    """Rolling-origin replay: every bucket from start + warmup whose outcome settled by ``end``."""
    s = Stream.build(acts, start, end)
    first = s.index(start + warmup)
    last = (_ns(end) - _GRACE - s.b0_ns) // _G                 # bucket o settled iff o + 1 <= last
    rows = []
    for o in range(max(first, 1), last):
        r = predict(s, o)
        obs = int(s.final[o])
        rows.append({**r, "observed": STATES[obs], "observed_i": obs, "scores": score(r, obs)})
    return rows


def _day(r: dict):
    return r["origin"].astimezone(LOCAL_TZ).date()


def _by_day(rows: Sequence[dict], model: str, field: str) -> dict:
    by: dict = {}
    for r in rows:
        by.setdefault(_day(r), []).append(r["scores"][model][field])
    return {d: float(np.mean(v)) for d, v in by.items()}


def _gap(cand: np.ndarray, ref: np.ndarray) -> dict:
    """Mean daily gap (cand - ref) with its overlap-aware SE and P(true gap < 0). No threshold."""
    mean, se, n = paired_gap(cand, ref, lag=1)
    if mean is None:
        return {"gap_per_day": None, "gap_se": None, "prob_below_zero": None, "n_days": n}
    prob = (1.0 if mean < 0 else 0.0) if se == 0 else float(stats.t.cdf(-mean / se, df=n - 1))
    return {"gap_per_day": mean, "gap_se": se, "prob_below_zero": prob, "n_days": n}


def _daily(rows, model, field):
    d = _by_day(rows, model, field)
    return np.array([d[k] for k in sorted(d)])


def _vs(rows: Sequence[dict], cand: str, ref: str) -> dict:
    g = _gap(_daily(rows, cand, "surprise"), _daily(rows, ref, "surprise"))
    return {"gap_per_day": g["gap_per_day"], "gap_se": g["gap_se"], "prob_better": g["prob_below_zero"],
            "n_days": g["n_days"]}


def excess(rows: Sequence[dict], model: str = MODEL) -> dict:
    """Mean excess surprise (surprise - entropy) with an overlap-aware SE over days. Calibrated -> 0.
    ``prob_below_zero`` near 0.5 is the calm reading; near 0 = the model is overconfident."""
    if not rows:
        return {"mean": None, "se": None, "n_buckets": 0}
    g = _gap(_daily(rows, model, "surprise"), _daily(rows, model, "entropy"))
    return {"mean": g["gap_per_day"], "se": g["gap_se"], "prob_below_zero": g["prob_below_zero"],
            "n_days": g["n_days"], "n_buckets": len(rows)}


def calibration(rows: Sequence[dict], model: str = MODEL) -> dict:
    """Top-state reliability (confidence vs hit rate) and, per state, mean forecast p vs observed share."""
    top_p = np.array([float(r["probs"][model].max()) for r in rows])
    hit = np.array([int(np.argmax(r["probs"][model])) == r["observed_i"] for r in rows])
    bins = []
    for lo, hi in zip(TOP_BINS[:-1], TOP_BINS[1:]):
        m = (top_p >= lo) & (top_p < hi)
        if m.any():
            bins.append({"bin": [lo, min(hi, 1.0)], "n": int(m.sum()), "mean_top_p": float(top_p[m].mean()),
                         "hit_rate": float(hit[m].mean())})
    P = np.array([r["probs"][model] for r in rows])
    obs = np.bincount([r["observed_i"] for r in rows], minlength=K_STATES) / max(len(rows), 1)
    per_state = {s: {"mean_p": float(P[:, i].mean()), "observed": float(obs[i])}
                 for i, s in enumerate(STATES) if P[:, i].mean() > 0.005 or obs[i] > 0}
    return {"top_state": bins, "per_state": per_state}


def rest_check(rows: Sequence[dict], model: str = MODEL, window: int = 4) -> dict:
    """Can the score return to calm? Rolling 1 h (4-bucket) mean excess: its median, share at or below
    0, and the excess inside the asleep block, where Orion's day is quietest."""
    x = np.array([r["scores"][model]["excess"] for r in rows])
    roll = np.convolve(x, np.ones(window) / window, mode="valid") if len(x) >= window else x
    asleep = [r for r in rows if r["block"] == "asleep"]
    return {"rolling_1h_median": float(np.median(roll)) if len(roll) else None,
            "rolling_1h_share_le_0": float((roll <= 0).mean()) if len(roll) else None,
            "rolling_1h_p05_p95": [float(np.percentile(roll, 5)), float(np.percentile(roll, 95))] if len(roll) else None,
            "asleep": excess(asleep, model)}


def verdict(rows: Iterable[dict]) -> dict:
    """Grade of every model against the baselines; served is always MODEL (Juniper's clock rules)."""
    rows = list(rows)
    days = {_day(r) for r in rows}
    out: dict = {"n_buckets": len(rows), "n_days": len(days),
                 "log_loss": {m: float(np.mean([r["scores"][m]["surprise"] for r in rows])) if rows else None
                              for m in MODELS},
                 "served": MODEL}
    best = min(BASELINES, key=lambda m: out["log_loss"][m] if rows else 0.0)
    out["best_baseline"] = best
    out["skill_vs_best_baseline"] = (1 - out["log_loss"][MODEL] / out["log_loss"][best]) if rows else None
    if len(days) >= MIN_DAYS:
        out["vs"] = {f"{MODEL}_vs_{b}": _vs(rows, MODEL, b) for b in BASELINES}
        out["vs"]["act_markov_vs_act_clock_hour"] = _vs(rows, "act_markov", "act_clock_hour")
        out["vs"][f"{MODEL}_vs_act_markov"] = _vs(rows, MODEL, "act_markov")
    out["excess"] = {m: excess(rows, m) for m in MODELS}
    out["calibration"] = calibration(rows) if rows else None
    out["rest"] = rest_check(rows) if rows else None
    return out
