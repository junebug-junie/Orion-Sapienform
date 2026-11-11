"""Juniper reachable: will Juniper send Orion a message in the next hour?

Spec: docs/superpowers/specs/2026-11-10-juniper-reachable-forecast-design.md (prediction-error suite,
family B). No consumer yet; outreach timing is the intended one, in a later patch.

One forecast per hour, issued at the top of the hour for the window ``[origin, origin + 1 h)``. The
outcome arrives on its own: a message from Juniper in the window, or none. A message is a
``chat_message`` row with ``role = 'user'`` outside Orion's own sessions (``store.JUNIPER_MESSAGE_SQL``).
Its ``timestamp`` is when the turn landed, not when she hit send (user and reply rows are stamped
together): "hour" means the hour the turn landed. Training and settling use the same stamp.

Every model is fit from scratch at each origin, from data strictly before the origin. Training rows
are the past hours; weight = 0.5 ** (age / 60 days).

    reachable_base_rate       baseline. Weighted rate over every past hour, smoothed toward 2 %.
    reachable_state           Markov state only: hours since her last message, 5 buckets, backed off
                              to the base rate.
    reachable_activity_logit  contender. L2 logistic regression (lambda 1) on:
        state buckets          <1 h, 1-3, 3-12, 12-48 (reference: >48 h or never)
        asleep                 03:00-08:00 Denver (her word: asleep by 3; 0 messages in 510 h)
        weekend_morning        Sat/Sun before noon (0 in 361 h)
        evening                18:00-03:00 Denver
        monday                 out of Claude tokens until the reset (0.4x)
        claude_1h / claude_1_3h / no_claude_6h
                               Juniper typed to Claude Code in the last hour / 1-3 h ago / not for
                               6 h: juniper_affective_state_log.message_count, her own typed words
                               only (orion-cocreation-signals). 4x lift overall, 3x in cold hours.
    Office camera and walkway/doorbell presence were tested and dropped (1.2-1.4x, no replay gain).

Grade: log loss per hour. The unit of the test is the local DAY (its mean log loss), because hours
inside a conversation are not independent. Overlap-aware paired test (backtest.paired_win_p) over
>= MIN_DAYS days. Reported as the daily loss gap, its standard error and the chance the model is really
better -- no alpha, no pass/fail. Two subsets:

    all     every hour
    cold    hours with no message in the 3 h before the origin: the question outreach timing needs
            ("will she come back?"). No skill yet (see below).

A Wilcoxon / sign test on daily differences is the wrong test here: most days have no message, the
model wins each of them by a hair, and the sign test calls that significant even where the mean loss
is worse (seen in the replay). The paired test is on the mean.

Served: whichever of the three had the lowest mean log loss over the last 60 days (see ``verdict``).
Replay on typing-covered hours (2026-08-14..11-10): the state model wins overall, and none of the three
beats the base rate in cold hours. Typing to Claude carries 3-4x lift on its own, but it overlaps the
state and the clock (she codes with Claude while chatting with Orion, in the evening).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from app.backtest import paired_gap

logger = logging.getLogger("orion-prediction.reachable")

MODEL = "reachable_activity_logit"
STATE_MODEL = "reachable_state"
BASELINE = "reachable_base_rate"
LOCAL_TZ = ZoneInfo("America/Denver")        # Juniper's clock; the hosts run UTC
WINDOW = timedelta(hours=1)
LOOKBACK = timedelta(days=120)
HALF_LIFE_DAYS = 60.0
K = 8.0                                      # state backoff strength, in weighted hours
L2 = 1.0                                     # logistic ridge penalty (bias unpenalized)
PRIOR_POS, PRIOR_HOURS = 0.5, 25.0           # base-rate prior: 2 % from 25 pseudo-hours
STATE_EDGES_H = (1.0, 3.0, 12.0, 48.0)       # buckets: <1 h, 1-3, 3-12, 12-48, >48 h (or never)
COLD_BUCKET = 2                              # bucket >= 2: no message in the 3 h before the origin
P_CLIP = 1e-4
MIN_DAYS = 20                                # settled local days before a verdict is read
FEATURES = ("state_lt1h", "state_1_3h", "state_3_12h", "state_12_48h", "asleep", "weekend_morning", "evening",
            "monday", "claude_1h", "claude_1_3h", "no_claude_6h")
COVERED_FRAC = 0.9                           # share of the prior 6 h the typing log must cover
CALIBRATION_BINS = (0.0, 0.02, 0.05, 0.15, 0.4, 1.0)
_HOUR_NS = 3_600_000_000_000


def _ns(ts: datetime) -> int:
    return int(ts.astimezone(timezone.utc).timestamp() * 1e9)


def hour_floor(ts: datetime) -> datetime:
    return ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def state_bucket(hours_since: float | None) -> int:
    return int(np.digitize([np.inf if hours_since is None else hours_since], STATE_EDGES_H)[0])


def _local(origins_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(hour, weekday) in Denver, DST-aware."""
    t = pd.DatetimeIndex(pd.to_datetime(origins_ns, utc=True)).tz_convert(LOCAL_TZ)
    return t.hour.to_numpy(), t.dayofweek.to_numpy()


def _count_before(ev_ns: np.ndarray, ev_w: np.ndarray, at: np.ndarray, back_h: float, skip_h: float = 0.0):
    """Sum of ev_w over events in [at - back_h, at - skip_h)."""
    cum = np.concatenate([[0.0], np.cumsum(ev_w)])
    hi = np.searchsorted(ev_ns, at - int(skip_h * _HOUR_NS), side="left")
    lo = np.searchsorted(ev_ns, at - int(back_h * _HOUR_NS), side="left")
    return cum[hi] - cum[lo]


def features(hours: np.ndarray, msg_ns: np.ndarray, typed_ns: np.ndarray,
             typed_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(feature matrix len(hours) x len(FEATURES), state bucket per hour), from events strictly
    before each hour."""
    if len(msg_ns):
        i_last = np.searchsorted(msg_ns, hours, side="left") - 1
        since = np.where(i_last >= 0, (hours - msg_ns[np.maximum(i_last, 0)]) / _HOUR_NS, np.inf)
    else:
        since = np.full(len(hours), np.inf)
    b = np.digitize(since, STATE_EDGES_H)
    hr, wd = _local(hours)
    c1 = _count_before(typed_ns, typed_n, hours, 1.0) > 0
    c13 = _count_before(typed_ns, typed_n, hours, 3.0, 1.0) > 0
    c6 = _count_before(typed_ns, typed_n, hours, 6.0) > 0
    cols = [b == 0, b == 1, b == 2, b == 3, (hr >= 3) & (hr < 8), (wd >= 5) & (hr < 12),
            (hr >= 18) | (hr < 3), wd == 0, c1, c13 & ~c1, ~c6]
    return np.column_stack(cols).astype(float), b


def fit_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = L2) -> np.ndarray | None:
    """Weighted ridge logistic regression; [bias, *coefs], or None when the optimiser fails."""
    Xb = np.column_stack([np.ones(len(X)), X])

    def f(beta):
        z = Xb @ beta
        # log(1 + e^z) - y z: exact log loss, no overflow, no clip
        loss = (w * (np.logaddexp(0.0, z) - y * z)).sum() + l2 * (beta[1:] @ beta[1:])
        g = Xb.T @ (w * (expit(z) - y))
        g[1:] += 2 * l2 * beta[1:]
        return loss, g

    res = minimize(f, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B")
    if not res.success:
        logger.warning("reachable_logit_fit_failed %s", res.message)
        return None
    return res.x


def _covered(intervals: Sequence[tuple[datetime, datetime]]):
    """Function (a_ns, b_ns arrays) -> seconds of [a, b) covered by the union of the intervals."""
    iv = sorted((_ns(a), _ns(b)) for a, b in intervals if b > a)
    merged: list[list[int]] = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    starts = np.array([a for a, _ in merged], dtype=np.int64)
    ends = np.array([b for _, b in merged], dtype=np.int64)
    cum = np.concatenate([[0], np.cumsum(ends - starts)]).astype(float)

    def upto(t: np.ndarray) -> np.ndarray:      # covered ns in (-inf, t)
        i = np.searchsorted(starts, t, side="right")          # intervals starting before t
        full = cum[np.maximum(i - 1, 0)]
        part = np.where(i > 0, np.minimum(t, ends[np.maximum(i - 1, 0)]) - starts[np.maximum(i - 1, 0)], 0)
        return np.where(i > 0, full + part, 0.0)

    return lambda a, b: (upto(b) - upto(a)) / 1e9


def _ns_sorted(ts: Iterable[datetime], before_ns: int) -> np.ndarray:
    a = np.array([_ns(t) for t in ts], dtype=np.int64)
    return np.sort(a[a < before_ns]) if len(a) else np.zeros(0, dtype=np.int64)


def forecast(messages: Sequence[datetime], origin: datetime,
             typed: Sequence[tuple[datetime, int]] = (),
             coverage: Sequence[tuple[datetime, datetime]] | None = None) -> dict:
    """p of a message in ``[origin, origin + 1 h)`` from data strictly before ``origin``.

    ``typed``: (window_until, message_count) rows of Juniper's typed Claude Code messages; each count
    is placed at its window's end, so nothing typed after the origin can leak in. "claude_1h" therefore
    means "a ~15 min window ended in the last hour" (typing up to ~75 min ago; 1 h cold-start windows).
    A live forecast can miss a window that lands late (3 of 1,482 rows landed > 10 min late).

    ``coverage``: (window_since, window_until) of EVERY typing-log row, zero counts included. An hour
    whose prior 6 h the log covers < 90 % (before the producer existed, 2026-08-14, or an outage) does
    not know whether she typed: it is left out of training, and when the origin itself is uncovered the
    served p is the state model's. None = treat everything as covered (tests)."""
    origin = hour_floor(origin)
    o_ns = _ns(origin)
    ts = _ns_sorted(messages, o_ns)
    tw = sorted((_ns(t), int(n)) for t, n in typed if _ns(t) < o_ns and int(n) > 0)
    typed_ns = np.array([t for t, _ in tw], dtype=np.int64)
    typed_n = np.array([n for _, n in tw], dtype=float)
    x0 = features(np.array([o_ns], dtype=np.int64), ts, typed_ns, typed_n)[0][0]
    last = ts[-1] if len(ts) else None
    hours_since = None if last is None else (o_ns - last) / _HOUR_NS
    bucket = state_bucket(hours_since)
    out = {"origin": origin, "window_end": origin + WINDOW, "hours_since": hours_since, "state_bucket": bucket,
           "local_hour": datetime.fromtimestamp(o_ns / 1e9, tz=LOCAL_TZ).hour, "n_messages": int(len(ts)),
           "features": {k: bool(v) for k, v in zip(FEATURES, x0)}}
    if not len(ts):
        p0 = PRIOR_POS / PRIOR_HOURS
        return {**out, "p": p0, "p_state": p0, "p_baseline": p0}
    # Fixed window, not "from the first message": live reads exactly LOOKBACK of messages, and the
    # replay must train on the same rows (review 2026-11-10).
    start = o_ns - int(LOOKBACK.total_seconds() * 1e9)
    start -= start % _HOUR_NS
    hours = np.arange(start, o_ns, _HOUR_NS, dtype=np.int64)
    # outcome per past hour: any message in [h, h + 1 h); every hour here ends at or before the origin
    y = (np.searchsorted(ts, hours + _HOUR_NS, side="left") > np.searchsorted(ts, hours, side="left")).astype(float)
    X, b = features(hours, ts, typed_ns, typed_n)
    w = 0.5 ** (((o_ns - hours) / (24 * _HOUR_NS)) / HALF_LIFE_DAYS)
    base = float((w @ y + PRIOR_POS) / (w.sum() + PRIOR_HOURS))
    m = b == bucket
    p_state = float((w[m] @ y[m] + K * base) / (w[m].sum() + K))
    if coverage is None:
        known, origin_known = np.ones(len(hours), bool), True
    else:
        cov = _covered(coverage)
        six = 6 * _HOUR_NS
        known = cov(hours - six, hours) >= COVERED_FRAC * six / 1e9
        origin_known = bool(cov(np.array([o_ns - six]), np.array([o_ns]))[0] >= COVERED_FRAC * six / 1e9)
    out["typing_log_covered"] = origin_known
    if not origin_known or y[known].sum() == 0:
        return {**out, "p": p_state, "p_state": p_state, "p_baseline": base}
    beta = fit_logit(X[known], y[known], w[known])
    if beta is None:
        return {**out, "p": p_state, "p_state": p_state, "p_baseline": base}
    p = float(expit(beta[0] + x0 @ beta[1:]))
    return {**out, "p": p, "p_state": p_state, "p_baseline": base}


def outcome(messages: Iterable[datetime], origin: datetime) -> tuple[bool, int]:
    end = origin + WINDOW
    n = sum(1 for m in messages if origin <= m < end)
    return n > 0, n


def log_loss(p: float, happened: bool) -> float:
    p = min(1 - P_CLIP, max(P_CLIP, float(p)))
    return -float(np.log(p if happened else 1 - p))


def _day_means(rows: Sequence[dict], key: str) -> dict:
    by: dict = {}
    for r in rows:
        day = r["origin"].astimezone(LOCAL_TZ).date()
        by.setdefault(day, []).append(log_loss(r[key], r["happened"]))
    return {d: float(np.mean(v)) for d, v in by.items()}


def _gap(cand: np.ndarray, ref: np.ndarray) -> dict:
    """Daily mean log-loss gap (cand - ref; negative = cand better), its overlap-aware standard error,
    and the chance cand is really better, P(true gap < 0)
    under a flat prior with the same t approximation (backtest.paired_gap). No threshold is applied here."""
    from scipy import stats

    mean, se, n = paired_gap(cand, ref, lag=1)
    if mean is None:
        return {"gap_per_day": None, "gap_se": None, "prob_better": None}
    prob = (1.0 if mean < 0 else 0.0) if se == 0 else float(stats.t.cdf(-mean / se, df=n - 1))
    return {"gap_per_day": mean, "gap_se": se, "prob_better": prob}


def _subset_verdict(rows: Sequence[dict]) -> dict:
    model, base = _day_means(rows, "p"), _day_means(rows, "p_baseline")
    state = _day_means(rows, "p_state")
    days = sorted(model)
    m = np.array([model[d] for d in days]); b = np.array([base[d] for d in days])
    s = np.array([state[d] for d in days])
    n_pos = int(sum(1 for r in rows if r["happened"]))
    ll = lambda key: float(np.mean([log_loss(r[key], r["happened"]) for r in rows])) if rows else None  # noqa: E731
    out = {"n_hours": len(rows), "n_days": len(days), "n_hours_with_message": n_pos,
           "log_loss": {"model": ll("p"), "state": ll("p_state"), "baseline": ll("p_baseline")},
           "skill_vs_baseline": None, "vs_baseline": None, "activity_vs_state": None}
    if rows and out["log_loss"]["baseline"]:
        out["skill_vs_baseline"] = 1 - out["log_loss"]["model"] / out["log_loss"]["baseline"]
    if len(days) >= MIN_DAYS and n_pos:
        out["vs_baseline"] = _gap(m, b)
        out["activity_vs_state"] = _gap(m, s)    # do the clock rules and Claude typing add anything?
    return out


def calibration(rows: Sequence[dict], key: str = "p") -> list[dict]:
    out = []
    for lo, hi in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
        sel = [r for r in rows if lo <= r[key] < hi or (hi == 1.0 and r[key] == 1.0)]
        if sel:
            out.append({"bin": [lo, hi], "n": len(sel), "mean_p": float(np.mean([r[key] for r in sel])),
                        "observed_rate": float(np.mean([r["happened"] for r in sel]))})
    return out


def verdict(rows: Iterable[dict]) -> dict:
    """``rows``: settled forecasts {origin, p, p_state, p_baseline, state_bucket, happened}.

    Reports how much better the model is and how sure that is; it draws no pass/fail line (any alpha
    would be a convention, not a cost). Served: whichever had the lowest mean log
    loss over the window among model, state and baseline, once MIN_DAYS days exist (the minimum for
    the variance estimate); before that, the model. A consumer that acts on the forecast sets its own bar on ``prob_better`` from the
    cost of acting wrongly."""
    rows = [r for r in rows if r.get("happened") is not None]
    out = {"all": _subset_verdict(rows), "cold": _subset_verdict([r for r in rows if r["state_bucket"] >= COLD_BUCKET])}
    out["calibration"] = calibration(rows)
    out["served"] = MODEL
    if out["all"]["n_days"] >= MIN_DAYS and out["all"]["n_hours_with_message"]:
        ll = out["all"]["log_loss"]
        out["served"] = min(((ll["model"], MODEL), (ll["state"], STATE_MODEL), (ll["baseline"], BASELINE)))[1]
    return out


def served_p(row: dict, served: str) -> float:
    return {BASELINE: row["p_baseline"], STATE_MODEL: row["p_state"]}.get(served, row["p"])


ISSUE_SLACK = timedelta(minutes=10)    # a forecast is issued only in the first 10 min of its hour
SETTLE_GRACE = timedelta(minutes=10)   # chat rows land via the hub; wait before reading the window
VERDICT_DAYS = 60


class Reachable:
    """Issues the hourly forecast and settles past ones. Store: app/store.py (reachable section)."""

    def __init__(self, store):
        self.store = store
        self.last_pass: dict = {}

    def _verdict(self, now: datetime) -> dict:
        return verdict(self.store.reachable_rows(now - timedelta(days=VERDICT_DAYS), settled_only=True))

    def tick(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        origin = hour_floor(now)
        issued = None
        if now - origin <= ISSUE_SLACK and not self.store.reachable_exists(origin):
            msgs = self.store.juniper_messages(origin - LOOKBACK, origin)
            typed = self.store.juniper_claude_typed(origin - LOOKBACK, origin)
            coverage = self.store.juniper_typing_coverage(origin - LOOKBACK - timedelta(hours=6), origin)
            f = forecast(msgs, origin, typed, coverage)
            served = self._verdict(now)["served"]
            row = {**f, "forecast_id": f"reachable:{origin.strftime('%Y-%m-%dT%H')}", "issued_at": now,
                   "model": MODEL, "served": served, "settled_at": None, "happened": None, "n_in_window": None}
            issued = self.store.insert_reachable(row)
        settled = 0
        for r in self.store.pending_reachable(now - SETTLE_GRACE):
            msgs = self.store.juniper_messages(r["origin"], r["window_end"])
            happened, n = outcome(msgs, r["origin"])
            settled += self.store.settle_reachable(r["forecast_id"], happened, n, now)
        self.last_pass = {"at": now, "issued": issued, "settled": settled}
        return self.last_pass

    def live(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        latest = self.store.reachable_rows(now - timedelta(hours=2), settled_only=False)
        current = max(latest, key=lambda r: r["origin"]) if latest else None
        v = self._verdict(now)
        if current is not None:
            current = {**current, "served_p": served_p(current, current["served"])}
        return {"model": MODEL, "baseline": BASELINE, "current": current, "grade": v,
                "window_days": VERDICT_DAYS}
