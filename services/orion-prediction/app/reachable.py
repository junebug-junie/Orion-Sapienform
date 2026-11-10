"""Juniper reachable: will Juniper send Orion a message in the next hour?

Spec: docs/superpowers/specs/2026-11-10-juniper-reachable-forecast-design.md (prediction-error suite,
family B). No consumer yet; outreach timing is the intended one, in a later patch.

One forecast per hour, issued at the top of the hour for the window ``[origin, origin + 1 h)``. The
outcome arrives on its own: a message from Juniper in the window, or none. A message is a
``chat_message`` row with ``role = 'user'`` outside Orion's own sessions (``store.JUNIPER_MESSAGE_SQL``).

Every model is fit from scratch at each origin, from messages strictly before the origin. Training
rows are the past hours: outcome = any message in that hour; weight = 0.5 ** (age / 14 days), so the
last two weeks count most (her chat rate moved ~10x in late October).

    reachable_base_rate     baseline. Weighted rate over every past hour, smoothed toward 2 %.
    reachable_state         Markov state only: hours since her last message, 5 buckets.
    reachable_markov_tod    served model. State x local hour of day (America/Denver), backed off to
                            the state rate with strength K, which is backed off to the base rate.

Grade: log loss per hour. The unit of the test is the local DAY (its mean log loss), because hours
inside a conversation are not independent. Overlap-aware paired test (backtest.paired_win_p) over
>= MIN_DAYS days, run two ways:

    all     every hour
    cold    hours with no message in the 3 h before the origin. This is the question outreach timing
            needs ("will she come back?"), and the replay of 07-24..11-10 says the model has NO skill
            here yet. Mid-conversation hours carry all of the overall win.

A Wilcoxon / sign test on daily differences is the wrong test here: most days have no message, the
model wins each of them by a hair, and the sign test calls that significant even where the mean loss
is worse (seen in the replay). The paired test is on the mean.

Served: the model, unless the ``all`` verdict says it is significantly WORSE than the baseline.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from app.backtest import paired_win_p
from app.live import LIVE_ALPHA, holm

MODEL = "reachable_markov_tod"
STATE_MODEL = "reachable_state"
BASELINE = "reachable_base_rate"
LOCAL_TZ = ZoneInfo("America/Denver")        # Juniper's clock; the hosts run UTC
WINDOW = timedelta(hours=1)
LOOKBACK = timedelta(days=120)
HALF_LIFE_DAYS = 14.0
K = 8.0                                      # backoff strength, in weighted hours
PRIOR_POS, PRIOR_HOURS = 0.5, 25.0           # base-rate prior: 2 % from 25 pseudo-hours
STATE_EDGES_H = (1.0, 3.0, 12.0, 48.0)       # buckets: <1 h, 1-3, 3-12, 12-48, >48 h (or never)
COLD_BUCKET = 2                              # bucket >= 2: no message in the 3 h before the origin
P_CLIP = 1e-4
MIN_DAYS = 20                                # settled local days before a verdict is read
CALIBRATION_BINS = (0.0, 0.02, 0.05, 0.15, 0.4, 1.0)
_HOUR_NS = 3_600_000_000_000


def _ns(ts: datetime) -> int:
    return int(ts.astimezone(timezone.utc).timestamp() * 1e9)


def hour_floor(ts: datetime) -> datetime:
    return ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def state_bucket(hours_since: float | None) -> int:
    return int(np.digitize([np.inf if hours_since is None else hours_since], STATE_EDGES_H)[0])


def _local_hours(origins_ns: np.ndarray) -> np.ndarray:
    # Hour of day in Denver, DST-aware: offsets come from zoneinfo per origin.
    out = np.empty(len(origins_ns), dtype=np.int64)
    for i, o in enumerate(origins_ns):
        out[i] = datetime.fromtimestamp(o / 1e9, tz=LOCAL_TZ).hour
    return out


def forecast(messages: Sequence[datetime], origin: datetime) -> dict:
    """p of a message in ``[origin, origin + 1 h)`` from messages strictly before ``origin``."""
    origin = hour_floor(origin)
    o_ns = _ns(origin)
    ts = np.sort(np.array([_ns(m) for m in messages if m < origin], dtype=np.int64))
    last = ts[-1] if len(ts) else None
    hours_since = None if last is None else (o_ns - last) / _HOUR_NS
    bucket = state_bucket(hours_since)
    local_hour = datetime.fromtimestamp(o_ns / 1e9, tz=LOCAL_TZ).hour
    out = {"origin": origin, "window_end": origin + WINDOW, "hours_since": hours_since, "state_bucket": bucket,
           "local_hour": local_hour, "n_messages": int(len(ts))}
    if not len(ts):
        p0 = PRIOR_POS / PRIOR_HOURS
        return {**out, "p": p0, "p_state": p0, "p_baseline": p0}
    start = max(ts[0] - ts[0] % _HOUR_NS, o_ns - int(LOOKBACK.total_seconds() * 1e9))
    start -= start % _HOUR_NS
    hours = np.arange(start, o_ns, _HOUR_NS, dtype=np.int64)
    # outcome per past hour: any message in [h, h + 1 h)
    y = (np.searchsorted(ts, hours + _HOUR_NS, side="left") > np.searchsorted(ts, hours, side="left")).astype(float)
    i_last = np.searchsorted(ts, hours, side="left") - 1           # last message strictly before h
    since = np.where(i_last >= 0, (hours - ts[np.maximum(i_last, 0)]) / _HOUR_NS, np.inf)
    b = np.digitize(since, STATE_EDGES_H)
    hod = _local_hours(hours)
    w = 0.5 ** (((o_ns - hours) / (24 * _HOUR_NS)) / HALF_LIFE_DAYS)
    base = float((w @ y + PRIOR_POS) / (w.sum() + PRIOR_HOURS))
    m = b == bucket
    p_state = float((w[m] @ y[m] + K * base) / (w[m].sum() + K))
    m2 = m & (hod == local_hour)
    p = float((w[m2] @ y[m2] + K * p_state) / (w[m2].sum() + K))
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
           "skill_vs_baseline": None, "p_better": None, "p_worse": None, "p_time_of_day_helps": None,
           "verdict": "pending"}
    if rows and out["log_loss"]["baseline"]:
        out["skill_vs_baseline"] = 1 - out["log_loss"]["model"] / out["log_loss"]["baseline"]
    if len(days) >= MIN_DAYS and n_pos:
        out["p_better"], _ = paired_win_p(m, b, lag=1)
        out["p_worse"], _ = paired_win_p(b, m, lag=1)
        out["p_time_of_day_helps"], _ = paired_win_p(m, s, lag=1)   # served model vs state-only
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
    """``rows``: settled forecasts {origin, p, p_state, p_baseline, state_bucket, happened}."""
    rows = [r for r in rows if r.get("happened") is not None]
    out = {"all": _subset_verdict(rows), "cold": _subset_verdict([r for r in rows if r["state_bucket"] >= COLD_BUCKET])}
    tested = {k: v["p_better"] for k, v in out.items() if v["p_better"] is not None}
    adj = holm(tested)
    for k, v in out.items():
        if v["p_better"] is None:
            continue
        v["p_better_adjusted"] = adj[k]
        v["verdict"] = ("better" if adj[k] < LIVE_ALPHA else "worse" if v["p_worse"] < LIVE_ALPHA else "unclear")
    out["calibration"] = calibration(rows)
    out["served"] = BASELINE if out["all"]["verdict"] == "worse" else MODEL
    return out


def served_p(row: dict, served: str) -> float:
    return row["p_baseline"] if served == BASELINE else row["p"]


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
            f = forecast(msgs, origin)
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
        return {"model": MODEL, "baseline": BASELINE, "current": current, "verdict": v,
                "window_days": VERDICT_DAYS}
