"""What a body signal tends to lead to: next-hour outcome rates when it is high vs normal. Pure.

Spec: docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md section E ("Phase 2: what
it tends to lead to"). Schemas: ``orion/schemas/self_signal_outcome.py``. No LLM, no I/O.

Three steps, each a pure function the producer (orion-equilibrium-service) and the evals share:

1. ``classify_signal_hour``: for one internal ``node:substrate.*`` signal and one UTC hour, judge
   every reading the way the attention seam would have at that moment -- the reading's strict-below
   percentile against the signal's own trailing 7 days (``prediction_error_magnitude``'s rule and
   ``DEFAULT_MIN_READINGS``), then ``world_first.node_candidate`` + ``judge_candidate`` (polarity,
   placeholder kinds, band cuts: the seam's one definition of "high/unusual for itself"). The hour
   is ``high`` when at least ``HIGH_HOUR_SHARE`` of its judged readings were eligible, ``normal``
   otherwise, and ``unknown`` when nothing in it could be judged (no rows, too little history, a
   placeholder): silence is never counted as normal. Trigger kinds (cabinet) are ``high`` when the
   trigger turned on in the hour (``self_readings.trigger_onsets``).
2. ``turn_outcome_hours`` / ``rpc_outcome_hours``: outcome counts per hour from rows Orion already
   records (harness_turn_trace run artifacts, transport_baseline_hourly).
3. ``associate``: pair hour H's condition with hour H+1's outcome (only where H+1 had exposure),
   count next hours with at least one event, and report the rate ratio with a Katz 95% interval.
   The unit is the hour, not the event: failures and timeouts come in bursts (one stuck hop logged
   984 timeouts), so an event-count Poisson interval would be badly overconfident.

A clause ("when this is high, my next hour had a failed turn 3.1x as often ...") exists only when
both sides have at least ``min_hours`` hours AND the interval excludes 1. Pairs whose signal is
computed from, or shares an upstream with, the outcome are ``excluded_circular`` (see
``CIRCULAR_PAIRS``) and never get a clause.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence

from orion.metacog.self_signal_episode import dwell_met
from orion.schemas.self_signal_outcome import (
    MAX_SIGNALS_PER_HOUR,
    OutcomeHourV1,
    SelfSignalHourV1,
    SelfSignalOutcomeAssociationV1,
    SignalHourV1,
)

HOUR = timedelta(hours=1)
WINDOW_7D = timedelta(days=7)
# A high hour = at some poll in the hour the self_signal_episode gate's own open dwell was met: hot
# (high/unusual for itself) for MORE than HIGH_HOUR_HOT_SEC within the trailing OPEN_WINDOW_SEC,
# counting polls every POLL_SEC. The same numbers as the gate (self_signal_episode.DEFAULT_*;
# equality pinned by a test), anchored on world_first.EVENT_ORIENTING_WINDOW_SEC: unusual for
# longer than one event can explain. Not "any hot reading", and not "hot time > 300 s": "high" is
# the top decile of a signal's own week, so a per-tick signal spends ~360 s of every typical hour
# there, scattered; only a sustained stretch counts.
HIGH_HOUR_HOT_SEC = 300.0
OPEN_WINDOW_SEC = 600.0
POLL_SEC = 30.0
# Both sides need at least this many hours before anything is said. PR report: chosen from live
# data so a 95% Katz interval on hour-level rates is not dominated by one or two hours.
DEFAULT_MIN_HOURS = 24
Z_95 = 1.959963984540054
# Hard caps: hours folded into one association (90 days), rows per input batch.
MAX_HOURS = 24 * 90
MAX_ROWS = 500_000

OUTCOMES: tuple[str, ...] = ("turn_failed", "response_repair", "rpc_timeout")

TURN_SOURCE = "harness_turn_trace.run_artifact: compliance_verdict='failed' / response_repair_ran=true, by created_at hour"
RPC_SOURCE = "transport_baseline_hourly: timeout_count>0 over success_count+timeout_count, keys not '#gpu_pool_wait'"
# A GPU-pool queue wait is not an RPC: it times out when the GPU is busy, by design.
RPC_EXCLUDED_KEY_SUFFIX = "#gpu_pool_wait"

# Signal computed from, or sharing a short upstream with, the outcome: an association would be the
# outcome measuring itself. Each entry names the producing code (metric quality gate step 2).
CIRCULAR_PAIRS: dict[tuple[str, str], str] = {
    ("node:substrate.execution", "turn_failed"): (
        "execution's surprise is computed from the same harness runs' failure_pressure and "
        "execution_friction (orion/substrate/execution_loop/grammar_extract.py compute_pressure_hints)"
    ),
    ("node:substrate.execution", "response_repair"): (
        "execution's surprise is computed from the same harness runs whose reply was repaired "
        "(orion/substrate/execution_loop/grammar_extract.py compute_pressure_hints: step and reasoning load)"
    ),
    ("node:substrate.chat", "response_repair"): (
        "chat's surprise includes repair_pressure_level, which also forces the reply judge and the "
        "repair overlay (orion/harness/finalize.py sensitive_turn_reason)"
    ),
    ("node:substrate.bus_synaptic", "rpc_timeout"): (
        "both read the same bus traffic: a timed-out RPC is a missing reply on a bus-synaptic edge "
        "(orion/substrate/prediction_error.py bus_synaptic_prediction_error: inter-arrival gap z-scores)"
    ),
}

_OUTCOME_WORDS = {
    "turn_failed": "a failed turn",
    "response_repair": "a reply I had to repair",
    "rpc_timeout": "an RPC timeout",
}


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def hour_floor(ts: datetime) -> datetime:
    return _aware(ts).astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def hour_id(hour_start: datetime) -> str:
    return f"ssh:{hour_floor(hour_start).strftime('%Y-%m-%dT%H:%M:%SZ')}"


def _z(ts: datetime) -> str:
    return _aware(ts).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


# ---------------------------------------------------------------------------
# 1. Signal condition per hour
# ---------------------------------------------------------------------------


def _judge_reading(node_id: str, value: float, ts: datetime, pct: float, n: int, min_readings: int):
    """The attention seam's verdict on one reading, as of the moment it was written."""
    from orion.attention.world_first import _band_for, judge_candidate, node_candidate
    from orion.schemas.attention_frame import PredictionErrorMagnitudeV1
    from orion.substrate.prediction_error_magnitude import DEFAULT_BAND_CUTS

    band = "insufficient_history" if n < max(1, min_readings) else _band_for(pct, DEFAULT_BAND_CUTS)
    mag = PredictionErrorMagnitudeV1(
        value=round(value, 6), age_sec=0.0, percentile_now=round(pct, 6), n_readings_7d=n,
        band=band, trend="insufficient_history",
    )
    cand = node_candidate(node_id=node_id, label=node_id, magnitude=mag, observed_at=ts, now=ts)
    return judge_candidate(cand)


def classify_signal_hours(
    node_id: str,
    history: Sequence[tuple[datetime, float]],
    hour_starts: Iterable[datetime],
    *,
    min_readings: int | None = None,
) -> dict[datetime, SignalHourV1]:
    """One ``SignalHourV1`` per requested hour. ``history`` must cover 7 days before the earliest
    hour for full-week percentiles (a shorter week is what the live seam had at that time too)."""
    from orion.attention.world_first import node_prediction_error_semantics
    from orion.substrate.prediction_error_magnitude import DEFAULT_MIN_READINGS

    min_n = DEFAULT_MIN_READINGS if min_readings is None else int(min_readings)
    targets = sorted({hour_floor(h) for h in hour_starts})
    out = {h: SignalHourV1(signal_id=node_id, condition="unknown") for h in targets}
    if not targets:
        return out
    rows = sorted(
        ((_aware(ts), float(v)) for ts, v in list(history)[:MAX_ROWS] if v is not None and math.isfinite(float(v))),
        key=lambda r: r[0],
    )
    value_kind, _ = node_prediction_error_semantics(node_id)
    target_set = set(targets)
    counts: dict[datetime, list[int]] = {h: [0, 0, 0] for h in targets}  # readings, judged, hot
    if value_kind in ("trigger", "binary"):
        prev_on = False
        for ts, v in rows:
            h = hour_floor(ts)
            on = v > 0.0
            if h in target_set:
                c = counts[h]
                c[0] += 1
                c[1] += 1
                if on and not prev_on:
                    c[2] += 1
            prev_on = on
        for h, (n_rows, n_judged, n_on) in counts.items():
            if n_rows:
                out[h] = SignalHourV1(
                    signal_id=node_id, condition="high" if n_on else "normal",
                    n_readings=n_rows, n_judged=n_judged, n_hot=n_on,
                )
        return out
    if value_kind not in ("level", "count"):
        # placeholder / bucket / unknown kind: never a measurement, every hour unknown.
        return out
    from orion.attention.world_first import (
        EVENT_ORIENTING_WINDOW_SEC,
        INTERNAL_MAX_AGE_SEC,
        node_prediction_error_event_written,
    )

    # How long one reading speaks for the body: an event-written reading is news for the seam's
    # orienting window; a per-tick reading until the next one (at most the staleness horizon).
    cap = timedelta(
        seconds=EVENT_ORIENTING_WINDOW_SEC if node_prediction_error_event_written(node_id) else INTERNAL_MAX_AGE_SEC
    )
    start, end = targets[0], targets[-1] + HOUR
    lookback = timedelta(seconds=OPEN_WINDOW_SEC)
    judge_from = start - lookback - cap
    window: list[float] = []
    live: deque[tuple[datetime, float]] = deque()
    judged: list[tuple[datetime, bool | None]] = []  # (ts, eligible); None = could not be judged
    in_hour: dict[datetime, list[int]] = {h: [0, 0] for h in targets}  # readings, hot readings
    for ts, v in rows:
        if ts >= end:
            break
        live.append((ts, v))
        bisect.insort(window, v)
        while live and live[0][0] < ts - WINDOW_7D:
            _, old = live.popleft()
            del window[bisect.bisect_left(window, old)]
        if ts < judge_from:
            continue
        n = len(window)
        verdict = _judge_reading(node_id, v, ts, bisect.bisect_left(window, v) / n, n, min_n)
        hot = bool(verdict.eligible) if verdict.score is not None else None
        judged.append((ts, hot))
        h = hour_floor(ts)
        if h in in_hour:
            in_hour[h][0] += 1
            in_hour[h][1] += 1 if hot else 0
    # Replay the episode gate's polls (every POLL_SEC) over the hour: at each poll the newest
    # judged reading speaks if it is within its cap. High = the gate's own open dwell was met at
    # some poll in the hour (hot for MORE than OPEN_DWELL_SEC within the trailing OPEN_WINDOW_SEC):
    # sustained, not the ~10% of scattered top-decile readings every hour has by construction.
    step = timedelta(seconds=POLL_SEC)
    polls: dict[datetime, list[int]] = {h: [0, 0, 0] for h in targets}  # judged polls, hot polls, dwell met
    recent_hot: deque[float] = deque()
    idx = -1
    t = start - lookback
    while t < end:
        while idx + 1 < len(judged) and judged[idx + 1][0] <= t:
            idx += 1
        say = judged[idx] if idx >= 0 and t - judged[idx][0] <= cap else None
        epoch = t.timestamp()
        if say is not None and say[1]:
            recent_hot.append(epoch)
        while recent_hot and recent_hot[0] <= epoch - OPEN_WINDOW_SEC:
            recent_hot.popleft()
        h = hour_floor(t)
        if h in polls and say is not None and say[1] is not None:
            p = polls[h]
            p[0] += 1
            p[1] += 1 if say[1] else 0
            if say[1] and dwell_met(
                tuple(recent_hot), epoch, poll_sec=POLL_SEC, open_dwell_sec=HIGH_HOUR_HOT_SEC,
                open_window_sec=OPEN_WINDOW_SEC,
            ):
                p[2] = 1
        t += step
    for h in targets:
        n_judged_polls, n_hot_polls, dwell = polls[h]
        if n_judged_polls == 0:
            cond = "unknown"
        elif dwell:
            cond = "high"
        else:
            cond = "normal"
        out[h] = SignalHourV1(
            signal_id=node_id, condition=cond, n_readings=in_hour[h][0], n_judged=int(n_judged_polls > 0),
            n_hot=in_hour[h][1], judged_sec=float(n_judged_polls * POLL_SEC), hot_sec=float(n_hot_polls * POLL_SEC),
        )
    return out


# ---------------------------------------------------------------------------
# 2. Outcomes per hour
# ---------------------------------------------------------------------------


def turn_outcome_hours(
    rows: Iterable[tuple[datetime, str | None, bool | None]],
) -> dict[datetime, list[OutcomeHourV1]]:
    """``(created_at, compliance_verdict, response_repair_ran)`` per harness run -> per hour.
    A run with no verdict is not exposure for turn_failed; no repair flag, not for repairs."""
    acc: dict[datetime, list[int]] = {}
    for i, (ts, verdict, repaired) in enumerate(rows):
        if i >= MAX_ROWS or ts is None:
            break
        c = acc.setdefault(hour_floor(ts), [0, 0, 0, 0])
        if verdict:
            c[0] += 1
            c[1] += 1 if str(verdict).strip().lower() == "failed" else 0
        if repaired is not None:
            c[2] += 1
            c[3] += 1 if repaired else 0
    return {
        h: [
            OutcomeHourV1(outcome="turn_failed", exposure=c[0], events=c[1], source=TURN_SOURCE),
            OutcomeHourV1(outcome="response_repair", exposure=c[2], events=c[3], source=TURN_SOURCE),
        ]
        for h, c in acc.items()
    }


def rpc_outcome_hours(
    rows: Iterable[tuple[datetime, str, int | None, int | None]],
) -> dict[datetime, OutcomeHourV1]:
    """``(hour_start, key, success_count, timeout_count)`` transport_baseline_hourly rows -> per hour."""
    acc: dict[datetime, list[int]] = {}
    for i, (hs, key, ok, to) in enumerate(rows):
        if i >= MAX_ROWS or hs is None:
            break
        if str(key or "").endswith(RPC_EXCLUDED_KEY_SUFFIX):
            continue
        c = acc.setdefault(hour_floor(hs), [0, 0])
        c[0] += max(0, int(ok or 0)) + max(0, int(to or 0))
        c[1] += max(0, int(to or 0))
    return {
        h: OutcomeHourV1(outcome="rpc_timeout", exposure=c[0], events=c[1], source=RPC_SOURCE)
        for h, c in acc.items()
    }


def build_hour_row(
    hour_start: datetime,
    *,
    signals: Iterable[SignalHourV1],
    outcomes: Iterable[OutcomeHourV1],
    computed_at: datetime,
    history_hours: float,
) -> SelfSignalHourV1:
    h = hour_floor(hour_start)
    return SelfSignalHourV1(
        hour_id=hour_id(h),
        hour_start=h,
        computed_at=_aware(computed_at),
        signals=sorted(signals, key=lambda s: s.signal_id)[:MAX_SIGNALS_PER_HOUR],
        outcomes=sorted(outcomes, key=lambda o: o.outcome)[:8],
        history_hours=round(max(0.0, float(history_hours)), 3),
    )


# ---------------------------------------------------------------------------
# 3. Association
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    ratio: float | None
    low: float | None
    high: float | None


def katz_rate_ratio(a: int, n1: int, c: int, n0: int, z: float = Z_95) -> Interval:
    """Risk ratio (a/n1)/(c/n0) with Katz's log interval. A zero cell gets the Haldane 0.5
    correction for the interval only; the point ratio stays raw (None when c == 0)."""
    if n1 <= 0 or n0 <= 0:
        return Interval(None, None, None)
    ratio = (a / n1) / (c / n0) if c > 0 else None
    if a == 0 and c == 0:
        return Interval(ratio, None, None)
    aa, cc, nn1, nn0 = float(a), float(c), float(n1), float(n0)
    if a == 0 or c == 0 or a == n1 or c == n0:
        aa, cc, nn1, nn0 = aa + 0.5, cc + 0.5, nn1 + 1.0, nn0 + 1.0
    log_rr = math.log((aa / nn1) / (cc / nn0))
    se = math.sqrt(max(0.0, 1.0 / aa - 1.0 / nn1 + 1.0 / cc - 1.0 / nn0))
    return Interval(ratio, math.exp(log_rr - z * se), math.exp(log_rr + z * se))


def association_id(signal_id: str, outcome: str, computed_at: datetime) -> str:
    digest = hashlib.sha1(f"{signal_id}|{outcome}|{_aware(computed_at).isoformat()}".encode()).hexdigest()[:10]
    return f"ssa-{digest}"


def _signal_condition(row: SelfSignalHourV1, signal_id: str) -> str:
    for s in row.signals:
        if s.signal_id == signal_id:
            return s.condition
    return "unknown"


def _outcome(row: SelfSignalHourV1, outcome: str) -> OutcomeHourV1 | None:
    for o in row.outcomes:
        if o.outcome == outcome:
            return o
    return None


def short_signal(signal_id: str) -> str:
    return signal_id.removeprefix("node:substrate.")


def clause_for(assoc: SelfSignalOutcomeAssociationV1) -> str:
    """The one sentence a prompt may show. "" unless the interval excludes 1."""
    if assoc.verdict not in ("tends_more", "tends_less") or assoc.rate_high is None or assoc.rate_normal is None:
        return ""
    what = _OUTCOME_WORDS.get(assoc.outcome, assoc.outcome)
    if assoc.ratio is not None:
        how = f"{assoc.ratio:.1f}x as often"
    else:
        how = "more often"
    return (
        f"when this is high, my next hour had {what} {how} ({assoc.rate_high:.0%} of {assoc.n_high} "
        f"high hours vs {assoc.rate_normal:.0%} of {assoc.n_normal} normal) [assoc:{assoc.association_id}]"
    )


def associate(
    hours: Sequence[SelfSignalHourV1],
    signal_id: str,
    outcome: str,
    *,
    computed_at: datetime,
    min_hours: int = DEFAULT_MIN_HOURS,
) -> SelfSignalOutcomeAssociationV1:
    by_hour = {hour_floor(r.hour_start): r for r in list(hours)[-MAX_HOURS:]}
    n1 = a = n0 = c = 0
    used: list[datetime] = []
    for h, row in by_hour.items():
        cond = _signal_condition(row, signal_id)
        if cond not in ("high", "normal"):
            continue
        nxt = by_hour.get(h + HOUR)
        o = _outcome(nxt, outcome) if nxt is not None else None
        if o is None or o.exposure <= 0:
            continue
        hit = 1 if o.events > 0 else 0
        if cond == "high":
            n1 += 1
            a += hit
        else:
            n0 += 1
            c += hit
        used.append(h)
    iv = katz_rate_ratio(a, n1, c, n0)
    circular = CIRCULAR_PAIRS.get((signal_id, outcome), "")
    if circular:
        verdict = "excluded_circular"
    elif n1 < min_hours or n0 < min_hours:
        verdict = "below_min_n"
    elif iv.low is not None and iv.low > 1.0:
        verdict = "tends_more"
    elif iv.high is not None and iv.high < 1.0:
        verdict = "tends_less"
    else:
        verdict = "no_clear_difference"
    computed_at = _aware(computed_at)
    assoc = SelfSignalOutcomeAssociationV1(
        association_id=association_id(signal_id, outcome, computed_at),
        computed_at=computed_at,
        signal_id=signal_id,
        outcome=outcome,  # type: ignore[arg-type]
        window_start=min(used) if used else None,
        window_end=max(used) if used else None,
        n_high=n1, events_high=a, n_normal=n0, events_normal=c,
        rate_high=round(a / n1, 6) if n1 else None,
        rate_normal=round(c / n0, 6) if n0 else None,
        ratio=None if iv.ratio is None else round(iv.ratio, 4),
        ci_low=None if iv.low is None else round(iv.low, 4),
        ci_high=None if iv.high is None else round(iv.high, 4),
        min_hours=int(min_hours),
        verdict=verdict,  # type: ignore[arg-type]
        excluded_reason=circular,
    )
    return assoc.model_copy(update={"clause": clause_for(assoc)})


def associate_all(
    hours: Sequence[SelfSignalHourV1],
    *,
    computed_at: datetime,
    min_hours: int = DEFAULT_MIN_HOURS,
) -> list[SelfSignalOutcomeAssociationV1]:
    signals = sorted({s.signal_id for r in hours for s in r.signals})[:MAX_SIGNALS_PER_HOUR]
    return [
        associate(hours, sid, outcome, computed_at=computed_at, min_hours=min_hours)
        for sid in signals
        for outcome in OUTCOMES
    ]


def _strength(a: SelfSignalOutcomeAssociationV1) -> float:
    """Distance of the interval's nearer bound from 1, on the log scale."""
    if a.verdict == "tends_more" and a.ci_low:
        return math.log(a.ci_low)
    if a.verdict == "tends_less" and a.ci_high:
        return -math.log(a.ci_high)
    return 0.0


def best_clause(
    associations: Iterable[SelfSignalOutcomeAssociationV1], signal_id: str
) -> SelfSignalOutcomeAssociationV1 | None:
    """The strongest renderable association for one signal, or None."""
    cands = [a for a in associations if a.signal_id == signal_id and a.clause]
    if not cands:
        return None
    return sorted(cands, key=lambda a: (-_strength(a), a.outcome))[0]


def latest_by_pair(
    rows: Iterable[SelfSignalOutcomeAssociationV1],
) -> list[SelfSignalOutcomeAssociationV1]:
    best: dict[tuple[str, str], SelfSignalOutcomeAssociationV1] = {}
    for r in rows:
        k = (r.signal_id, r.outcome)
        if k not in best or _aware(r.computed_at) > _aware(best[k].computed_at):
            best[k] = r
    return [best[k] for k in sorted(best)]


def index_by_signal(
    rows: Iterable[SelfSignalOutcomeAssociationV1],
) -> Mapping[str, SelfSignalOutcomeAssociationV1]:
    """signal_id -> its one renderable association (only signals that have one)."""
    latest = latest_by_pair(rows)
    out: dict[str, SelfSignalOutcomeAssociationV1] = {}
    for sid in sorted({r.signal_id for r in latest}):
        b = best_clause(latest, sid)
        if b is not None:
            out[sid] = b
    return out
