"""Street rhythm surprise -> ``node:prediction.street_rhythm``. Pure, stdlib only.

Spec: docs/superpowers/specs/2026-11-24-pe-retire-perception-and-alarms-design.md rev 4, item 5 and
retire step 6. Orion's one real perception prediction error: how surprised the walkway rhythm forecast
was by who did or didn't appear.

orion-prediction publishes one ``prediction.street_rhythm_surprise.v1`` per scored walkway expectation
(``services/orion-prediction/app/street_rhythm.py``): the raw stated p, clamped, scored by log loss,
``excess = surprisal - binary entropy`` (mean 0 under calibration) and ``calm_var`` (its calm variance).

Aggregation (the node's value), same shape as ``node:prediction.chat_turn``
(``orion/substrate/prediction_turn_surprise.py``): keep the outcomes scored in the last ``WINDOW``
(30 min of wall-clock time; the rhythm learner scores in batches, so one window is usually one scoring
pass, median 2 outcomes), ``z = sum(excess) / sqrt(sum(calm_var))``, stored as ``clip(z / 3, 0, 1)``. One
outcome per expectation_id (a republish replaces itself). Written on outcomes only (no clock rewrite):
about 21 writes a day. The attention contests' 200-readings-in-7-days floor (#3917) would read that as
insufficient history; #3917's fix (calm-theory bands for forecast-surprise nodes, one table) must carry a
street_rhythm entry before this node can compete. It is in shadow until then anyway.

Window choice (14-day fixture, 2026-11-11 .. 11-25, writes at each scoring time): 30 min exact 0 on 67 %
of writes, p99 0.89; 6 h held a miss for 6 h and re-served it as news on every later write (the "old
alarm arriving as news" error), so it was not used.

- calm: a calibrated forecaster reads z ~ N(0, 1) -> exact 0 about half the time. Rest point 0.
- a confident miss (stated 0.9, the subject did not come): excess ~ +1.98 nats, z alone ~ +3.0 -> 1.0.
- no outcome (night, prediction down, kill switch): no writes; the node ages to absent (world-first
  600 s, every reader 1800 s), never calm. A backlog older than the window is dropped, never written as
  now. Lost on restart by design.

WORLD node (``orion/schemas/attention_candidate.py`` ``_WORLD_NODE_IDS``): its causes are people and
vehicles on the street. SHADOW until the 7-day calm check passes
(``services/orion-prediction/evals/run_street_rhythm_calm_eval.py``): ``SHADOW_NODE_IDS`` fences it from the
attention contests (and so from goals and Prime) and from curiosity seeds; it is still written, sampled
into the history table and shown on /health.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

STREET_RHYTHM_NODE_ID = "node:prediction.street_rhythm"
PREDICTION_STREET_RHYTHM_NODE_IDS: frozenset[str] = frozenset({STREET_RHYTHM_NODE_ID})
WINDOW = timedelta(minutes=30)
REDUCER_KEY = "prediction_street_rhythm"
Z_FULL_SCALE = 3.0


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class RhythmSample:
    expectation_id: str
    scored_at: datetime
    excess: float
    calm_var: float
    happened: bool
    p: float


@dataclass(frozen=True)
class StreetRhythmReading:
    node_id: str
    mean_excess: float      # per outcome, signed
    z: float                # signed
    value: float            # clip(z / 3, 0, 1): what the node stores
    n: int                  # outcomes in the window
    newest_scored_at: datetime
    expectation_ids: tuple[str, ...]


@dataclass
class StreetRhythmState:
    """In-memory window. Thread-safe: the worker folds in a worker thread while /health reads on the loop."""

    samples: dict[str, RhythmSample] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def bump(self, key: str) -> None:
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + 1

    def counts_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counts)

    def _prune(self, now: datetime) -> None:
        cutoff = _aware(now) - WINDOW
        self.samples = {k: s for k, s in self.samples.items() if s.scored_at >= cutoff}

    def add(self, event: Mapping[str, Any] | Any, *, now: datetime) -> tuple[str | None, str]:
        """Fold one validated ``PredictionStreetRhythmSurpriseV1`` (model or mapping). Only "accepted"
        changes the node. The same expectation twice replaces itself (idempotent)."""
        get = event.get if isinstance(event, Mapping) else (lambda k: getattr(event, k, None))
        if str(get("stream_id") or "") != "walkway":
            return None, "unknown_stream"
        scored = _aware(get("scored_at"))
        now = _aware(now)
        if scored < now - WINDOW:
            return STREET_RHYTHM_NODE_ID, "older_than_window"
        if scored > now + timedelta(minutes=5):
            return STREET_RHYTHM_NODE_ID, "future_scored_at"
        excess, var = float(get("excess")), float(get("calm_var"))
        if not (math.isfinite(excess) and math.isfinite(var)) or var < 0:
            return STREET_RHYTHM_NODE_ID, "non_finite"
        eid = str(get("expectation_id"))
        with self._lock:
            self._prune(now)
            self.samples[eid] = RhythmSample(expectation_id=eid, scored_at=scored, excess=excess, calm_var=var,
                                             happened=bool(get("happened")), p=float(get("p")))
        return STREET_RHYTHM_NODE_ID, "accepted"

    def reading(self, *, now: datetime) -> StreetRhythmReading | None:
        """The window as a z-score of summed excess; None when it holds no outcome with information
        (empty, or only 50/50 forecasts: calm_var 0, no z)."""
        with self._lock:
            self._prune(now)
            rows = sorted(self.samples.values(), key=lambda s: s.scored_at)
        if not rows:
            return None
        var = sum(s.calm_var for s in rows)
        if var <= 1e-12:
            return None
        total = sum(s.excess for s in rows)
        z = total / math.sqrt(var)
        return StreetRhythmReading(
            node_id=STREET_RHYTHM_NODE_ID, mean_excess=total / len(rows), z=z,
            value=max(0.0, min(1.0, z / Z_FULL_SCALE)), n=len(rows),
            newest_scored_at=rows[-1].scored_at,
            expectation_ids=tuple(s.expectation_id for s in rows),
        )


def replay_node_values(samples: list[RhythmSample]) -> list[tuple[datetime, float]]:
    """What the event writer would store at each distinct scoring time: (t, value), skipping a time whose
    window carries no information. The calm eval reads this for the node-level rest."""
    rows = sorted(samples, key=lambda s: s.scored_at)
    out: list[tuple[datetime, float]] = []
    for t in sorted({s.scored_at for s in rows}):
        win = [s for s in rows if t - WINDOW <= s.scored_at <= t]
        var = sum(s.calm_var for s in win)
        if var <= 1e-12:
            continue
        z = sum(s.excess for s in win) / math.sqrt(var)
        out.append((t, max(0.0, min(1.0, z / Z_FULL_SCALE))))
    return out
