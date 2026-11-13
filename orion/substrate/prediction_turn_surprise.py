"""Chat-turn surprise -> ``node:prediction.chat_turn``. Pure, stdlib only.

Spec: docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md ("Retire order"
step 6: chat's successor is the turn self-bets). Replaces ``node:substrate.chat``.

orion-prediction publishes one ``prediction.turn_surprise.v1`` per settled chat turn whose used
stance bets (``services/orion-prediction/app/turn_surprise.py``: asks a question, more steps than
usual) carry information. Each event holds the turn's summed ``excess`` surprise (nats over what the
held-out-recalibrated bets expected) and its calm variance ``calm_var``.

Aggregation (the node's value), same shape as ``node:prediction.room`` / ``.body``
(``orion/substrate/prediction_surprise.py``): keep the turns settled in the last ``WINDOW``
(30 min) of wall-clock time, ``z = sum(excess) / sqrt(sum(calm_var))``, stored as
``clip(z / 3, 0, 1)``. A turn is one sample however many bets it holds (its bets are already summed).

- calm: turns that went the way the bets leaned read z < 0 -> exactly 0. Rest point 0.
- a surprising turn (a question at calibrated p ~0.1) reads z ~ 3 -> 1.0 for the 30 min it stays.
- no chat turn: no events, no writes; ``observed_at`` ages and every reader treats the node as
  stale/absent after 1800 s (``orion/substrate/prediction_error_freshness.py``), never as calm.
  A turn settled before the window is dropped, so a backlog never writes the past as now.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

CHAT_TURN_NODE_ID = "node:prediction.chat_turn"
PREDICTION_TURN_SURPRISE_NODE_IDS: frozenset[str] = frozenset({CHAT_TURN_NODE_ID})
CALLER_NODE_IDS: dict[str, str] = {"chat": CHAT_TURN_NODE_ID}
WINDOW = timedelta(minutes=30)
REDUCER_KEY = "prediction_turn_surprise"
Z_FULL_SCALE = 3.0


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class TurnSample:
    correlation_id: str
    settled_at: datetime
    excess: float
    calm_var: float
    n_bets: int


@dataclass(frozen=True)
class TurnSurpriseReading:
    node_id: str
    mean_excess: float      # per turn, signed
    z: float                # signed
    value: float            # clip(z / 3, 0, 1): what the node stores
    n: int                  # turns in the window
    newest_settled_at: datetime
    correlation_ids: tuple[str, ...]


@dataclass
class TurnSurpriseState:
    """In-memory window per node. Lost on restart by design (the node then ages to stale). Thread-
    safe: the worker folds in a worker thread while /health reads on the event loop."""

    samples: dict[str, dict[str, TurnSample]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def bump(self, key: str) -> None:
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + 1

    def counts_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counts)

    def _prune(self, node_id: str, now: datetime) -> None:
        cutoff = _aware(now) - WINDOW
        self.samples[node_id] = {k: s for k, s in (self.samples.get(node_id) or {}).items()
                                 if s.settled_at >= cutoff}

    def add(self, event: Mapping[str, Any] | Any, *, now: datetime) -> tuple[str | None, str]:
        """Fold one validated ``PredictionTurnSurpriseV1`` (model or mapping). Only "accepted"
        changes a node. The same turn twice replaces itself (idempotent)."""
        get = event.get if isinstance(event, Mapping) else (lambda k: getattr(event, k, None))
        node_id = CALLER_NODE_IDS.get(str(get("caller") or ""))
        if node_id is None:
            return None, "unknown_caller"
        settled = _aware(get("settled_at"))
        if settled < _aware(now) - WINDOW:
            return node_id, "older_than_window"
        excess, var = float(get("excess")), float(get("calm_var"))
        if not (math.isfinite(excess) and math.isfinite(var)) or var <= 0:
            return node_id, "non_finite_or_no_variance"
        bets = get("bets") or ()
        with self._lock:
            self._prune(node_id, now)
            self.samples.setdefault(node_id, {})[str(get("correlation_id"))] = TurnSample(
                correlation_id=str(get("correlation_id")), settled_at=settled, excess=excess, calm_var=var,
                n_bets=len(bets))
        return node_id, "accepted"

    def reading(self, node_id: str, *, now: datetime) -> TurnSurpriseReading | None:
        with self._lock:
            self._prune(node_id, now)
            rows = sorted((self.samples.get(node_id) or {}).values(), key=lambda s: s.settled_at)
        if not rows:
            return None
        total = sum(s.excess for s in rows)
        z = total / math.sqrt(sum(s.calm_var for s in rows))
        return TurnSurpriseReading(
            node_id=node_id, mean_excess=total / len(rows), z=z,
            value=max(0.0, min(1.0, z / Z_FULL_SCALE)), n=len(rows),
            newest_settled_at=rows[-1].settled_at,
            correlation_ids=tuple(s.correlation_id for s in rows),
        )
