"""The street as an attention source: the gate, the held node value, Rung 6.

Spec: docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md,
section 6 ("How loud"), with Juniper's rule from the walkway follow-ups spec
(PR #2763, answer 7): the street may win attention with EVENTS, not raw pixel
surprise; it needs habituation, a camera-down check, an outside-world label
and a measured cap with a kill switch.

Pure, no I/O. ``orion-substrate-runtime`` feeds it ``StreetEventV1`` messages
from ``orion:vision:street:events`` and writes the result to
``node:substrate.street``.

**Gate (one verdict per event, with a stated reason):**

- ``doorbell_ring`` always passes (a button press is a fact, not a guess). It
  still counts toward the hourly budget, so rings use up the street's share.
- Every other kind needs, in order: a ``p_value``; ``p_value < 0.05``; every
  camera in ``stream_ids`` delivered a frame recently (camera up); and fewer
  than ``max_per_hour`` passes in the trailing hour (the cap).
- "Unusual for the hour, or a known person": no event-level "unusual for this
  hour" reading exists that this runtime can read. The rhythm learner's grades
  live in ``vision_percept_expectation`` per subject and window, and the
  ``vision_rollup`` hour rows exist only for CLOSED hours, so neither answers
  "is this event unusual right now". The event's own ``p_value`` stands in for
  the hour clause, and the hourly cap does the habituation. A known person
  (``person_guess`` set) is recorded in the reason but changes nothing yet,
  because the p-value clause already decides. When the producer can say
  "unusual for the hour", this is the line to change.

**Calm state.** The node is written by a clock tick, every tick: the strongest
passed event's salience while it is held (``hold_sec``), else a real 0.0.
Nothing decays it: the field digester's staleness decay
(``NODE_DECAY_CHANNELS``) only touches FieldStateV1 node vectors, and this node
is a FalkorDB substrate node, not a field vector. So 0.0 here is a measured
"nothing on the street passed the gate", never a decayed leftover. When the
runtime cannot see the street (listener down, or no street camera delivered a
frame recently) the tick writes NOTHING, the reading ages, and attention marks
the node absent (``world_first.PERCEPTION_MAX_AGE_SEC``): silence is not calm.

**Rung 6 domain ``street_handoff``.** One raw reading per graded handoff
(``handoff_met`` / ``handoff_late`` / ``handoff_missed``): ``-ln(p_value)`` in
nats, the surprisal of the tail probability. ``StreetEventV1`` carries no
predicted transit distribution, so the arrival log-score the spec names cannot
be computed here (the producer must add ``arrival_log_score``). Under a
calibrated model p is uniform on (0, 1), so ``-ln p`` is Exponential(1):
**E[-ln U] = 1 nat and Var = 1.** A reading of ~1 nat is the rest point, not
signal; only a running mean well above 1 says the model is miscalibrated.
p is floored at ``P_VALUE_FLOOR`` (1e-6 -> 13.8 nats max) so p = 0 cannot
write infinity.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from orion.bus.ewma import compute_ewma_update
from orion.schemas.attention_candidate import STREET_NODE_ID

STREET_HANDOFF_DOMAIN = "street_handoff"
HANDOFF_KINDS: frozenset[str] = frozenset({"handoff_met", "handoff_late", "handoff_missed"})
RING_KIND = "doorbell_ring"

# Spec section 6: under a calibrated model ~5% of ordinary outcomes fall below
# this by chance, which is what the hourly cap is for.
GATE_P_VALUE = 0.05
P_VALUE_FLOOR = 1e-6
CAP_WINDOW_SEC = 3600.0

# Rung 6 starting expectation for street_handoff. alpha 0.2: the same
# real-observation alpha execution and chat use (prediction_error.py), no
# domain reason found to differ. min_variance 1e-3: the calibrated variance of
# -ln U is 1, so a floor three orders below it never binds once warm, and only
# stops a cold start dividing by ~0. Kept so the tuner's x10 step (1e-2) stays
# inside the prediction_expectation_patch contract's outer bound (0.03,
# orion/substrate/mutation_contracts.py) -- a larger floor would be refused.
STREET_HANDOFF_EWMA_ALPHA = 0.2
STREET_HANDOFF_MIN_VARIANCE = 1e-3

__all__ = [
    "STREET_NODE_ID",
    "STREET_HANDOFF_DOMAIN",
    "GateVerdict",
    "StreetAttentionState",
    "street_gate",
    "street_handoff_raw_reading",
]


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    reason: str


def _stale_camera(
    stream_ids: Sequence[str],
    *,
    now_epoch: float,
    last_frame_at: Mapping[str, float],
    max_age_sec: float,
) -> str | None:
    """Why the camera check fails, or None when every stream is up."""
    for stream in stream_ids:
        seen = last_frame_at.get(stream)
        if seen is None:
            return f"camera_unseen:{stream}"
        age = now_epoch - float(seen)
        if age > max_age_sec:
            return f"camera_down:{stream} (last frame {age:.0f}s ago)"
    return None


def street_gate(
    event: Any,
    *,
    now_epoch: float,
    recent_pass_epochs: Sequence[float],
    max_per_hour: int,
    last_frame_at: Mapping[str, float],
    camera_up_max_age_sec: float,
) -> GateVerdict:
    """The whole gate for one ``StreetEventV1``. See the module docstring."""
    if event.kind == RING_KIND:
        return GateVerdict(True, "doorbell_ring always passes")
    p = event.p_value
    if p is None:
        return GateVerdict(False, "no_p_value: only graded events can pass")
    if not p < GATE_P_VALUE:
        return GateVerdict(False, f"p_value {p:.3g} >= {GATE_P_VALUE}")
    stale = _stale_camera(
        event.stream_ids,
        now_epoch=now_epoch,
        last_frame_at=last_frame_at,
        max_age_sec=camera_up_max_age_sec,
    )
    if stale is not None:
        return GateVerdict(False, stale)
    used = sum(1 for t in recent_pass_epochs if now_epoch - t < CAP_WINDOW_SEC)
    if used >= max(0, int(max_per_hour)):
        return GateVerdict(False, f"hourly_cap: {used} of {max_per_hour} street passes used")
    who = f"known person {event.person_guess}" if event.person_guess else "p_value stands in for unusual-for-hour"
    return GateVerdict(True, f"p_value {p:.3g} < {GATE_P_VALUE} ({who})")


def street_handoff_raw_reading(event: Any) -> float | None:
    """Rung 6 raw reading for a graded handoff, in nats; None otherwise."""
    if event.kind not in HANDOFF_KINDS or event.p_value is None:
        return None
    p = float(event.p_value)
    if not math.isfinite(p):
        return None
    return -math.log(max(p, P_VALUE_FLOOR))


def _reason_class(reason: str) -> str:
    return reason.split(":", 1)[0].split(" ", 1)[0]


@dataclass
class _Held:
    epoch: float
    salience: float
    event_id: str
    correlation_id: str
    kind: str


@dataclass
class StreetAttentionState:
    """In-process state for one runtime: passes, the held value, counters."""

    pass_epochs: deque = field(default_factory=lambda: deque(maxlen=512))
    held: deque = field(default_factory=lambda: deque(maxlen=64))
    last_frame_at: dict[str, float] = field(default_factory=dict)
    received_total: int = 0
    invalid_total: int = 0
    passed_total: int = 0
    blocked_total: int = 0
    blocked_by_reason: dict[str, int] = field(default_factory=dict)
    passed_by_kind: dict[str, int] = field(default_factory=dict)
    attention_wins_total: int = 0
    node_writes_total: int = 0
    ticks_unmeasured_total: int = 0
    last_event_at: float | None = None
    # Last value the runtime wrote to the node (0.0 before the first write).
    last_written_value: float = 0.0
    last_verdict: dict[str, Any] | None = None
    # street_handoff EWMA, read only for the trace (handoff_surprise_z).
    handoff_ewma: float = 0.0
    handoff_variance: float = 0.0
    handoff_count: int = 0
    # The bus handler, the tick and the perception listener touch this from
    # different threads (asyncio.to_thread); every read and write holds it.
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def bump(self, counter: str) -> None:
        """Increment one ``*_total`` counter under the lock."""
        with self._lock:
            setattr(self, counter, int(getattr(self, counter)) + 1)

    def frame_seen(self, stream_id: str | None, now_epoch: float) -> None:
        if stream_id:
            with self._lock:
                self.last_frame_at[str(stream_id)] = float(now_epoch)

    def judge(self, event: Any, *, now_epoch: float, max_per_hour: int, camera_up_max_age_sec: float) -> GateVerdict:
        with self._lock:
            return self._judge(
                event, now_epoch=now_epoch, max_per_hour=max_per_hour, camera_up_max_age_sec=camera_up_max_age_sec
            )

    def _judge(self, event: Any, *, now_epoch: float, max_per_hour: int, camera_up_max_age_sec: float) -> GateVerdict:
        verdict = street_gate(
            event,
            now_epoch=now_epoch,
            recent_pass_epochs=list(self.pass_epochs),
            max_per_hour=max_per_hour,
            last_frame_at=self.last_frame_at,
            camera_up_max_age_sec=camera_up_max_age_sec,
        )
        self.received_total += 1
        self.last_event_at = now_epoch
        if verdict.passed:
            self.passed_total += 1
            self.passed_by_kind[event.kind] = self.passed_by_kind.get(event.kind, 0) + 1
            self.pass_epochs.append(now_epoch)
            self.held.append(
                _Held(now_epoch, float(event.salience), str(event.event_id), str(event.correlation_id), event.kind)
            )
        else:
            self.blocked_total += 1
            key = _reason_class(verdict.reason)
            self.blocked_by_reason[key] = self.blocked_by_reason.get(key, 0) + 1
        self.last_verdict = {
            "event_id": str(event.event_id),
            "kind": event.kind,
            "passed": verdict.passed,
            "reason": verdict.reason,
            "correlation_id": str(event.correlation_id),
            "at": now_epoch,
        }
        return verdict

    def held_reading(self, now_epoch: float, hold_sec: float) -> tuple[float, list[_Held]]:
        """(value, events held): the strongest passed salience inside the hold."""
        with self._lock:
            live = [h for h in self.held if now_epoch - h.epoch < hold_sec]
        if not live:
            return 0.0, []
        return max(h.salience for h in live), live

    def street_visible(self, stream_ids: Sequence[str], now_epoch: float, max_age_sec: float) -> bool:
        """True when any street camera delivered a frame recently."""
        with self._lock:
            return any(
                now_epoch - self.last_frame_at[s] <= max_age_sec for s in stream_ids if s in self.last_frame_at
            )

    def handoff_surprise_z(self, raw: float, *, alpha: float, min_variance: float) -> float | None:
        """Fold one street_handoff raw reading into the running expectation
        (Orion's adopted Rung 6 setting when live) and return its z against
        the expectation held BEFORE it. None on the first reading."""
        with self._lock:
            return self._handoff_surprise_z(raw, alpha=alpha, min_variance=min_variance)

    def _handoff_surprise_z(self, raw: float, *, alpha: float, min_variance: float) -> float | None:
        update = compute_ewma_update(
            prev_ewma=self.handoff_ewma,
            prev_variance=self.handoff_variance,
            prev_count=self.handoff_count,
            value=raw,
            alpha=alpha,
            min_variance=min_variance,
        )
        self.handoff_ewma, self.handoff_variance = update.ewma, update.variance
        self.handoff_count += 1
        return update.zscore

    def status(self, now_epoch: float, hold_sec: float) -> dict[str, Any]:
        with self._lock:
            return self._status(now_epoch, hold_sec)

    def _status(self, now_epoch: float, hold_sec: float) -> dict[str, Any]:
        value, live = self.held_reading(now_epoch, hold_sec)
        return {
            "node_id": STREET_NODE_ID,
            "received_total": self.received_total,
            "invalid_total": self.invalid_total,
            "passed_total": self.passed_total,
            "blocked_total": self.blocked_total,
            "blocked_by_reason": dict(self.blocked_by_reason),
            "passed_by_kind": dict(self.passed_by_kind),
            "passes_last_hour": sum(1 for t in self.pass_epochs if now_epoch - t < CAP_WINDOW_SEC),
            "attention_wins_total": self.attention_wins_total,
            "node_writes_total": self.node_writes_total,
            "ticks_unmeasured_total": self.ticks_unmeasured_total,
            "held_value": value,
            "held_event_ids": [h.event_id for h in live],
            "last_event_age_sec": None if self.last_event_at is None else round(now_epoch - self.last_event_at, 1),
            "last_verdict": self.last_verdict,
            "camera_last_frame_age_sec": {
                s: round(now_epoch - t, 1) for s, t in sorted(self.last_frame_at.items())
            },
            "handoff_readings": self.handoff_count,
        }
