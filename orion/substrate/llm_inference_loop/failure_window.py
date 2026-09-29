"""Rolling inference-failure reading for one serving node (2026-09-29).

Before: ``inference_failure_pressure`` was one 60-second gateway window's
``upstream_failed / (served + upstream_failed)`` with no floor. A single timeout on
a window with one call read 1.0, and the field digester holds the reading until the
next window with traffic, so one timeout could sit at 1.0 for hundreds of ticks.
Live, 2026-09-26..29 (72 h): 35 windows had an upstream failure, every one of them
exactly one, and ``node:circe``'s channel was nonzero on 1,167 field ticks (max 1.0).

Now the same rule the RPC delivery bridge uses (``orion/substrate/rpc_delivery.py``,
``hop_pressure``, ``RpcDeliveryConfig`` defaults -- reused, not re-picked):

    failures / max(attempts, 10) over the last 600 s of this node's windows,
    0.0 until at least 2 failures are in that span.

Rolling rather than per-window because failures are sparse: no 60 s window in the
72 h replay held two, so a per-window 2-failure minimum would never fire. 10-minute
spans did hold two or three during the real slow periods (e.g. 2026-09-28
18:03-18:15, served p95 latency 47-274 s). The reading is the worse of
the node-pooled share and the worst single worker's share (``max``, like the RPC
bridge's worst hop): pooling alone lets a busy healthy lane dilute a failing one,
and a per-worker reading alone misses failures spread across lanes. The receipt
names which one set the number.

Event time, not wall clock: each window is placed at its own ``emitted_at``, so a
backlog replays the same readings it would have produced live.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from orion.schemas.llm_inference_projection import LlmInferenceWindowCountV1
from orion.substrate.rpc_delivery import RpcDeliveryConfig, hop_pressure

_RPC = RpcDeliveryConfig()
FAILURE_WINDOW_SEC: float = _RPC.window_s
FAILURE_MIN_DENOMINATOR: int = _RPC.min_denominator
FAILURE_MIN_COUNT: int = _RPC.min_timeouts
# Bound on stored windows per node. The gateway flushes every ~60 s, so a 600 s
# span needs ~10; the cap only matters if the flush interval is shortened a lot.
MAX_WINDOWS_PER_NODE: int = 120
POOLED_SCOPE = "node"


@dataclass(frozen=True)
class FailureWindowReading:
    pressure: float | None
    scope: str | None  # "node" (pooled) or the worker label that set the reading
    scope_failed: int
    scope_attempted: int
    attempted: int
    failed: int
    windows: int
    window_sec: float = FAILURE_WINDOW_SEC
    min_denominator: int = FAILURE_MIN_DENOMINATOR
    min_failures: int = FAILURE_MIN_COUNT

    def as_dict(self) -> dict[str, Any]:
        return {
            "pressure": self.pressure,
            "scope": self.scope,
            "scope_failed": self.scope_failed,
            "scope_attempted": self.scope_attempted,
            "attempted": self.attempted,
            "failed": self.failed,
            "windows": self.windows,
            "window_sec": self.window_sec,
            "min_denominator": self.min_denominator,
            "min_failures": self.min_failures,
        }


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def fold_window(
    history: list[LlmInferenceWindowCountV1],
    window: LlmInferenceWindowCountV1,
    *,
    window_sec: float = FAILURE_WINDOW_SEC,
) -> list[LlmInferenceWindowCountV1]:
    """Add ``window`` (replacing a stored copy with the same ``window_id`` -- a
    replayed trace must not count twice) and drop windows that ended more than
    ``window_sec`` before the newest one. Oldest first."""
    kept = [w for w in history if w.window_id != window.window_id]
    kept.append(window)
    kept.sort(key=lambda w: _aware(w.window_end))
    newest = _aware(kept[-1].window_end)
    cutoff = newest - timedelta(seconds=window_sec)
    kept = [w for w in kept if _aware(w.window_end) > cutoff]
    return kept[-MAX_WINDOWS_PER_NODE:]


def failure_reading(
    history: list[LlmInferenceWindowCountV1],
    *,
    min_denominator: int = FAILURE_MIN_DENOMINATOR,
    min_failures: int = FAILURE_MIN_COUNT,
) -> FailureWindowReading:
    served = sum(w.served for w in history)
    failed = sum(w.upstream_failed for w in history)
    attempted = served + failed
    if attempted <= 0:
        return FailureWindowReading(
            pressure=None, scope=None, scope_failed=0, scope_attempted=0,
            attempted=0, failed=0, windows=len(history),
        )
    best_scope = POOLED_SCOPE
    best = hop_pressure(failed, served, min_denominator, min_failures)
    best_failed, best_attempted = failed, attempted

    worker_attempted: dict[str, int] = {}
    worker_failed: dict[str, int] = {}
    for w in history:
        for label, n in w.worker_attempted.items():
            worker_attempted[label] = worker_attempted.get(label, 0) + max(0, int(n))
        for label, n in w.worker_failed.items():
            worker_failed[label] = worker_failed.get(label, 0) + max(0, int(n))
    for label in sorted(worker_attempted):
        w_failed = min(worker_failed.get(label, 0), worker_attempted[label])
        w_served = worker_attempted[label] - w_failed
        value = hop_pressure(w_failed, w_served, min_denominator, min_failures)
        if value > best:
            best, best_scope = value, label
            best_failed, best_attempted = w_failed, worker_attempted[label]

    return FailureWindowReading(
        pressure=min(1.0, best),
        scope=best_scope,
        scope_failed=best_failed,
        scope_attempted=best_attempted,
        attempted=attempted,
        failed=failed,
        windows=len(history),
    )
