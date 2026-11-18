"""Rolling write-failure reading over the sql-writer's recent windows.

Same rule as the RPC delivery bridge and the inference lane
(``orion/substrate/rpc_delivery.py`` ``hop_pressure`` with ``RpcDeliveryConfig``
defaults -- reused, not re-picked)::

    failed / max(attempted, 10) over the last 600 s of windows (event time),
    0.0 until at least 2 failures are in that span,

taken for every table family and for the pool, and the reading is the WORST.
``max`` rather than pooled alone: the writer commits ~265 grammar events and
~300 table rows a minute, so a table whose every row is being rejected (the
2026-09-26 home-cooling burst: 863 rows, ~7/min, all lost) would vanish inside
the pooled share. The receipt names the family that set the number.

Not measured (``pressure=None``) when nothing was attempted in the span: the
reducer then writes no hint, and the field channel expires rather than reading
calm.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from orion.schemas.storage_write_projection import StorageWriteWindowCountV1
from orion.substrate.rpc_delivery import RpcDeliveryConfig, hop_pressure

_RPC = RpcDeliveryConfig()
FAILURE_WINDOW_SEC: float = _RPC.window_s
FAILURE_MIN_DENOMINATOR: int = _RPC.min_denominator
FAILURE_MIN_COUNT: int = _RPC.min_timeouts
# The writer flushes every ~60 s, so a 600 s span holds ~10 windows.
MAX_WINDOWS: int = 120
POOLED_SCOPE = "all"


@dataclass(frozen=True)
class WriteFailureReading:
    pressure: float | None
    scope: str | None  # "all" (pooled) or the table family that set the reading
    scope_failed: int
    scope_attempted: int
    attempted: int
    failed: int
    windows: int
    failing_families: int
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
            "failing_families": self.failing_families,
            "window_sec": self.window_sec,
            "min_denominator": self.min_denominator,
            "min_failures": self.min_failures,
        }


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def fold_window(
    history: list[StorageWriteWindowCountV1],
    window: StorageWriteWindowCountV1,
    *,
    window_sec: float = FAILURE_WINDOW_SEC,
) -> list[StorageWriteWindowCountV1]:
    """Add ``window``. A stored window with the same ``window_id`` is merged by
    family key (a split trace's second half adds its families; a replayed atom
    overwrites its own family instead of counting twice). Windows that ended
    more than ``window_sec`` before the newest are dropped. Oldest first."""
    kept: list[StorageWriteWindowCountV1] = []
    merged = window
    for w in history:
        if w.window_id == window.window_id:
            merged = StorageWriteWindowCountV1(
                window_id=w.window_id,
                window_end=max(_aware(w.window_end), _aware(window.window_end)),
                attempted={**w.attempted, **window.attempted},
                failed={**w.failed, **window.failed},
            )
        else:
            kept.append(w)
    kept.append(merged)
    kept.sort(key=lambda w: _aware(w.window_end))
    newest = _aware(kept[-1].window_end)
    cutoff = newest - timedelta(seconds=window_sec)
    kept = [w for w in kept if _aware(w.window_end) > cutoff]
    return kept[-MAX_WINDOWS:]


def failure_reading(
    history: list[StorageWriteWindowCountV1],
    *,
    min_denominator: int = FAILURE_MIN_DENOMINATOR,
    min_failures: int = FAILURE_MIN_COUNT,
) -> WriteFailureReading:
    attempted_by: dict[str, int] = {}
    failed_by: dict[str, int] = {}
    for w in history:
        for fam, n in w.attempted.items():
            attempted_by[fam] = attempted_by.get(fam, 0) + max(0, int(n))
        for fam, n in w.failed.items():
            failed_by[fam] = failed_by.get(fam, 0) + max(0, int(n))
    attempted = sum(attempted_by.values())
    failed = sum(min(failed_by.get(f, 0), a) for f, a in attempted_by.items())
    failing = sum(1 for f in attempted_by if failed_by.get(f, 0) > 0)
    if attempted <= 0:
        return WriteFailureReading(
            pressure=None, scope=None, scope_failed=0, scope_attempted=0,
            attempted=0, failed=0, windows=len(history), failing_families=0,
        )
    best_scope = POOLED_SCOPE
    best = hop_pressure(failed, attempted - failed, min_denominator, min_failures)
    best_failed, best_attempted = failed, attempted
    for fam in sorted(attempted_by):
        f_att = attempted_by[fam]
        f_failed = min(failed_by.get(fam, 0), f_att)
        value = hop_pressure(f_failed, f_att - f_failed, min_denominator, min_failures)
        if value > best:
            best, best_scope = value, fam
            best_failed, best_attempted = f_failed, f_att
    return WriteFailureReading(
        pressure=min(1.0, best),
        scope=best_scope,
        scope_failed=best_failed,
        scope_attempted=best_attempted,
        attempted=attempted,
        failed=failed,
        windows=len(history),
        failing_families=failing,
    )
