"""Bounded "find the oldest unprocessed row" scan, with a backstop that keeps it safe.

ROADMAP D2 follow-through, 2026-08-19.

THE PROBLEM THIS SOLVES
-----------------------
Orion's substrate is a pipeline -- proposal -> policy decision -> execution dispatch -> feedback
-- and each stage finds its next unit of work by asking "which row upstream has no row
downstream yet". Written as an unbounded anti-join over both tables, that is O(whole history)
per poll, forever, no matter how little work is actually pending.

Measured live 2026-08-19 with EXPLAIN (ANALYZE, BUFFERS), 419,526 rows per table:

    unbounded:  106,052 blocks read (829 MB) + 465 MB spilled to temp, PER EXECUTION
    bounded 1h:     916 blocks read, no temp spill  ->  116x fewer disk reads

At a 2s poll interval that was the single largest contributor to athena being fully I/O-stalled
~20% of wall time (`/proc/pressure/io`), with Postgres reading 149 GB/hour against a 47 GB
database.

STATUS: DISABLED BY DEFAULT. THE BOUND IS UNSAFE AS DESIGNED.
------------------------------------------------------------
Code review, 2026-08-19, found a defect that no amount of window tuning fixes:

    `fetch()` returns as soon as the BOUNDED query finds a row, so the backstop is only ever
    reached when the fast path is EMPTY. During a real backlog -- precisely the condition the
    backstop exists for -- fresh in-window work always exists, the backstop never fires, and
    every row older than the window is stranded PERMANENTLY. Worse, it self-perpetuates: any
    row that ages past the window before being reached joins the untouchable set.

Live evidence that this is not hypothetical: on 2026-08-14 the dispatch->feedback stage produced
29,264 feedback frames for dispatch rows that had waited ~34 HOURS, while 26,148 new rows
arrived the same day. 8 of the last 30 days were entirely in that regime, ~30k rows/day.

The measurement that justified a 1h window (n=514 over 24h, max 85.6s) was taken during an
unrepresentative quiet spell -- 08-18 saw 133 dispatch frames against 26k-39k on active days.
Over 7 days the real lag is p50 124,613s, max 975,770s. This pipeline legitimately runs hours
to days behind, which makes "only look at recent rows" the wrong shape for it entirely.

A rate limit compounds it: `_last_backstop` is stamped even when the backstop RETURNS a row, so
straggler drain is capped at one row per interval -- 288/day against 26k-39k/day of real work.

WHAT WOULD ACTUALLY WORK is a marker column (`*_pending boolean`) with a partial index, so
"oldest unprocessed" is O(pending) instead of O(history) regardless of how far behind the stage
is. That is a schema migration across the substrate pipeline and has not been done.

Everything below describes the design AS BUILT. It is left in place, defaulted off, because the
shape is reusable once the strand-the-backlog defect is fixed -- not because it is ready.

WHY THE BOUND ALONE IS NOT SAFE, AND WHAT MAKES IT SAFE (as-built, INSUFFICIENT -- see above)
--------------------------------------------------------------------------------------------
The bound assumes the pipeline is roughly current. Measured, it is:

    proposal -> policy   lag over 24h:  p50  3.4s   p99 17.0s   max 31.8s
    dispatch -> feedback lag over 24h:  p50 16.0s   p99 77.1s   max 85.6s

But if a stage ever falls behind by more than its window -- an outage, a slow policy, a burst --
the bounded query returns nothing while real work sits just outside it, and that work would be
skipped SILENTLY AND FOREVER. A pipeline that quietly drops its own backlog is a far worse
failure than a slow one.

So the unbounded query survives as a **rate-limited backstop**: it runs only when the fast path
finds nothing, at most once per `backstop_interval_sec`, and logs loudly when it actually picks
something up. Nothing is ever permanently skipped -- a straggler is picked up within one
backstop interval instead of instantly, and the tripwire says so.

Setting `window_sec <= 0` disables the bound entirely and restores the original behaviour. That
is the rollback, and it needs no code change.

WHY THIS LIVES IN orion/db AND NOT orion/substrate
--------------------------------------------------
`orion/substrate/__init__.py` eagerly imports the materializer, which imports `requests`. The
services that need this helper are thin bus workers that do not ship `requests`, so placing it
under orion/substrate put BOTH orion-policy-runtime and orion-feedback-runtime into a restart
loop on ModuleNotFoundError -- confirmed live 2026-08-19, and briefly disguised as a
performance win, because a crash-looping service does no sequential scans either. Keep
`orion/db/__init__.py` empty.

NOT USED BY orion-execution-dispatch-runtime
--------------------------------------------
That service solved the same problem differently on 2026-07-30 (batching the FIFO drain to
LIMIT 200 once per tick, plus `NOT EXISTS` for the newest-first direction) and documented the
reasoning at length in its own store.py. Its shape is not this shape; do not "unify" them
without re-reading that account first.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping, Optional

DEFAULT_WINDOW_SEC = 3600.0
DEFAULT_BACKSTOP_INTERVAL_SEC = 300.0


class BoundedPendingScan:
    """One stage's pending-work lookup. Not thread-safe; one instance per store."""

    def __init__(
        self,
        *,
        label: str,
        window_sec: float = DEFAULT_WINDOW_SEC,
        backstop_interval_sec: float = DEFAULT_BACKSTOP_INTERVAL_SEC,
        logger: Optional[logging.Logger] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.label = str(label)
        self.window_sec = float(window_sec)
        self.backstop_interval_sec = float(backstop_interval_sec)
        self._log = logger or logging.getLogger("orion.db.pending_scan")
        self._monotonic = monotonic
        self._last_backstop: Optional[float] = None

    @property
    def bounded(self) -> bool:
        return self.window_sec > 0

    def _backstop_due(self, now: float) -> bool:
        if self.backstop_interval_sec <= 0:
            return True
        return self._last_backstop is None or (now - self._last_backstop) >= self.backstop_interval_sec

    def fetch(self, conn: Any, *, bounded_sql: Any, unbounded_sql: Any) -> Optional[Mapping[str, Any]]:
        """Run the bounded query; fall back to the rate-limited unbounded one.

        `bounded_sql` is executed with a single bind parameter `window_sec`.
        """
        row = None
        if self.bounded:
            row = conn.execute(bounded_sql, {"window_sec": self.window_sec}).mappings().first()
            if row is not None:
                return row

        now = self._monotonic()
        if not self.bounded or self._backstop_due(now):
            self._last_backstop = now
            row = conn.execute(unbounded_sql).mappings().first()
            if row is not None and self.bounded:
                # The tripwire. A row the fast path could not see means this stage is further
                # behind than its own window -- the one condition under which the bound could
                # lose work. Recovering quietly would hide exactly that.
                self._log.warning(
                    "pending_scan_backstop_hit stage=%s generated_at=%s window_sec=%.0f "
                    "-- unprocessed row older than the scan window; this stage is behind. "
                    "Raise the window if this repeats.",
                    self.label,
                    _row_get(row, "generated_at"),
                    self.window_sec,
                )
        return row


def _row_get(row: Mapping[str, Any], key: str) -> Any:
    """Tolerate a bounded query that does not select the diagnostic column."""
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return None
