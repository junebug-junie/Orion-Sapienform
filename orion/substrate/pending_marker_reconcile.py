"""Bounded safety-net sweep for the substrate `*_pending` work-queue markers.

Three substrate runtimes (policy, execution-dispatch, feedback) track "this parent row still
needs a child frame" with a boolean marker that is cleared in the same transaction as the
child insert. Each runtime also runs a reconciler that sets the marker back to TRUE for any
row whose marker is FALSE but whose child does not exist -- the guard against silent work loss
from manual SQL, restores, or a future bug. It can only ADD work, never remove it.

Until 2026-09-25 that reconciler was an unbounded anti-join UPDATE over the full history of
both tables, every 15 minutes, in one write transaction. On athena (Postgres on a spinning
10k SAS disk) the three of them were the top I/O statements in the database -- feedback-
runtime's alone: mean 13.4 s, 184 M shared blocks read and 27 M temp blocks written across 329
calls -- and none of them found anything in 72 h of logs.

This module keeps the guarantee but splits it in two:

* The FREQUENT sweep (every ``interval_sec``) is one short UPDATE over parent rows generated
  inside ``window_sec``. It rides the parent's ``generated_at`` index and probes the child's
  foreign-key index per row, so its cost is O(rows in the window), not O(history). Those
  probes are random reads, which is why the default window is small (2 h): each parent row is
  re-checked window/interval times, and the marker is cleared minutes after generation in
  practice.
* The FULL sweep (at most once per ``full_sweep_interval_sec``, optionally only during
  ``full_sweep_hour_utc``) still covers all history, but as a READ-ONLY hash anti-join SELECT
  (sequential scans -- the cheap access pattern on a spinning disk, unlike a history-long
  chain of random index probes) that returns candidate ids, followed by short batched
  UPDATEs that re-check the condition per id. No write transaction ever spans the history.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import text
from sqlalchemy.engine import Engine

# Full-sweep slack when hour-gated: the sweep may fire again once `interval - this` has elapsed,
# so a sweep that ran late in its hour yesterday cannot push today's out of the hour entirely.
_HOUR_GATE_SLACK_SEC = 3600.0


@dataclass(frozen=True)
class PendingMarkerSpec:
    """Which parent/child pair a reconciler guards. Identifiers are code constants, never
    user input -- they are interpolated into SQL."""

    parent_table: str
    marker_column: str
    child_table: str
    child_fk_column: str
    # Log event prefix, e.g. "feedback_pending" -> "feedback_pending_reconciled".
    log_prefix: str
    # Human wording for the warning, e.g. "dispatch frames" / "feedback frame".
    parent_label: str
    child_label: str


# Rows re-queued per UPDATE in the full sweep: bounds each write transaction.
FULL_SWEEP_UPDATE_BATCH = 5000


def bounded_requeue_sql(spec: PendingMarkerSpec) -> str:
    """The frequent sweep: one UPDATE over rows generated inside the window.

    The bound is DB-relative (``now()``), so it is an index condition on ``generated_at`` and
    there is no clock skew between the container and Postgres. No upper bound: a row stamped
    slightly in the future is still covered.
    """
    return f"""
        UPDATE {spec.parent_table} p
           SET {spec.marker_column} = true
         WHERE p.generated_at >= now() - make_interval(secs => :window_sec)
           AND NOT p.{spec.marker_column}
           AND NOT EXISTS (
                 SELECT 1 FROM {spec.child_table} c
                  WHERE c.{spec.child_fk_column} = p.frame_id
           )
    """


def full_candidates_sql(spec: PendingMarkerSpec) -> str:
    """Read-only, whole-history candidate scan. Plans as a Hash Anti Join over two seq scans."""
    return f"""
        SELECT p.frame_id
          FROM {spec.parent_table} p
         WHERE NOT p.{spec.marker_column}
           AND NOT EXISTS (
                 SELECT 1 FROM {spec.child_table} c
                  WHERE c.{spec.child_fk_column} = p.frame_id
           )
    """


def full_requeue_batch_sql(spec: PendingMarkerSpec) -> str:
    """Re-queue one batch of candidate ids, re-checking the condition: a child written between
    the scan and this UPDATE must not get its parent re-queued."""
    return f"""
        UPDATE {spec.parent_table} p
           SET {spec.marker_column} = true
         WHERE p.frame_id = ANY(:ids)
           AND NOT p.{spec.marker_column}
           AND NOT EXISTS (
                 SELECT 1 FROM {spec.child_table} c
                  WHERE c.{spec.child_fk_column} = p.frame_id
           )
    """


class PendingMarkerReconciler:
    def __init__(
        self,
        spec: PendingMarkerSpec,
        *,
        interval_sec: float = 900.0,
        window_sec: float = 7200.0,
        full_sweep_interval_sec: float = 86400.0,
        full_sweep_hour_utc: int = 9,
        logger: logging.Logger | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be > 0")
        if not (-1 <= int(full_sweep_hour_utc) <= 23):
            raise ValueError("full_sweep_hour_utc must be -1 (any hour) or 0..23")
        self.spec = spec
        self.interval_sec = float(interval_sec)
        self.window_sec = float(window_sec)
        self.full_sweep_interval_sec = float(full_sweep_interval_sec)
        self.full_sweep_hour_utc = int(full_sweep_hour_utc)
        self._log = logger or logging.getLogger("orion.substrate.pending_marker_reconcile")
        self._monotonic = monotonic
        self._utcnow = utcnow
        # Seeded to NOW, not None: otherwise a sweep runs on the first tick of every process
        # start, and a crash loop would re-run it per restart. The full sweep is only ever
        # evaluated inside a due frequent sweep, so it inherits this protection.
        self.last_sweep_mono: float | None = monotonic()
        # Hour-gated: None, so the first matching hour after >= one interval_sec of uptime
        # runs it. Ungated: seeded to now, so it first runs one full interval after boot.
        self.last_full_sweep_mono: float | None = (
            None if self.full_sweep_hour_utc >= 0 else monotonic()
        )

    # -- scheduling -----------------------------------------------------------------------

    def full_sweep_due(self, now_mono: float) -> bool:
        if self.full_sweep_interval_sec <= 0:
            return False
        last = self.last_full_sweep_mono
        if self.full_sweep_hour_utc < 0:
            return last is None or (now_mono - last) >= self.full_sweep_interval_sec
        if self._utcnow().hour != self.full_sweep_hour_utc:
            return False
        min_gap = max(0.0, self.full_sweep_interval_sec - _HOUR_GATE_SLACK_SEC)
        return last is None or (now_mono - last) >= min_gap

    # -- execution ------------------------------------------------------------------------

    def run(self, engine: Engine, *, force: bool = False, full: bool = False) -> int:
        """Rate-limited entry point, safe to call every tick. Returns rows re-queued.

        ``force`` bypasses the frequent-sweep rate limit; ``full`` forces a full sweep.
        """
        now = self._monotonic()
        if not force and self.last_sweep_mono is not None:
            if (now - self.last_sweep_mono) < self.interval_sec:
                return 0
        self.last_sweep_mono = now
        if full or self.full_sweep_due(now):
            self.last_full_sweep_mono = now
            return self.full_sweep(engine)
        return self.bounded_sweep(engine)

    def bounded_sweep(self, engine: Engine) -> int:
        with engine.begin() as conn:
            result = conn.execute(
                text(bounded_requeue_sql(self.spec)), {"window_sec": self.window_sec}
            )
        requeued = int(result.rowcount or 0)
        self._warn_if_requeued(requeued, scope="window")
        return requeued

    def full_sweep(self, engine: Engine) -> int:
        started = time.monotonic()
        with engine.connect() as conn:
            ids = [str(i) for i in conn.execute(text(full_candidates_sql(self.spec))).scalars()]
        requeued = 0
        batches = 0
        for i in range(0, len(ids), FULL_SWEEP_UPDATE_BATCH):
            with engine.begin() as conn:
                result = conn.execute(
                    text(full_requeue_batch_sql(self.spec)),
                    {"ids": ids[i : i + FULL_SWEEP_UPDATE_BATCH]},
                )
            requeued += int(result.rowcount or 0)
            batches += 1
        self._warn_if_requeued(requeued, scope="full")
        self._log.info(
            "%s_full_sweep_done candidates=%s batches=%s requeued=%s elapsed_ms=%.0f",
            self.spec.log_prefix,
            len(ids),
            batches,
            requeued,
            (time.monotonic() - started) * 1000.0,
        )
        return requeued

    def _warn_if_requeued(self, requeued: int, *, scope: str) -> None:
        if requeued:
            self._log.warning(
                "%s_reconciled requeued=%s scope=%s -- %s had their pending marker cleared "
                "with no %s present. Work would have been lost.",
                self.spec.log_prefix,
                requeued,
                scope,
                self.spec.parent_label,
                self.spec.child_label,
            )
