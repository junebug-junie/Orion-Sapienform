"""Periodic grammar/substrate retention.

Retention already existed and was correct per-run; its only trigger was process start, so
it could not converge against continuous arrival. Measured live 2026-08-19/20 before this
module existed:

    arrival      1,117,440 rows/day across the four managed tables (2.42 GB/day)
    deletion       365,000 rows per process start, then exactly 0 until the next one
    standing debt  5,352,878 rows already past the cutoff, and growing

Breaking even would have needed ~3 restarts a day; clearing the backlog, ~15 more on top.
No cap tuning reaches that. This runs the same bounded pass on a timer instead.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import threading

from app.grammar_truth import run_one_retention_cycle
from app.settings import Settings

logger = logging.getLogger("sql-writer.grammar_retention_loop")


def retention_days_for(settings: Settings) -> dict[str, int]:
    return {
        "grammar_events": int(getattr(settings, "grammar_events_retention_days", 0) or 0),
        "grammar_edges": int(getattr(settings, "grammar_edges_retention_days", 0) or 0),
        "grammar_atoms": int(getattr(settings, "grammar_atoms_retention_days", 0) or 0),
        "substrate_organ_emissions": int(
            getattr(settings, "substrate_organ_emissions_retention_days", 0) or 0
        ),
    }


async def grammar_retention_loop(settings: Settings) -> None:
    """Run a bounded retention cycle every `grammar_retention_interval_sec`.

    Sleeps FIRST. The startup retention pass in main.py has just run with the much larger
    startup caps, and starting a second pass immediately would stack disk load at exactly
    the moment the service is also replaying its bus backlog.
    """
    interval = float(getattr(settings, "grammar_retention_interval_sec", 0.0) or 0.0)
    if interval <= 0:
        logger.warning(
            "grammar_retention_loop DISABLED (GRAMMAR_RETENTION_INTERVAL_SEC=%s); "
            "retention reverts to startup-only, which cannot keep up with arrival",
            interval,
        )
        return

    max_batches = int(getattr(settings, "grammar_retention_periodic_max_batches", 3) or 3)
    max_elapsed = float(
        getattr(settings, "grammar_retention_periodic_max_elapsed_sec", 20.0) or 20.0
    )
    days_for = retention_days_for(settings)
    logger.info(
        "grammar_retention_loop starting interval_sec=%.0f max_batches=%s "
        "max_elapsed_sec=%.0f days=%s",
        interval,
        max_batches,
        max_elapsed,
        days_for,
    )

    # Handed to the worker thread so a shutdown can stop it between tables. to_thread is
    # not cancellable once started, so cancelling the task alone leaves the thread running.
    stop = threading.Event()

    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            stop.set()
            raise

        try:
            # to_thread, not a direct call: run_one_retention_cycle uses blocking
            # SQLAlchemy connections, and this event loop is also draining the bus. A
            # ~20s in-loop stall would stop consuming events, which is a far worse
            # failure than retention running a cycle late.
            states = await asyncio.to_thread(
                run_one_retention_cycle,
                days_for=days_for,
                max_batches=max_batches,
                max_elapsed_sec=max_elapsed,
                stop=stop,
            )
        except asyncio.CancelledError:
            # Signal the worker, then give it a bounded moment to finish the batch it is
            # inside. Shielded because we are already being cancelled; without the shield
            # the await returns instantly and we are back to joining at interpreter exit.
            stop.set()
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(asyncio.shield(_settle()), timeout=10.0)
            raise
        except Exception:
            # Never let one bad cycle end the loop -- that would silently restore the
            # startup-only behaviour this module exists to replace.
            logger.exception("grammar_retention_cycle_failed (loop continues)")
            continue

        pruned = sum(int(st.rows_pruned_last_run or 0) for st in states.values())
        debts = {t: st.remaining_debt for t, st in states.items() if st.remaining_debt}
        floored = [t for t, st in states.items() if st.cursor_floor_applied]
        skipped = [t for t, st in states.items() if st.failure_reason]

        # Unconditional. An earlier version logged only when pruned or debts were non-zero,
        # which went silent in exactly the state that needs a voice: floor pinned, nothing
        # pruned, debt reported against the clamped cutoff as 0. It also made "loop alive
        # and caught up" indistinguishable from "loop dead". One line per cycle is cheap.
        logger.info(
            "grammar_retention_cycle pruned=%s remaining_debt=%s floored=%s skipped=%s",
            pruned,
            debts or "{}",
            floored or "[]",
            skipped or "[]",
        )
        if floored:
            logger.warning(
                "grammar_retention_floor_pinned tables=%s -- a reducer lane is behind and "
                "retention is deliberately holding back; disk will grow until it catches up",
                floored,
            )


async def _settle() -> None:
    """Yield briefly so a stopping worker thread can observe the stop flag."""
    await asyncio.sleep(0.1)
