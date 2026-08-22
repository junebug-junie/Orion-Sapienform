"""Tests for the loop wrapper's control flow -- separate from the reducer's
own logic tests, this file is about the async plumbing: does disabling it via
config actually disable it, and does cancellation actually stop it.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.vision_object_permanence_loop import vision_object_permanence_loop  # noqa: E402


class _Settings:
    vision_permanence_sweep_interval_sec = 1800.0
    postgres_uri = "postgresql://x"
    vision_permanence_lookback_ceiling_sec = 3600.0
    vision_permanence_absence_fraction = 0.1
    vision_permanence_min_absence_sec = 3600.0
    vision_permanence_max_absence_sec = 86400.0


@pytest.mark.asyncio
async def test_zero_interval_disables_the_loop_and_returns_immediately() -> None:
    s = _Settings()
    s.vision_permanence_sweep_interval_sec = 0.0
    await asyncio.wait_for(vision_object_permanence_loop(s), timeout=1.0)


@pytest.mark.asyncio
async def test_missing_postgres_uri_disables_the_loop() -> None:
    s = _Settings()
    s.postgres_uri = ""
    await asyncio.wait_for(vision_object_permanence_loop(s), timeout=1.0)


@pytest.mark.asyncio
async def test_cancellation_during_sleep_actually_stops_the_task() -> None:
    """The bug this pins: an earlier draft suppressed CancelledError during
    the sleep, which fell through into running one more sweep cycle after
    being told to stop. A cancelled task must raise CancelledError, not
    swallow it and continue.

    Confirmed live against the real bug: reverting the fix (suppressing
    CancelledError so the loop falls through into a real sweep cycle) made
    this test genuinely HANG rather than fail. `run_one_sweep_cycle` runs in
    `asyncio.to_thread`; that blocking thread is not interruptible once
    started, so once the cancellation is swallowed once, the coroutine is
    simply running normally from then on -- asyncio cancellation is one-shot,
    a suppressed CancelledError does not re-arm itself at the next await
    point. `asyncio.wait_for` alone (no `shield`, which was tried here first
    and made no difference for the same reason) at least bounds how long a
    regression can hang THIS test, even though the underlying thread is still
    leaked in that failure case -- acceptable for a test process that exits
    right after.
    """
    s = _Settings()
    s.vision_permanence_sweep_interval_sec = 3600.0   # long enough it won't fire on its own
    task = asyncio.create_task(vision_object_permanence_loop(s))
    await asyncio.sleep(0.05)   # let it reach the sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)
    assert task.cancelled() or task.done()
