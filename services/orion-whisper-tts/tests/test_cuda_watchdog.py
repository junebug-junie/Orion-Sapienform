"""2026-08-26: GPU/CUDA liveness watchdog (app/cuda_watchdog.py).

Real incident this exists to close: this container's Coqui TTS backend
crashed on first real use with "CUDA is not available on this machine" while
STT (same container, same GPU) kept working -- a docker+nvidia-container-
toolkit staleness quirk that a plain container restart fixed. Nothing in
app code can fix the underlying quirk; what these tests hold the line on is
that the watchdog correctly turns "torch.cuda.is_available() went False" into
a clean process exit, and does so without false-triggering on a single
transient check failure, and without letting the check itself crash the loop.
"""

from __future__ import annotations

import asyncio
import os
import signal
from unittest.mock import patch

import pytest

from app.cuda_watchdog import (
    cuda_watchdog_loop,
    restart_process,
    should_trigger_restart,
)


# --------------------------------------------------------- pure decision ---

@pytest.mark.parametrize(
    "consecutive_failures,threshold,expected",
    [
        (0, 2, False),
        (1, 2, False),
        (2, 2, True),
        (3, 2, True),
        (1, 1, True),
        (0, 1, False),
    ],
)
def test_should_trigger_restart_matrix(consecutive_failures, threshold, expected):
    assert should_trigger_restart(consecutive_failures, threshold) is expected


# ------------------------------------------------------------- restart_process

def test_restart_process_sends_sigterm_to_self():
    """NOT os._exit(): SIGTERM lets the app's own shutdown() handler run
    (cancels other background tasks, closes the bus cleanly) before the
    process actually exits -- restart: unless-stopped should restart a
    container that shut down cleanly, not one killed mid-write."""
    with patch("app.cuda_watchdog.os.kill") as fake_kill:
        restart_process()
    fake_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)


# --------------------------------------------------------------- the loop --

async def _run_loop_to_completion(*, is_cuda_available, failure_threshold=2, poll_sec=0.001):
    """Runs the real loop with real asyncio.sleep (tiny) so cancellation and
    timing behave like production; only is_cuda_available/on_trigger are
    faked. Returns the list of on_trigger call counts (0 or 1) and stops the
    loop itself via on_trigger raising StopAsyncIteration-style sentinel --
    actually the loop already `return`s after on_trigger(), so we just await
    the task with a timeout as a safety net against a logic bug hanging the
    test suite.
    """
    triggered = []

    def _on_trigger():
        triggered.append(1)

    task = asyncio.create_task(
        cuda_watchdog_loop(
            poll_sec=poll_sec,
            failure_threshold=failure_threshold,
            is_cuda_available=is_cuda_available,
            on_trigger=_on_trigger,
        )
    )
    await asyncio.wait_for(task, timeout=5.0)
    return triggered


def test_single_failure_does_not_trigger_a_restart():
    """One bad check must not restart an otherwise-healthy service -- an
    isolated NVML hiccup is not proof of real staleness."""
    calls = {"n": 0}

    def _flaky_then_healthy():
        calls["n"] += 1
        return calls["n"] != 1  # False once, then True forever

    async def _run():
        triggered = []

        def _on_trigger():
            triggered.append(1)

        task = asyncio.create_task(
            cuda_watchdog_loop(
                poll_sec=0.001,
                failure_threshold=2,
                is_cuda_available=_flaky_then_healthy,
                on_trigger=_on_trigger,
            )
        )
        # Let a handful of ticks run, then cancel -- the loop never returns
        # on its own unless it triggers, since CUDA is healthy after tick 1.
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return triggered

    triggered = asyncio.run(_run())
    assert triggered == []


def test_two_consecutive_failures_triggers_restart():
    calls = {"n": 0}

    def _always_broken():
        calls["n"] += 1
        return False

    triggered = asyncio.run(
        _run_loop_to_completion(is_cuda_available=_always_broken, failure_threshold=2)
    )
    assert triggered == [1]
    # Fired on the 2nd failure, not the 1st or 3rd -- exact threshold match.
    assert calls["n"] == 2


def test_recovery_resets_the_counter_so_a_later_blip_is_not_pre_escalated():
    """Otherwise one early transient failure would leave every later single
    blip one check away from triggering, defeating the debounce entirely."""
    sequence = iter([False, True, True, False])  # 1 fail, recover, recover, 1 fail

    def _is_available():
        return next(sequence, True)

    async def _run():
        triggered = []
        task = asyncio.create_task(
            cuda_watchdog_loop(
                poll_sec=0.001,
                failure_threshold=2,
                is_cuda_available=_is_available,
                on_trigger=lambda: triggered.append(1),
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return triggered

    triggered = asyncio.run(_run())
    # Never two CONSECUTIVE failures in this sequence -- must never trigger.
    assert triggered == []


def test_a_broken_check_itself_does_not_count_as_a_cuda_failure():
    """The check raising is a different failure mode from the check
    succeeding and reporting False -- a broken CHECK is not proof of a
    broken GPU, and must not silently advance toward a restart."""
    calls = {"n": 0}

    def _raises_then_healthy():
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RuntimeError("transient check error")
        return True

    async def _run():
        triggered = []
        task = asyncio.create_task(
            cuda_watchdog_loop(
                poll_sec=0.001,
                failure_threshold=2,
                is_cuda_available=_raises_then_healthy,
                on_trigger=lambda: triggered.append(1),
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return triggered

    triggered = asyncio.run(_run())
    assert triggered == []


def test_loop_stops_cleanly_on_cancellation():
    """Same convention as heartbeat_loop: CancelledError is caught and the
    loop returns quietly, not re-raised into the shutdown handler."""
    async def _run():
        task = asyncio.create_task(
            cuda_watchdog_loop(
                poll_sec=10.0,  # long enough that we cancel mid-sleep
                failure_threshold=2,
                is_cuda_available=lambda: True,
                on_trigger=lambda: None,
            )
        )
        await asyncio.sleep(0.01)
        task.cancel()
        # Must not raise CancelledError back out.
        await task

    asyncio.run(_run())
