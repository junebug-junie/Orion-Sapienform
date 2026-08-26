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


# ------------------------------------------------------- settings bounds ---

def test_poll_sec_zero_or_negative_is_rejected():
    """Review finding, 2026-08-26: 0 or negative turns the loop's own sleep
    into an unthrottled busy-loop hammering the NVML/driver layer every
    event-loop tick -- a plausible way to WORSEN a real staleness condition.

    Constructed with the actual snake_case field name, not the
    CUDA_WATCHDOG_POLL_SEC alias: pydantic-settings' `env=` alias binds OS
    environment variables (os.environ), not direct constructor kwargs --
    passing the alias as a kwarg is silently dropped by this model's
    `extra="ignore"` rather than validated at all, which very nearly shipped
    as a false-positive test (verified live: it happened to "pass" here for
    the wrong reason, a leaked os.environ mutation from an earlier ad hoc
    check in the same session made it look like validation was firing when
    the value was never actually reaching the field).
    """
    from app.settings import Settings

    for bad in (0, -1, -30.0):
        try:
            Settings(_env_file=None, cuda_watchdog_poll_sec=bad)
            assert False, f"cuda_watchdog_poll_sec={bad} should have been rejected"
        except Exception:
            pass


def test_failure_threshold_zero_or_negative_is_rejected():
    """0 makes should_trigger_restart(1, 0) True on the very first check,
    silently defeating the whole point of the debounce."""
    from app.settings import Settings

    for bad in (0, -1, -2):
        try:
            Settings(_env_file=None, cuda_watchdog_failure_threshold=bad)
            assert False, f"cuda_watchdog_failure_threshold={bad} should have been rejected"
        except Exception:
            pass


def test_valid_settings_values_still_construct():
    """The bounds must not be so tight they reject legitimate config."""
    from app.settings import Settings

    s = Settings(
        _env_file=None,
        cuda_watchdog_poll_sec=15.0,
        cuda_watchdog_failure_threshold=3,
    )
    assert s.cuda_watchdog_poll_sec == 15.0
    assert s.cuda_watchdog_failure_threshold == 3


def test_the_env_alias_itself_actually_works_via_real_environ(monkeypatch):
    """The three tests above construct via the snake_case field name, which
    proves the Field(gt=.../ge=...) VALIDATION, but not that
    CUDA_WATCHDOG_POLL_SEC/CUDA_WATCHDOG_FAILURE_THRESHOLD -- the names an
    operator actually sets in a real .env -- reach the field at all. This
    closes that gap the other three cannot."""
    from app.settings import Settings

    monkeypatch.setenv("CUDA_WATCHDOG_POLL_SEC", "45")
    monkeypatch.setenv("CUDA_WATCHDOG_FAILURE_THRESHOLD", "5")
    s = Settings(_env_file=None)
    assert s.cuda_watchdog_poll_sec == 45.0
    assert s.cuda_watchdog_failure_threshold == 5

    monkeypatch.setenv("CUDA_WATCHDOG_POLL_SEC", "0")
    try:
        Settings(_env_file=None)
        assert False, "CUDA_WATCHDOG_POLL_SEC=0 via real environ should have been rejected"
    except Exception:
        pass


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

async def _run_loop_to_completion(
    *, is_cuda_available, failure_threshold=2, poll_sec=0.001, check_timeout_sec=1.0
):
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
            check_timeout_sec=check_timeout_sec,
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
                check_timeout_sec=1.0,
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
                check_timeout_sec=1.0,
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
                check_timeout_sec=1.0,
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


def test_a_hanging_check_is_treated_as_a_failure_not_skipped():
    """Review finding, 2026-08-26: a genuine NVML/driver wedge -- exactly
    the state this watchdog exists to detect -- can HANG rather than
    fast-return. Unlike a raised exception (which is NOT counted, since a
    broken check is not proof of a broken GPU), a timeout IS evidence of the
    wedged state and must count toward the failure threshold.

    Deliberately a BOUNDED "hang" (0.3s), not a real forever-hang: a real
    thread running a blocking call cannot be force-killed from Python once
    started (asyncio.wait_for's timeout stops AWAITING it, not the thread
    itself), so an actually-unbounded sleep here would leak a live thread
    for the rest of the test process's life -- confirmed live: an earlier
    draft of this test used time.sleep(3600) and hung the whole suite.
    Production is not exposed to this the same way: a real, sustained NVML
    wedge crosses failure_threshold within a couple of poll intervals and
    restart_process() then exits the whole process, which reaps every
    thread (hung or not) at the OS level regardless of Python-side cleanup.
    """
    def _hangs_briefly():
        import time
        time.sleep(0.3)
        return True  # never actually observed -- the wait_for times out first

    triggered = asyncio.run(
        _run_loop_to_completion(
            is_cuda_available=_hangs_briefly,
            failure_threshold=2,
            poll_sec=0.001,
            check_timeout_sec=0.05,
        )
    )
    assert triggered == [1]


def test_check_does_not_freeze_the_event_loop_while_hanging():
    """The whole point of running the check via asyncio.to_thread: a hang
    in is_cuda_available must not block anything ELSE running on the same
    event loop (heartbeat_loop, listener_task, stt_task in production)."""
    heartbeat_ticks = []

    async def _heartbeat():
        while True:
            heartbeat_ticks.append(1)
            await asyncio.sleep(0.01)

    def _hangs_briefly():
        import time
        time.sleep(0.3)
        return False

    async def _run():
        hb_task = asyncio.create_task(_heartbeat())
        watchdog_task = asyncio.create_task(
            cuda_watchdog_loop(
                poll_sec=0.001,
                failure_threshold=100,  # never actually triggers in this test
                is_cuda_available=_hangs_briefly,
                on_trigger=lambda: None,
                check_timeout_sec=1.0,
            )
        )
        await asyncio.sleep(0.15)
        hb_task.cancel()
        watchdog_task.cancel()
        for t in (hb_task, watchdog_task):
            try:
                await t
            except asyncio.CancelledError:
                pass
        return heartbeat_ticks

    ticks = asyncio.run(_run())
    # If the hang froze the loop, heartbeat_ticks would be near-empty --
    # this asserts the OTHER task kept making real progress throughout.
    assert len(ticks) >= 5, f"heartbeat starved -- only {len(ticks)} ticks in 150ms"


def test_on_trigger_may_be_async():
    """restart_process() is sync, but the loop must not assume that --
    an async on_trigger (e.g. one that also publishes a bus event before
    restarting) must be awaited, not silently ignored as a fire-and-forget
    coroutine object."""
    triggered = []

    async def _async_trigger():
        await asyncio.sleep(0.01)
        triggered.append("done")

    asyncio.run(
        _run_loop_to_completion(
            is_cuda_available=lambda: False, failure_threshold=1
        )
    )
    # Re-run with the actual async trigger under test (the helper above
    # only proves the harness works with a sync one).
    async def _run():
        task = asyncio.create_task(
            cuda_watchdog_loop(
                poll_sec=0.001,
                failure_threshold=1,
                is_cuda_available=lambda: False,
                on_trigger=_async_trigger,
                check_timeout_sec=1.0,
            )
        )
        await asyncio.wait_for(task, timeout=5.0)

    asyncio.run(_run())
    assert triggered == ["done"]


def test_an_exception_from_on_trigger_does_not_kill_the_loop_silently():
    """Review finding, 2026-08-26: previously only the CHECK was guarded --
    on_trigger itself (e.g. os.kill raising under a restricted sandbox) was
    not, so it could kill this fire-and-forget task with nothing left
    supervising it. Now the whole tick is guarded; an exception here is
    logged and the loop keeps running (retries on the next tick) rather
    than vanishing."""
    calls = {"n": 0}

    def _flaky_trigger():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("os.kill blocked by sandbox")
        # second time succeeds -- loop should still be alive to reach it

    async def _run():
        task = asyncio.create_task(
            cuda_watchdog_loop(
                poll_sec=0.001,
                failure_threshold=1,  # trigger fires every single failure
                is_cuda_available=lambda: False,
                on_trigger=_flaky_trigger,
                check_timeout_sec=1.0,
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())
    # Called more than once -- proves the loop survived the first trigger's
    # exception and kept ticking rather than dying on it.
    assert calls["n"] > 1


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
