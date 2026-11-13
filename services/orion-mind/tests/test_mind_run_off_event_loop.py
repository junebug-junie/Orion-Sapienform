"""/v1/mind/run must not freeze the event loop (2026-11-13: /health timed out for minutes while a
world-pulse read ran, the container went unhealthy, and Prime's ticks and Claude-tool consent asks
stalled with it)."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app import main


@pytest.fixture(autouse=True)
def _fresh_lock(monkeypatch):
    # asyncio.Lock binds to the loop of its first contended wait; each test (own asyncio.run) gets its own.
    monkeypatch.setattr(main, "_MIND_RUN_LOCK", asyncio.Lock())


def test_health_answers_while_a_mind_run_is_in_flight(monkeypatch):
    async def _scenario():
        started = threading.Event()

        def slow_run(body):
            started.set()
            time.sleep(0.6)  # a synchronous model wait, as run_mind does
            return "done"

        monkeypatch.setattr(main, "_run_mind_sync", slow_run)
        run = asyncio.create_task(main.mind_run(object()))
        while not started.is_set():
            await asyncio.sleep(0.01)
        t0 = time.monotonic()
        health = await asyncio.wait_for(main.health(), timeout=0.2)
        assert health["ok"] is True and time.monotonic() - t0 < 0.2
        assert not run.done()  # the run was still going when health answered
        assert await run == "done"

    asyncio.run(_scenario())



def test_runs_stay_one_at_a_time(monkeypatch):
    async def _scenario():
        active = 0
        peak = 0
        lock = threading.Lock()

        def run(body):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.1)
            with lock:
                active -= 1
            return body

        monkeypatch.setattr(main, "_run_mind_sync", run)
        out = await asyncio.gather(*(main.mind_run(i) for i in range(3)))
        assert out == [0, 1, 2] and peak == 1

    asyncio.run(_scenario())



def test_a_juniper_turn_is_tracked_from_the_worker_thread(monkeypatch):
    seen = []
    monkeypatch.setattr(main, "carries_live_turn", lambda body: True)
    monkeypatch.setattr(main, "run_mind", lambda body, **kw: seen.append(main.JUNIPER_TURNS.active()) or "ok")
    assert asyncio.run(main.mind_run(object())) == "ok"
    assert seen == [True] and main.JUNIPER_TURNS.active() is False


def test_a_cancelled_request_keeps_the_next_run_waiting_for_its_thread(monkeypatch):
    async def _scenario():
        active = 0
        peak = 0
        lock = threading.Lock()
        first_started = threading.Event()

        def run(body):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            if body == "first":
                first_started.set()
            time.sleep(0.3)
            with lock:
                active -= 1
            return body

        monkeypatch.setattr(main, "_run_mind_sync", run)
        first = asyncio.create_task(main.mind_run("first"))
        while not first_started.is_set():
            await asyncio.sleep(0.01)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await main.mind_run("second") == "second"
        assert peak == 1


    asyncio.run(_scenario())
