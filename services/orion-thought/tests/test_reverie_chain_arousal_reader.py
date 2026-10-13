"""R3 reader 5: strained arousal stretches the pause between reverie chains; nothing shortens it.

Must-not: arousal never shortens the interval, never reaches the theme refractory or the step
cap, and unknown / flag off / Redis failure is exactly today's interval.
"""
from __future__ import annotations

import asyncio
import inspect
import types

import pytest

from orion.regulation.arousal_view import UNKNOWN_DISABLED, ArousalView


@pytest.mark.parametrize("level,mult,expected", [
    ("strained", 2.0, 180.0),
    ("strained", 1.0, 90.0),
    ("strained", 0.5, 90.0),   # can never shorten
    ("engaged", 2.0, 90.0),
    ("idle", 2.0, 90.0),
    (None, 2.0, 90.0),         # unknown
])
def test_sleep_by_level(level, mult, expected):
    from app import chain
    assert chain.reverie_chain_sleep_sec(90.0, level, mult) == expected


@pytest.mark.parametrize("level", ["engaged", "idle", "strained", None])
@pytest.mark.parametrize("mult", [0.1, 1.0, 2.0, 5.0])
def test_must_not_never_shorter_than_base(level, mult):
    from app import chain
    assert chain.reverie_chain_sleep_sec(90.0, level, mult) >= 90.0


def _patch(monkeypatch, *, enabled=True, view=None, mult=2.0):
    from app import chain
    seen = []

    async def fake_read(bus, *, enabled, reader, max_age_sec, **_kw):
        seen.append((enabled, reader, max_age_sec))
        return view if enabled else UNKNOWN_DISABLED

    monkeypatch.setattr(chain, "read_arousal_async", fake_read)
    monkeypatch.setattr(chain.settings, "reverie_chain_arousal_enabled", enabled, raising=False)
    monkeypatch.setattr(chain.settings, "reverie_chain_arousal_strained_multiplier", mult, raising=False)
    monkeypatch.setattr(chain.settings, "regulation_reader_max_age_sec", 360.0, raising=False)
    monkeypatch.setattr(chain.settings, "reverie_interval_sec", 90.0, raising=False)
    return chain, seen


def test_next_sleep_strained_doubles(monkeypatch):
    chain, seen = _patch(monkeypatch, view=ArousalView(level="strained", reason="strained"))
    assert asyncio.run(chain._next_chain_sleep_sec(object())) == 180.0
    assert seen == [(True, "reverie_chain", 360.0)]


@pytest.mark.parametrize("view", [ArousalView(level="unknown", reason="stale"),
                                  ArousalView(level="unknown", reason="read_failed"),
                                  ArousalView(level="engaged", reason="engaged"),
                                  ArousalView(level="idle", reason="idle")])
def test_next_sleep_other_levels_are_today(monkeypatch, view):
    chain, _ = _patch(monkeypatch, view=view)
    assert asyncio.run(chain._next_chain_sleep_sec(object())) == 90.0


def test_flag_off_is_today_even_when_strained(monkeypatch):
    chain, _ = _patch(monkeypatch, enabled=False, view=ArousalView(level="strained", reason="strained"))
    assert asyncio.run(chain._next_chain_sleep_sec(object())) == 90.0


def test_worker_reads_arousal_after_the_chain_and_sleeps_by_it(monkeypatch):
    """The read happens at the pause (after the chain ran), and the pause uses its value."""
    chain, _ = _patch(monkeypatch, view=ArousalView(level="strained", reason="strained"))
    order, timeouts = [], []
    monkeypatch.setattr(chain.settings, "reverie_chain_enabled", True, raising=False)
    monkeypatch.setattr(chain.settings, "orion_bus_enabled", True, raising=False)

    class FakeBus:
        def __init__(self, url=None):
            pass

        async def connect(self):
            pass

        async def close(self):
            pass

    async def fake_chain(*a, **kw):
        order.append("chain")

    real_next = chain._next_chain_sleep_sec

    async def spy_next(bus):
        order.append("arousal")
        return await real_next(bus)

    async def wait_for(aw, timeout):
        # Record the pause and end it as if stop was set, so the worker exits its loop.
        timeouts.append(timeout)
        aw.close()
        return True

    monkeypatch.setattr(chain, "OrionBusAsync", FakeBus)
    monkeypatch.setattr(chain, "DbRefractoryStore", lambda: None)
    monkeypatch.setattr(chain, "run_reverie_chain", fake_chain)
    monkeypatch.setattr(chain, "fold_bus", lambda bus: None)
    monkeypatch.setattr(chain, "_next_chain_sleep_sec", spy_next)
    monkeypatch.setattr(chain, "asyncio", types.SimpleNamespace(
        Event=asyncio.Event, wait_for=wait_for, sleep=asyncio.sleep,
        TimeoutError=asyncio.TimeoutError, CancelledError=asyncio.CancelledError))

    async def run():
        stop = asyncio.Event()
        await asyncio.wait_for(chain.run_reverie_chain_worker(stop), timeout=1.0)

    asyncio.run(run())
    assert order[:2] == ["chain", "arousal"]
    assert timeouts[0] == 180.0


def test_must_not_refractory_and_step_cap_do_not_read_arousal():
    from app import chain
    src = inspect.getsource(chain.run_reverie_chain) + inspect.getsource(chain.resolve_reverie_chain_stuck_loop)
    assert "arousal" not in src
