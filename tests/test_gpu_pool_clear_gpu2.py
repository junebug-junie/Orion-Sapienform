"""scripts/gpu_pool_clear_gpu2.py: takes gpu1 + both gpu2 slots, refuses half a card, always releases."""
from __future__ import annotations

import asyncio
import importlib.util
import os
import signal
import sys
from pathlib import Path

import pytest

from orion.evals.model_replay.pool_hold import HoldLedger, PoolHolds
from orion.evals.model_replay.tests.test_pool_hold import GRANTS, FakePool

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gpu_pool_clear_gpu2.py"
_spec = importlib.util.spec_from_file_location("gpu_pool_clear_gpu2", _PATH)
clear = importlib.util.module_from_spec(_spec)
sys.modules["gpu_pool_clear_gpu2"] = clear
_spec.loader.exec_module(clear)


class _Transport:
    def __init__(self, pool):
        self.pool = pool

    def __call__(self, bus_url):
        return self

    async def __aenter__(self):
        return self.pool

    async def __aexit__(self, *exc):
        return None


def _run(monkeypatch, tmp_path, pool, minutes="0.001", after_ready=None):
    monkeypatch.setattr(clear, "BusControlTransport", _Transport(pool))
    monkeypatch.setattr(clear, "http_sweeper", lambda *a: None)
    monkeypatch.setenv("ORION_BUS_URL", "redis://x")
    return clear.main(["--minutes", minutes, "--out", str(tmp_path)])


def test_specs_ask_for_both_gpu2_slots_without_changing_the_replay(tmp_path):
    pool = FakePool()

    async def go():
        async with PoolHolds(transport=pool, ledger=HoldLedger(tmp_path / "h.jsonl"), specs=clear.SPECS):
            assert len(pool.live) == 3
        async with PoolHolds(transport=pool, ledger=HoldLedger(tmp_path / "r.jsonl")):
            assert len(pool.live) == 2  # the replay's default is still one gpu2 hold

    asyncio.run(go())
    assert [v[1] for v in pool.verbs if v[0] == "hold"][:3] == ["memory_distill", "agent", "agent"]


def test_holds_gpu1_and_both_gpu2_slots_then_releases(monkeypatch, tmp_path):
    pool = FakePool()
    assert _run(monkeypatch, tmp_path, pool) == 0
    assert len([v for v in pool.verbs if v[0] == "hold"]) == 3
    assert pool.live == set()
    assert HoldLedger(tmp_path / "holds.jsonl").outstanding() == []


def test_second_gpu2_slot_landing_elsewhere_releases_everything(monkeypatch, tmp_path):
    chat = {"role": "chat", "profile_name": "x", "url": "http://c:8011", "cards": ["gpu0"]}
    pool = FakePool(plan=[GRANTS["memory_distill"], GRANTS["agent"], chat])
    assert _run(monkeypatch, tmp_path, pool) == 2
    assert pool.live == set()


def test_signal_while_holding_releases(monkeypatch, tmp_path):
    pool = FakePool()
    real_wait_for = asyncio.wait_for

    async def wait_for(aw, timeout):
        assert len(pool.live) == 3  # READY: all held
        os.kill(os.getpid(), signal.SIGTERM)
        return await real_wait_for(aw, timeout)

    monkeypatch.setattr(clear.asyncio, "wait_for", wait_for)
    assert _run(monkeypatch, tmp_path, pool, minutes="60") == 0
    assert pool.live == set()


def test_no_bus_url_refuses(monkeypatch):
    monkeypatch.delenv("ORION_BUS_URL", raising=False)
    assert clear.main([]) == 2
