"""scripts/gpu_pool_clear_gpu2.py: takes gpu1 + both gpu2 slots, refuses half a card, always releases."""
from __future__ import annotations

import asyncio
import importlib.util
import os
import signal
import sys
from pathlib import Path

from orion.evals.model_replay.pool_hold import HoldLedger
from orion.evals.model_replay.tests.test_pool_hold import GRANTS, FakePool

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gpu_pool_clear_gpu2.py"
_spec = importlib.util.spec_from_file_location("gpu_pool_clear_gpu2", _PATH)
clear = importlib.util.module_from_spec(_spec)
sys.modules["gpu_pool_clear_gpu2"] = clear
_spec.loader.exec_module(clear)

CHAT = {"role": "chat", "profile_name": "x", "url": "http://c:8011", "cards": ["gpu0"]}
FREE = {"leases": [{"role": "agent-gpu2", "status": "granted", "holder": clear.HOLDER}]}
BUSY = {"leases": [{"role": "agent-gpu2", "status": "granted", "holder": "cortex-exec"}]}


class _Transport:
    def __init__(self, pool):
        self.pool = pool

    def __call__(self, bus_url):
        return self

    async def __aenter__(self):
        return self.pool

    async def __aexit__(self, *exc):
        return None


def _run(monkeypatch, tmp_path, pool, states=None, extra=()):
    monkeypatch.setattr(clear, "BusControlTransport", _Transport(pool))
    monkeypatch.setattr(clear, "http_sweeper", lambda *a: None)
    seq = states if states is not None else [FREE]  # shared: a test can watch it drain
    monkeypatch.setattr(clear, "http_pool_state", lambda url: seq.pop(0) if len(seq) > 1 else seq[0])
    monkeypatch.setattr(clear.asyncio, "sleep", _no_sleep)
    monkeypatch.setenv("ORION_BUS_URL", "redis://x")
    return clear.main(["--minutes", "0.001", "--out", str(tmp_path), *extra])


_real_sleep = asyncio.sleep


async def _no_sleep(_s):
    await _real_sleep(0)


def test_free_slot_reads_the_pool_lease_table():
    assert clear.gpu2_has_free_slot(FREE)
    assert not clear.gpu2_has_free_slot(BUSY)
    assert clear.gpu2_has_free_slot({"leases": [{"role": "agent-gpu2", "status": "released", "holder": "x"}]})


def test_holds_gpu1_and_both_gpu2_slots_then_releases(monkeypatch, tmp_path):
    pool = FakePool()
    assert _run(monkeypatch, tmp_path, pool) == 0
    assert [v[1] for v in pool.verbs if v[0] == "hold"] == ["memory_distill", "agent", "agent"]
    assert pool.live == set()
    assert HoldLedger(tmp_path / "holds.jsonl").outstanding() == []


def test_second_hold_waits_until_gpu2_slot_is_free(monkeypatch, tmp_path):
    pool = FakePool()
    sent_while_busy = []
    orig = pool.hold

    async def hold(work_class, actor):
        sent_while_busy.append(len(states_left) > 1)
        return await orig(work_class, actor)

    states_left = [BUSY, BUSY, FREE]
    pool.hold = hold
    assert _run(monkeypatch, tmp_path, pool, states=states_left) == 0
    assert sent_while_busy == [True, True, False]  # the third hold only after gpu2 read free
    assert pool.live == set()


def test_second_hold_landing_elsewhere_is_retried(monkeypatch, tmp_path):
    pool = FakePool(plan=[GRANTS["memory_distill"], GRANTS["agent"], CHAT, GRANTS["agent"]])
    assert _run(monkeypatch, tmp_path, pool) == 0
    assert len([v for v in pool.verbs if v[0] == "hold"]) == 4
    assert pool.live == set()


def test_gpu2_second_slot_never_ours_releases_everything(monkeypatch, tmp_path):
    pool = FakePool(plan=[GRANTS["memory_distill"], GRANTS["agent"], CHAT, CHAT])
    assert _run(monkeypatch, tmp_path, pool, extra=["--extra-tries", "2"]) == 2
    assert pool.live == set()
    assert HoldLedger(tmp_path / "holds.jsonl").outstanding() == []


def test_signal_while_holding_releases(monkeypatch, tmp_path):
    pool = FakePool()
    real_wait_for = asyncio.wait_for

    async def wait_for(aw, timeout):
        assert len(pool.live) == 3  # READY: all held
        os.kill(os.getpid(), signal.SIGTERM)
        return await real_wait_for(aw, timeout)

    monkeypatch.setattr(clear.asyncio, "wait_for", wait_for)
    assert _run(monkeypatch, tmp_path, pool, extra=["--minutes", "60"]) == 0
    assert pool.live == set()


def test_no_bus_url_refuses(monkeypatch):
    monkeypatch.delenv("ORION_BUS_URL", raising=False)
    assert clear.main([]) == 2
