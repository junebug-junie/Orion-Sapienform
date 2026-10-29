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
    async def sweep():  # what the pool itself shows live for us
        return sorted(pool.live)

    monkeypatch.setattr(clear, "BusControlTransport", _Transport(pool))
    monkeypatch.setattr(clear, "http_sweeper", lambda *a: sweep)
    seq = states if states is not None else [FREE]  # shared: a test can watch it drain
    monkeypatch.setattr(clear, "http_pool_state", lambda url: seq.pop(0) if len(seq) > 1 else seq[0])
    monkeypatch.setattr(clear, "POLL_SEC", 0)
    monkeypatch.setattr(clear, "RETRY_SEC", 0)
    monkeypatch.setenv("ORION_BUS_URL", "redis://x")
    return clear.main(["--minutes", "0.001", "--out", str(tmp_path), *extra])


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


def test_signal_while_holding_releases(monkeypatch, tmp_path, capsys):
    pool = FakePool()
    orig = pool.hold

    async def hold(work_class, actor):
        res = await orig(work_class, actor)
        if pool.n == 3:  # the last hold: SIGTERM shortly after READY
            asyncio.get_running_loop().call_later(0.05, os.kill, os.getpid(), signal.SIGTERM)
        return res

    pool.hold = hold
    assert _run(monkeypatch, tmp_path, pool, extra=["--minutes", "60"]) == 0
    assert "READY" in capsys.readouterr().out
    assert pool.live == set()


def test_second_hold_rpc_dying_mid_flight_is_swept(monkeypatch, tmp_path):
    # The pool created the lease, but the reply never came (bus timeout / Ctrl+C mid-RPC).
    pool = FakePool()
    orig = pool.hold

    async def hold(work_class, actor):
        res = await orig(work_class, actor)
        if pool.n == 3:
            raise TimeoutError("rpc reply lost")
        return res

    pool.hold = hold
    try:
        _run(monkeypatch, tmp_path, pool)
    except TimeoutError:
        pass
    assert pool.live == set()


def test_failed_release_is_reported_not_called_released(monkeypatch, tmp_path, capsys):
    pool = FakePool()

    async def stuck(lease_id, actor):
        return {"ok": False, "reason": "timeout"}

    pool.release = stuck
    pool.cancel = stuck
    assert _run(monkeypatch, tmp_path, pool) == 4
    err = capsys.readouterr().err
    assert "NOT RELEASED" in err and "--release-leftovers" in err


def test_no_bus_url_refuses(monkeypatch):
    monkeypatch.delenv("ORION_BUS_URL", raising=False)
    assert clear.main([]) == 2
