"""Regression: the shared substrate store is built once and never on the event loop.

Live 2026-10-20: at boot six tick threads each hydrated their own Falkor store and a
loop-thread caller built one inline, freezing the event loop ~3 min. Every bus
subscription timed out and chat lost finalize appraisal.
"""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from app import post_turn_closure_listener as ptc
from app.worker import BiometricsSubstrateWorker


def _worker() -> BiometricsSubstrateWorker:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._substrate_graph_store = None
    return worker


def _slow_build(calls: list[str], seconds: float = 0.2):
    def build():
        calls.append(threading.current_thread().name)
        time.sleep(seconds)
        return object()

    return build


def test_concurrent_threads_build_the_store_once(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("orion.substrate.graphdb_store.build_substrate_store_from_env", _slow_build(calls))
    worker = _worker()
    results: list[object] = []
    threads = [
        threading.Thread(target=lambda: results.append(worker._get_substrate_graph_store(log_label="t")))
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert len(calls) == 1
    assert len(results) == 6 and len({id(r) for r in results}) == 1


def test_loop_thread_caller_never_builds_inline(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("orion.substrate.graphdb_store.build_substrate_store_from_env", _slow_build(calls, 0.5))
    worker = _worker()

    async def on_loop():
        started = time.monotonic()
        store = worker._get_substrate_graph_store(log_label="loop")
        return store, time.monotonic() - started

    store, elapsed = asyncio.run(on_loop())
    assert store is None
    assert elapsed < 0.1  # returned at once instead of waiting out the 0.5 s build
    deadline = time.monotonic() + 5
    while worker._substrate_graph_store is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert worker._substrate_graph_store is not None
    assert calls == ["substrate-store-warm"]


def test_failed_warm_build_is_retried_by_the_next_caller(monkeypatch):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("falkor down")
        return object()

    monkeypatch.setattr("orion.substrate.graphdb_store.build_substrate_store_from_env", flaky)
    worker = _worker()
    assert worker._get_substrate_graph_store(log_label="t") is None
    assert worker._get_substrate_graph_store(log_label="t") is not None
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_post_turn_closure_handler_runs_off_the_event_loop(monkeypatch):
    seen: list[bool] = []

    def on_closure(closure):
        try:
            asyncio.get_running_loop()
            seen.append(True)
        except RuntimeError:
            seen.append(False)

    closure = SimpleNamespace(
        correlation_id="c", surprise_unresolved=True, outcome_molecule_id="o", grammar_event_ids=[]
    )
    await ptc.handle_post_turn_closure_message(closure, on_closure=on_closure)
    assert seen == [False]
