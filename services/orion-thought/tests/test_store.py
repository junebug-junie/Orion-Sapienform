"""Tests for app.store's shared Postgres engine and startup pool warm-up.

Live-verified 2026-07-17: the first query against a freshly-created engine
pays a full TCP+auth handshake to Postgres (~400ms), enough to trip a caller
with a tight 400ms budget (orion-thought's drive_state_compact facet fetch)
on turn one of every fresh container start. `warm_pool()` fixes that by
opening one throwaway connection at startup, unconditionally, since every
`_get_engine()` caller in this service shares the one pool.
"""
from __future__ import annotations

import threading

import pytest


def _fresh_store():
    """Reimport app.store through the module object so monkeypatch.setattr
    targets the same module object the function under test resolves its
    globals through (same pattern as test_mind_drive_state_facet.py)."""
    import importlib

    import app.store as store

    importlib.reload(store)
    return store


# --- _get_engine: concurrent-construction race guard ---


def test_get_engine_returns_same_instance_across_calls() -> None:
    store = _fresh_store()

    class _FakeEngine:
        pass

    monkeypatch_created = []

    def _fake_create_engine(*_args, **_kwargs):
        engine = _FakeEngine()
        monkeypatch_created.append(engine)
        return engine

    import sqlalchemy

    original = sqlalchemy.create_engine
    sqlalchemy.create_engine = _fake_create_engine
    try:
        first = store._get_engine()
        second = store._get_engine()
    finally:
        sqlalchemy.create_engine = original

    assert first is second
    assert len(monkeypatch_created) == 1


def test_get_engine_concurrent_calls_construct_only_one_engine() -> None:
    """Two threads racing through the check-then-create-then-assign must not
    both win -- the lock added 2026-07-17 makes the second thread block until
    the first has assigned `_engine`, then reuse it instead of constructing a
    second, silently-discarded Engine/pool."""
    store = _fresh_store()
    created: list[object] = []
    entered = threading.Barrier(2, timeout=5.0)

    class _FakeEngine:
        pass

    def _fake_create_engine(*_args, **_kwargs):
        # Rendezvous both threads inside the lock-protected critical section's
        # construction call, so if the lock were absent, both would be here
        # concurrently and both would construct an engine.
        engine = _FakeEngine()
        created.append(engine)
        return engine

    import sqlalchemy

    original = sqlalchemy.create_engine
    sqlalchemy.create_engine = _fake_create_engine
    results: list[object] = []
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            entered.wait()
        except threading.BrokenBarrierError:
            pass
        try:
            results.append(store._get_engine())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        threads = [threading.Thread(target=_worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
    finally:
        sqlalchemy.create_engine = original

    assert not errors
    assert len(created) == 1, f"expected exactly one Engine constructed, got {len(created)}"
    assert results[0] is results[1]


# --- _warm_pool_sync: throwaway connection ---


def test_warm_pool_sync_executes_select_1(monkeypatch) -> None:
    store = _fresh_store()
    executed: list[str] = []

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, stmt):
            executed.append(str(stmt))

    class _FakeEngine:
        def connect(self):
            return _FakeConn()

    monkeypatch.setattr(store, "_get_engine", lambda: _FakeEngine())

    store._warm_pool_sync()
    assert any("SELECT 1" in stmt for stmt in executed)


def test_warm_pool_sync_never_raises_on_connect_failure(monkeypatch) -> None:
    store = _fresh_store()

    class _FakeEngine:
        def connect(self):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(store, "_get_engine", lambda: _FakeEngine())

    # Must not raise -- a DB that isn't reachable yet at boot must not fail startup.
    store._warm_pool_sync()


# --- warm_pool: bounded async wrapper ---


@pytest.mark.asyncio
async def test_warm_pool_never_raises_on_sync_side_exception(monkeypatch) -> None:
    store = _fresh_store()

    def _boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(store, "_warm_pool_sync", _boom)

    # Must not raise even if _warm_pool_sync's own internal guard somehow
    # didn't catch it (defense-in-depth, not expected in practice).
    await store.warm_pool()


@pytest.mark.asyncio
async def test_warm_pool_outer_except_covers_non_timeout_wrapper_failure(monkeypatch) -> None:
    """Exercises the outer `except Exception` in warm_pool -- the belt-and-
    suspenders guard on top of _warm_pool_sync's own internal try/except,
    for a failure in the asyncio scaffolding itself rather than the query."""
    store = _fresh_store()

    async def _boom_to_thread(*_args, **_kwargs):
        raise RuntimeError("executor unavailable")

    monkeypatch.setattr(store.asyncio, "to_thread", _boom_to_thread)

    # Must not raise -- the outer except Exception must catch this too.
    await store.warm_pool()
