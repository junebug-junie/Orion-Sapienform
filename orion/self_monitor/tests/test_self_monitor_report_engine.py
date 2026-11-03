from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from orion.self_monitor import report


class FakeEngine:
    def __init__(self, url: str):
        self.url = url
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


@pytest.fixture(autouse=True)
def empty_engine_cache():
    report._dispose_cached_engines()
    yield
    report._dispose_cached_engines()


def install_factory(monkeypatch, factory):
    monkeypatch.setitem(sys.modules, "sqlalchemy", SimpleNamespace(create_engine=factory))


def test_concurrent_first_use_builds_exactly_one_engine(monkeypatch):
    calls: list[str] = []
    calls_lock = threading.Lock()

    def create_engine(url, **_kwargs):
        with calls_lock:
            calls.append(url)
        time.sleep(0.02)
        return FakeEngine(url)

    install_factory(monkeypatch, create_engine)
    with ThreadPoolExecutor(max_workers=8) as executor:
        engines = list(executor.map(lambda _index: report._engine("postgresql://one"), range(16)))

    assert len(calls) == 1
    assert all(engine is engines[0] for engine in engines)


def test_cache_is_bounded_without_evicting_live_engines(monkeypatch):
    created: list[FakeEngine] = []

    def create_engine(url, **_kwargs):
        engine = FakeEngine(url)
        created.append(engine)
        return engine

    install_factory(monkeypatch, create_engine)
    for index in range(report._ENGINE_CACHE_MAX):
        report._engine(f"postgresql://{index}")

    with pytest.raises(RuntimeError, match="engine cache is full"):
        report._engine("postgresql://overflow")
    assert not any(engine.disposed for engine in created)

    report._dispose_cached_engines()
    assert all(engine.disposed for engine in created)
    assert report._ENGINE_CACHE == {}


def test_factory_failure_does_not_consume_a_cache_slot(monkeypatch):
    attempts = 0

    def create_engine(url, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("database driver unavailable")
        return FakeEngine(url)

    install_factory(monkeypatch, create_engine)
    with pytest.raises(OSError, match="driver unavailable"):
        report._engine("postgresql://retry")
    assert "postgresql://retry" not in report._ENGINE_CACHE

    engine = report._engine("postgresql://retry")
    assert isinstance(engine, FakeEngine)
    assert attempts == 2


def test_engine_preserves_read_only_pool_settings(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def create_engine(url, **kwargs):
        calls.append((url, kwargs))
        return FakeEngine(url)

    install_factory(monkeypatch, create_engine)
    report._engine("postgresql://settings")

    assert calls == [
        (
            "postgresql://settings",
            {
                "pool_pre_ping": True,
                "pool_size": 1,
                "max_overflow": 0,
                "connect_args": {
                    "connect_timeout": 5,
                    "options": "-c statement_timeout=60000 -c default_transaction_read_only=on",
                },
            },
        )
    ]
