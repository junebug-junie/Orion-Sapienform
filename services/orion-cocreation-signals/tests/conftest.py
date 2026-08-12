"""Shared test fixtures for orion-cocreation-signals producer tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


@dataclass
class FakeRedis:
    """Minimal in-memory stand-in for the real ``aioredis.Redis`` client
    exposed by ``OrionBusAsync.redis`` -- just enough (``get``/``set``) for
    producers that persist durable state (e.g. doc_semantic_drift_loop's
    ``last_sha``). Backed by a plain dict a test can share across two
    separate ``FakeBus`` instances to simulate state surviving a real
    process restart."""

    store: dict

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str) -> None:
        self.store[key] = value


@dataclass
class FakeBus:
    """Records every publish call instead of touching a real Redis connection.
    ``enabled`` mirrors OrionBusAsync's own attribute the producers check."""

    enabled: bool = True
    published: list[tuple[str, BaseEnvelope]] = field(default_factory=list)
    closed: bool = False
    redis_store: dict = field(default_factory=dict)

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        self.published.append((channel, envelope))

    async def close(self) -> None:
        # Mirrors OrionBusAsync.close() -- lets producers that fork a
        # dedicated RPC client (e.g. doc_semantic_drift_loop) exercise their
        # real teardown path in tests instead of hitting an AttributeError
        # that only happens to get swallowed by a broad except Exception.
        self.closed = True

    @property
    def redis(self) -> FakeRedis:
        return FakeRedis(self.redis_store)


@pytest.fixture
def fake_bus() -> FakeBus:
    return FakeBus()


@pytest.fixture
def source() -> ServiceRef:
    return ServiceRef(name="cocreation-signals", version="0.1.0", node="test-node")


@pytest.fixture
def stop_event() -> asyncio.Event:
    return asyncio.Event()
