from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from orion.schemas.causal_geometry import CausalGeometrySnapshotV1
from orion.substrate import causal_geometry_bus_publish as publish_module

BASE_TS = datetime(2026, 7, 16, tzinfo=timezone.utc)


def _snapshot() -> CausalGeometrySnapshotV1:
    return CausalGeometrySnapshotV1(
        snapshot_id="snap-1",
        generated_at=BASE_TS,
        window_start=BASE_TS,
        window_end=BASE_TS,
        insufficient_data=True,
        notes=["no significant edges"],
    )


class _FakeBus:
    def __init__(self, url: str, *, enabled: bool) -> None:
        self.url = url
        self.enabled = enabled
        self.connected = False
        self.closed = False
        self.published: list[tuple[str, Any]] = []

    async def connect(self) -> None:
        self.connected = True

    async def publish(self, channel: str, msg: Any) -> None:
        self.published.append((channel, msg))

    async def close(self) -> None:
        self.closed = True


class _BoomingBus:
    def __init__(self, url: str, *, enabled: bool) -> None:
        pass

    async def connect(self) -> None:
        raise RuntimeError("bus unreachable")


def test_publish_snapshot_connects_publishes_and_closes(monkeypatch) -> None:
    created: list[_FakeBus] = []

    def _factory(url: str, *, enabled: bool) -> _FakeBus:
        bus = _FakeBus(url, enabled=enabled)
        created.append(bus)
        return bus

    monkeypatch.setattr(publish_module, "OrionBusAsync", _factory)

    result = publish_module.publish_snapshot(
        bus_url="redis://unused/0",
        bus_enabled=True,
        snapshot=_snapshot(),
        service_name="orion-field-digester",
        service_version="0.1.0",
        node_name="athena",
    )

    assert result == {"ok": True, "error": None}
    assert len(created) == 1
    bus = created[0]
    assert bus.connected is True
    assert bus.closed is True
    assert len(bus.published) == 1
    channel, envelope = bus.published[0]
    assert channel == publish_module.CAUSAL_GEOMETRY_SNAPSHOT_CHANNEL
    assert envelope.kind == publish_module.CAUSAL_GEOMETRY_SNAPSHOT_KIND
    assert envelope.payload["snapshot_id"] == "snap-1"
    assert envelope.source.name == "orion-field-digester"


def test_publish_snapshot_disabled_is_a_noop_no_bus_constructed(monkeypatch) -> None:
    mock_factory = AsyncMock()
    monkeypatch.setattr(publish_module, "OrionBusAsync", mock_factory)

    result = publish_module.publish_snapshot(
        bus_url="redis://unused/0", bus_enabled=False, snapshot=_snapshot()
    )

    assert result == {"ok": False, "error": "bus_disabled"}
    mock_factory.assert_not_called()


def test_publish_snapshot_never_raises_on_connect_failure(monkeypatch) -> None:
    monkeypatch.setattr(publish_module, "OrionBusAsync", _BoomingBus)

    result = publish_module.publish_snapshot(
        bus_url="redis://unused/0", bus_enabled=True, snapshot=_snapshot()
    )

    assert result["ok"] is False
    assert "bus unreachable" in result["error"]
