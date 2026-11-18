"""vision-edge's heartbeat and /health report the real host, not "vision-edge-node".

Equilibrium keys its state by `service@node`; before this patch both surfaces sent
the made-up literal "vision-edge-node" (live 2026-10-20 as
`vision-edge@vision-edge-node`). The loop itself stays hand-rolled on purpose: the
shared HeartbeatOnly chassis always reports status="ok", and this loop reports
"degraded" while the camera is blind (2026-10-11 frozen-frame incident).
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def edge(monkeypatch):
    """Imports app.main/app.routes against a stub app.context (the real one opens the
    camera and loads torch detectors at import), same approach as
    test_listener_resubscribes.py."""
    monkeypatch.setenv("SOURCE", "rtsp://test-camera")
    edge_settings = importlib.import_module("app.settings")
    edge_settings.get_settings.cache_clear()
    camera = SimpleNamespace(get_frame=lambda: None, frame_age_sec=lambda: None, reopens=0)
    context = SimpleNamespace(
        settings=edge_settings.get_settings(), bus=AsyncMock(), camera=camera, detectors=[]
    )
    monkeypatch.setitem(sys.modules, "app.context", context)
    before = set(sys.modules)
    for name in ("app.main", "app.routes", "app.detector_worker", "app.capture_worker"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    main = importlib.import_module("app.main")
    routes = importlib.import_module("app.routes")
    try:
        yield SimpleNamespace(main=main, routes=routes, settings=context.settings, camera=camera)
    finally:
        for key in set(sys.modules) - before:
            if key.startswith("app."):
                sys.modules.pop(key, None)
        edge_settings.get_settings.cache_clear()


async def _one_tick(monkeypatch, edge, frame):
    bus = AsyncMock()
    monkeypatch.setattr(edge.camera, "get_frame", lambda: frame)

    async def _stop_after_first(_sec):
        raise asyncio.CancelledError

    monkeypatch.setattr(edge.main.asyncio, "sleep", _stop_after_first)
    await edge.main.heartbeat_loop(bus)
    bus.publish.assert_awaited_once()
    channel, env = bus.publish.await_args.args
    assert channel == "orion:system:health"
    return env.payload


@pytest.mark.parametrize("frame,status", [(object(), "ok"), (None, "degraded")])
def test_heartbeat_reports_real_node_and_camera_status(monkeypatch, edge, frame, status) -> None:
    monkeypatch.setattr(edge.settings, "NODE_NAME", "athena")
    payload = asyncio.run(_one_tick(monkeypatch, edge, frame))
    assert payload["service"] == edge.settings.SERVICE_NAME == "vision-edge"
    assert payload["node"] == "athena"
    assert payload["status"] == status


def test_health_route_reports_same_node(monkeypatch, edge) -> None:
    monkeypatch.setattr(edge.settings, "NODE_NAME", "circe")
    body = asyncio.run(edge.routes.health())
    assert body["node"] == "circe"
