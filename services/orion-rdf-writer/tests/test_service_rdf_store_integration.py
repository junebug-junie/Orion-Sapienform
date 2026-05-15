from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(SERVICE_ROOT)]


@pytest.fixture
def svc_sync(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("ORION_BUS_URL", "redis://example.test/0")
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb.example")
    monkeypatch.setenv("RDF_WRITE_ASYNC_ENABLED", "false")
    monkeypatch.setenv("RDF_SKIP_KINDS", "")
    monkeypatch.setenv("RDF_WRITE_DEAD_LETTER_PATH", str(tmp_path / "dl.ndjson"))
    import app.settings as app_settings
    import app.service as app_service

    importlib.reload(app_settings)
    importlib.reload(app_service)

    class FakeStore:
        backend = "fake"

        def __init__(self) -> None:
            self.writes: list[tuple[str, str | None]] = []

        async def write_graph(self, content: str, graph_name: str | None = None):
            from app.rdf_store import RdfWriteResult

            self.writes.append((content, graph_name))
            return RdfWriteResult(
                backend="fake",
                graph_name=graph_name,
                normalized_graph_uri=None,
                byte_count=len(content),
                status_code=200,
            )

        async def health(self) -> dict:
            return {}

    fake = FakeStore()
    monkeypatch.setattr(app_service, "build_rdf_store_client", lambda _s, _c: fake)
    asyncio.run(app_service.init_rdf_write_pipeline())
    monkeypatch.setattr(
        app_service,
        "build_triples_from_envelope",
        lambda _kind, _payload: ("<urn:x> <urn:y> <urn:z> .\n", "orion:chat"),
    )
    try:
        yield app_service, fake
    finally:
        asyncio.run(app_service.shutdown_rdf_write_pipeline())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "chat.history",
        "chat.history.message.v1",
        "cognition.trace",
        "metacognitive.trace.v1",
        "tags.enriched",
        "collapse.mirror.entry",
        "rdf.write.request",
        "world.pulse.graph.upsert.v1",
    ],
)
async def test_handle_envelope_writes_for_kind(kind: str, svc_sync, monkeypatch: pytest.MonkeyPatch) -> None:
    svc, fake = svc_sync
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    env = BaseEnvelope(
        kind=kind,
        source=ServiceRef(name="test", node=None, version=None),
        payload={"memory_status": "accepted", "memory_tier": "durable"},
    )
    fake.writes.clear()
    await svc.handle_envelope(env)
    assert len(fake.writes) == 1
    assert fake.writes[0][1] == "orion:chat"
