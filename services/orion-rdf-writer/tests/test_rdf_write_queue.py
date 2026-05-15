from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(SERVICE_ROOT)]


def _reload_service_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    monkeypatch.setenv("ORION_BUS_URL", "redis://example.test/0")
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb.example")
    monkeypatch.setenv("RDF_WRITE_DEAD_LETTER_PATH", str(tmp_path / "dead.ndjson"))
    import app.settings as app_settings
    import app.service as app_service

    importlib.reload(app_settings)
    importlib.reload(app_service)
    return app_service


@pytest.mark.asyncio
async def test_retrying_write_succeeds_after_oserror(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    svc = _reload_service_modules(monkeypatch, tmp_path)
    from app.rdf_store import RdfWriteResult

    class Flaky:
        backend = "fake"

        def __init__(self) -> None:
            self.n = 0

        async def write_graph(self, content: str, graph_name: str | None = None) -> RdfWriteResult:
            self.n += 1
            if self.n < 2:
                raise OSError("transient")
            return RdfWriteResult(
                backend="fake",
                graph_name=graph_name,
                normalized_graph_uri=None,
                byte_count=len(content),
                status_code=200,
            )

        async def health(self) -> dict:
            return {}

    job = svc.RdfWriteJob(
        kind="k",
        graph_name="g",
        content=".",
        correlation_id=None,
        source=None,
        created_at=0.0,
        payload_fingerprint=None,
    )
    res = await svc._retrying_write(Flaky(), job)
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_retrying_write_no_retry_on_http_400(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    svc = _reload_service_modules(monkeypatch, tmp_path)

    class Client400:
        backend = "fake"

        async def write_graph(self, content: str, graph_name: str | None = None):
            req = httpx.Request("POST", "http://example.invalid/statements")
            resp = httpx.Response(400, request=req)
            raise httpx.HTTPStatusError("bad", request=req, response=resp)

        async def health(self) -> dict:
            return {}

    job = svc.RdfWriteJob(
        kind="k",
        graph_name=None,
        content=".",
        correlation_id=None,
        source=None,
        created_at=0.0,
        payload_fingerprint=None,
    )
    with pytest.raises(httpx.HTTPStatusError):
        await svc._retrying_write(Client400(), job)


@pytest.mark.asyncio
async def test_retrying_write_deadletters_on_total_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RDF_WRITE_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("RDF_WRITE_RETRY_BASE_DELAY_SEC", "0.01")
    monkeypatch.setenv("RDF_WRITE_RETRY_MAX_DELAY_SEC", "0.02")
    svc = _reload_service_modules(monkeypatch, tmp_path)

    class AlwaysFail:
        backend = "fake"

        async def write_graph(self, content: str, graph_name: str | None = None):
            raise OSError("nope")

        async def health(self) -> dict:
            return {}

    job = svc.RdfWriteJob(
        kind="k",
        graph_name=None,
        content="data",
        correlation_id=None,
        source=None,
        created_at=0.0,
        payload_fingerprint=None,
    )
    monkeypatch.setattr(svc, "_store", AlwaysFail())
    with pytest.raises(OSError):
        await svc._execute_write_job(job)
    dl = tmp_path / "dead.ndjson"
    text = dl.read_text(encoding="utf-8").strip()
    assert "rdf_write_failed" in text
    row = json.loads(text.splitlines()[-1])
    assert row.get("kind") == "rdf_write_failed"
    assert row.get("reason") == "write_failed"


@pytest.mark.asyncio
async def test_queue_full_deadletters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RDF_WRITE_ASYNC_ENABLED", "true")
    monkeypatch.setenv("RDF_WRITE_QUEUE_MAXSIZE", "1")
    monkeypatch.setenv("RDF_WRITE_WORKERS", "1")
    monkeypatch.setenv("RDF_WRITE_MAX_IN_FLIGHT", "8")
    svc = _reload_service_modules(monkeypatch, tmp_path)

    hold = asyncio.Event()
    released = asyncio.Event()

    class Blocking:
        backend = "fake"

        async def write_graph(self, content: str, graph_name: str | None = None):
            from app.rdf_store import RdfWriteResult

            if not hold.is_set():
                hold.set()
                await released.wait()
            return RdfWriteResult(
                backend="fake",
                graph_name=graph_name,
                normalized_graph_uri=None,
                byte_count=len(content),
                status_code=200,
            )

        async def health(self) -> dict:
            return {}

    monkeypatch.setattr(svc, "build_rdf_store_client", lambda _s, _c: Blocking())

    await svc.init_rdf_write_pipeline()
    try:
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        env = lambda: BaseEnvelope(
            kind="chat.history",
            source=ServiceRef(name="t", node=None, version=None),
            payload={},
        )
        monkeypatch.setattr(svc, "build_triples_from_envelope", lambda _k, _p: ("<urn:a> <urn:b> <urn:c> .\n", "orion:chat"))

        t1 = asyncio.create_task(svc.handle_envelope(env()))
        await asyncio.wait_for(hold.wait(), timeout=2.0)
        t2 = asyncio.create_task(svc.handle_envelope(env()))
        await asyncio.sleep(0.05)
        with pytest.raises(asyncio.QueueFull):
            await svc._push_to_rdf_store("x", "orion:chat", env=env())
        released.set()
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=5.0)
    finally:
        await svc.shutdown_rdf_write_pipeline()

    lines = (tmp_path / "dead.ndjson").read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(line).get("reason") == "queue_full" for line in lines)
