from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = ROOT / "services" / "orion-spark-concept-induction"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app import main as concept_main  # noqa: E402


class FakeWorker:
    instances: list["FakeWorker"] = []

    def __init__(self, _settings):
        self.started = asyncio.Event()
        self.stop_calls = 0
        self.start_calls = 0
        FakeWorker.instances.append(self)

    async def start(self) -> None:
        self.start_calls += 1
        self.started.set()
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise

    async def stop(self) -> None:
        self.stop_calls += 1

    def trigger_status(self) -> dict:
        return {"trigger": "active", "start_calls": self.start_calls}

    def worker_liveness_status(self) -> dict:
        return {"worker_initialized": True, "worker_task_created": True, "bus_consumer_started": True}

    def mark_task_created(self) -> None:
        return None

    def record_task_exit(self, _task) -> None:
        return None


def test_lifespan_attaches_worker_and_starts_task(monkeypatch) -> None:
    FakeWorker.instances = []
    monkeypatch.setattr(concept_main, "ConceptWorker", FakeWorker)

    with TestClient(concept_main.app):
        assert hasattr(concept_main.app.state, "worker")
        assert concept_main.app.state.worker is FakeWorker.instances[0]
        assert concept_main.app.state.concept_worker is FakeWorker.instances[0]
        assert hasattr(concept_main.app.state, "worker_task")
        assert concept_main.app.state.worker_task.done() is False
        assert FakeWorker.instances[0].started.is_set() is True


def test_debug_endpoint_returns_structured_json_when_worker_present(monkeypatch) -> None:
    FakeWorker.instances = []
    monkeypatch.setattr(concept_main, "ConceptWorker", FakeWorker)

    with TestClient(concept_main.app) as client:
        response = client.get("/debug/concept-induction")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["worker_attached"] is True
    assert payload["worker_liveness"]["worker_initialized"] is True
    assert payload["worker_task"]["worker_task_present"] is True
    assert payload["status"]["trigger"] == "active"


def test_shutdown_stops_worker_once(monkeypatch) -> None:
    FakeWorker.instances = []
    monkeypatch.setattr(concept_main, "ConceptWorker", FakeWorker)

    with TestClient(concept_main.app):
        pass

    assert FakeWorker.instances[0].stop_calls == 1


def test_debug_route_reports_missing_worker_explicitly() -> None:
    original_worker = getattr(concept_main.app.state, "worker", None)
    original_concept_worker = getattr(concept_main.app.state, "concept_worker", None)

    if hasattr(concept_main.app.state, "worker"):
        delattr(concept_main.app.state, "worker")
    if hasattr(concept_main.app.state, "concept_worker"):
        delattr(concept_main.app.state, "concept_worker")

    try:
        response = asyncio.run(concept_main.concept_induction_debug())
    finally:
        if original_worker is not None:
            concept_main.app.state.worker = original_worker
        if original_concept_worker is not None:
            concept_main.app.state.concept_worker = original_concept_worker

    assert response == {"ok": False, "error": "worker_not_initialized"}
