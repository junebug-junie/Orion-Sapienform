import asyncio
from pathlib import Path
import sys
import types
import types
import logging

ROOT = Path(__file__).resolve().parents[3]
APP_PATH = ROOT / "services" / "orion-recall"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(APP_PATH) not in sys.path:
    sys.path.append(str(APP_PATH))
if str(APP_PATH / "app") not in sys.path:
    sys.path.append(str(APP_PATH / "app"))

sys.modules.setdefault("loguru", types.SimpleNamespace(logger=logging.getLogger("test")))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # type: ignore  # noqa: E402
from orion.core.contracts.recall import RecallQueryV1  # type: ignore  # noqa: E402
from worker import handle_recall, process_recall  # type: ignore  # noqa: E402


class _BusStub:
    def __init__(self):
        self.published = []

    async def publish(self, channel, msg):
        self.published.append((channel, msg))


def test_recall_handler_returns_bundle():
    bus = _BusStub()
    env = BaseEnvelope(
        kind="recall.query.v1",
        source=ServiceRef(name="test", version="1", node="n1"),
        payload=RecallQueryV1(fragment="test fragment", profile="reflect.v1").model_dump(mode="json"),
    )
    # Avoid hitting real backends during unit test
    import settings as recall_settings  # type: ignore

    recall_settings.settings.RECALL_ENABLE_VECTOR = False
    recall_settings.settings.RECALL_ENABLE_RDF = False
    recall_settings.settings.RECALL_ENABLE_SQL_TIMELINE = False

    res = asyncio.get_event_loop().run_until_complete(handle_recall(env, bus=bus))
    assert res.kind == "recall.reply.v1"
    assert isinstance(res.payload, dict)
    assert "bundle" in res.payload
    assert bus.published  # telemetry fired


def test_process_recall_includes_sql(monkeypatch):
    bus = _BusStub()

    async def fake_fetch_recent(session_id, node_id, since_minutes, limit):
        return [
            types.SimpleNamespace(
                id="sql1",
                ts=1.0,
                source="sql_timeline",
                title=None,
                text="sql fragment",
                tags=["sql"],
                session_id=session_id,
                node_id=node_id,
                source_ref="collapse_mirror",
            )
        ]

    async def fake_fetch_related(entities, since_hours, limit):
        return []

    monkeypatch.setattr("worker.fetch_recent_fragments", fake_fetch_recent)
    monkeypatch.setattr("worker.fetch_related_by_entities", fake_fetch_related)
    monkeypatch.setattr("worker.get_profile", lambda name: {"enable_sql_timeline": True, "sql_top_k": 5})

    q = RecallQueryV1(fragment="sql test", profile="reflect.v1")
    bundle, decision = asyncio.get_event_loop().run_until_complete(process_recall(q, corr_id="c1"))
    assert any(item.source == "sql_timeline" for item in bundle.items)
