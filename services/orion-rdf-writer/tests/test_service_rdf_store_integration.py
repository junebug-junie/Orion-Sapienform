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


def test_social_room_stored_event_materializes_turn_and_evidence():
    """Real (unmocked) rdf_builder dispatch for social.turn.stored.v1 --
    moved from test_autonomy_materialization.py (deleted 2026-07-18) since
    it tests a live handler unrelated to the autonomy artifact kinds that
    file used to cover."""
    from rdflib import Graph, Literal, Namespace
    from rdflib.namespace import RDF, XSD

    import app.rdf_builder as rdf_builder_module

    ORION = Namespace("http://conjourney.net/orion#")

    nt, graph_name = rdf_builder_module.build_triples_from_envelope(
        "social.turn.stored.v1",
        {
            "turn_id": "social-turn-1",
            "correlation_id": "corr-social-1",
            "session_id": "sid-social",
            "source": "hub_ws",
            "profile": "social_room",
            "prompt": "stay with me socially",
            "response": "I’m right here.",
            "text": "User: stay with me socially\nOrion: I’m right here.",
            "created_at": "2026-03-21T12:00:00+00:00",
            "stored_at": "2026-03-21T12:00:01+00:00",
            "recall_profile": "social.room.v1",
            "trace_verb": "chat_social_room",
            "tags": ["social_room", "chat_social_room"],
            "concept_evidence": [
                {
                    "ref_id": "identity-1",
                    "source_kind": "memory.identity.snapshot.v1",
                    "summary": "Oríon maintained peer continuity.",
                    "confidence": 0.8,
                }
            ],
            "grounding_state": {
                "profile": "social_room",
                "identity_label": "Oríon",
                "relationship_frame": "peer",
                "self_model_hint": "distributed social presence",
                "continuity_anchor": "Juniper ↔ Oríon ongoing peer dialogue",
                "stance": "warm, direct, grounded",
            },
            "redaction": {
                "prompt_score": 0.0,
                "response_score": 0.0,
                "memory_score": 0.0,
                "overall_score": 0.0,
                "recall_safe": True,
                "redaction_level": "low",
                "reasons": [],
            },
            "client_meta": {"chat_profile": "social_room"},
        },
    )
    graph = Graph()
    graph.parse(data=nt, format="nt")

    assert graph_name == "orion:chat:social"
    turn_uri = next(graph.subjects(ORION.artifactId, Literal("social-turn-1", datatype=XSD.string)))
    assert (turn_uri, RDF.type, ORION.SocialRoomTurn) in graph
    assert (turn_uri, ORION.recallSafe, Literal(True, datatype=XSD.boolean)) in graph
    assert list(graph.objects(turn_uri, ORION.supportedByEvidence))
