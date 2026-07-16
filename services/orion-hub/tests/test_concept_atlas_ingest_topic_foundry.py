"""Tests for the topic-foundry -> concept-atlas ingestion route.

Covers ``POST /api/substrate/concepts/ingest-topic-foundry``
(``services/orion-hub/scripts/concept_atlas_routes.py``) and its HTTP client
(``services/orion-hub/scripts/topic_foundry_client.py``). Mirrors the
isolated-router testing convention already used by
``test_concept_atlas_routes.py``: build a minimal FastAPI app that only
includes ``concept_atlas_routes.router`` and monkeypatch collaborators
directly, rather than pulling in the full ``scripts.main`` app or requiring
a live topic-foundry service.

All topic-foundry HTTP calls are mocked at the ``requests.get`` boundary
inside ``scripts.topic_foundry_client`` using fixture payloads shaped exactly
like the real ``GET /runs``, ``GET /topics``, and
``GET /topics/{topic_id}/keywords`` responses (see
``services/orion-topic-foundry/app/routers/runs.py`` and
``.../routers/topics.py`` for the real shapes).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def _concept_atlas_test_app() -> FastAPI:
    from scripts.concept_atlas_routes import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client() -> TestClient:
    _ensure_hub_scripts_import_path()
    return TestClient(_concept_atlas_test_app())


FAKE_BASE_URL = "http://fake-topic-foundry:8615"
FAKE_RUN_ID = "11111111-1111-1111-1111-111111111111"


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self) -> Any:
        return self._payload


def _runs_payload(run_id: str = FAKE_RUN_ID) -> dict[str, Any]:
    # Shaped exactly like RunListPage/RunListItem from
    # services/orion-topic-foundry/app/models.py, as returned by
    # GET /runs?format=wrapped&status=complete&limit=1
    return {
        "items": [
            {
                "run_id": run_id,
                "status": "complete",
                "stage": "complete",
                "created_at": "2026-07-15T00:00:00Z",
                "started_at": "2026-07-15T00:00:01Z",
                "completed_at": "2026-07-15T00:05:00Z",
                "model": {"model_id": "22222222-2222-2222-2222-222222222222", "name": "m", "version": "v1", "stage": "active"},
                "dataset": {"dataset_id": "33333333-3333-3333-3333-333333333333", "name": "d", "source_table": "t"},
                "window": {"start_at": None, "end_at": None},
                "stats_summary": {
                    "docs_generated": 611,
                    "segments_generated": 611,
                    "cluster_count": 2,
                    "outlier_pct": 0.67,
                    "segments_enriched": 0,
                },
            }
        ],
        "limit": 1,
        "offset": 0,
        "total": 1,
    }


def _topics_payload_normal() -> dict[str, Any]:
    # Shaped exactly like TopicSummaryPage/TopicSummaryItem from
    # GET /topics?run_id=...&limit=200. Includes the HDBSCAN noise bucket
    # (topic_id=-1) and a below-min_doc_count topic (count=2 < default floor
    # of 3) to confirm both are excluded downstream.
    return {
        "items": [
            {"topic_id": -1, "count": 411, "outlier_pct": 1.0, "label": None},
            {"topic_id": 0, "count": 200, "outlier_pct": 0.0, "label": None},
            {"topic_id": 1, "count": 50, "outlier_pct": 0.1, "label": None},
            {"topic_id": 2, "count": 2, "outlier_pct": 0.0, "label": None},
        ],
        "limit": 200,
        "offset": 0,
        "total": 4,
    }


def _topics_payload_empty() -> dict[str, Any]:
    return {"items": [], "limit": 200, "offset": 0, "total": 0}


def _keywords_payload(topic_id: int) -> dict[str, Any]:
    fixtures = {
        0: ["like", "meow", "just", "user", "assistant", "juniper", "let", "hi"],
        1: ["python", "code", "bug"],
        2: ["rare", "stray"],
    }
    return {"topic_id": topic_id, "keywords": fixtures.get(topic_id, [])}


def _make_fake_get(*, topics_payload: dict[str, Any], run_id: str = FAKE_RUN_ID, unreachable: bool = False):
    calls: list[tuple[str, Optional[dict[str, Any]]]] = []

    def fake_get(url: str, params: Optional[dict[str, Any]] = None, timeout: Optional[float] = None):
        calls.append((url, params))
        if unreachable:
            raise requests.exceptions.ConnectionError("connection refused")
        if url.endswith("/runs"):
            return _FakeResponse(200, _runs_payload(run_id))
        if url.endswith("/topics"):
            return _FakeResponse(200, topics_payload)
        if "/topics/" in url and url.endswith("/keywords"):
            topic_id = int(url.rsplit("/", 2)[1])
            return _FakeResponse(200, _keywords_payload(topic_id))
        raise AssertionError(f"unexpected URL in fake_get: {url}")

    return fake_get, calls


def _patch_topic_foundry_client(monkeypatch: pytest.MonkeyPatch, fake_get) -> None:
    from scripts import topic_foundry_client as tfc

    monkeypatch.setattr(tfc.requests, "get", fake_get)


def _patch_base_url(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    from scripts import concept_atlas_routes

    monkeypatch.setattr(concept_atlas_routes.settings, "TOPIC_FOUNDRY_BASE_URL", url)


def _patch_store(monkeypatch: pytest.MonkeyPatch, store) -> None:
    from scripts import concept_atlas_routes

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: store)


# --- normal run: real concept nodes written, outlier + below-floor excluded ---


def test_ingest_normal_run_writes_concepts_excludes_outlier_and_below_floor(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, calls = _make_fake_get(topics_payload=_topics_payload_normal())
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()

    assert body["available"] is True
    assert body["run_id"] == FAKE_RUN_ID
    assert body["topics_fetched"] == 4
    # Only topic_id 0 and 1 survive: -1 is HDBSCAN noise, topic_id 2 (count=2)
    # is below the adapter's default min_doc_count floor of 3.
    assert body["concepts_written"] == 2
    assert body["evidence_nodes_written"] == 2
    assert body["edges_written"] == 2  # supports edges only; no co_occurs_with (empty segment_topic_map)

    snapshot = store.snapshot()
    concept_nodes = [n for n in snapshot.nodes.values() if n.node_kind == "concept"]
    evidence_nodes = [n for n in snapshot.nodes.values() if n.node_kind == "evidence"]
    assert len(concept_nodes) == 2
    assert len(evidence_nodes) == 2
    concept_topic_ids = {n.metadata.get("topic_id") for n in concept_nodes}
    assert concept_topic_ids == {0, 1}
    assert -1 not in concept_topic_ids
    assert 2 not in concept_topic_ids
    labels = {n.label for n in concept_nodes}
    assert any("like" in label or "python" in label for label in labels)

    # Never fetch keywords for the outlier bucket -- no HTTP call wasted on it.
    keyword_call_urls = [url for url, _params in calls if url.endswith("/keywords")]
    assert all("/topics/-1/" not in url for url in keyword_call_urls)
    assert len(keyword_call_urls) == 3  # topics 0, 1, 2 (adapter drops 2 later, but client fetches before filtering)


def test_ingest_is_idempotent_on_repeated_calls(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running ingestion for the same run must upsert, not duplicate, nodes/edges."""
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, _calls = _make_fake_get(topics_payload=_topics_payload_normal())
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r1 = client.post("/api/substrate/concepts/ingest-topic-foundry")
    r2 = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r1.status_code == 200 and r2.status_code == 200

    snapshot = store.snapshot()
    concept_nodes = [n for n in snapshot.nodes.values() if n.node_kind == "concept"]
    assert len(concept_nodes) == 2  # not 4 -- deterministic node_ids upsert in place


# --- degraded paths: never a 500, never a fabricated success ---


def test_ingest_topic_foundry_unreachable_degrades_honestly(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, _calls = _make_fake_get(topics_payload=_topics_payload_normal(), unreachable=True)
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200  # never a raw 500
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "topic_foundry_fetch_failed"
    assert "error" in body
    assert body["concepts_written"] == 0
    assert store.snapshot().nodes == {}  # nothing fabricated into the store


def test_ingest_empty_run_no_topics_degrades_honestly(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, _calls = _make_fake_get(topics_payload=_topics_payload_empty())
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "topic_foundry_no_usable_topics"
    assert body["run_id"] == FAKE_RUN_ID
    assert body["topics_fetched"] == 0
    assert body["concepts_written"] == 0
    assert store.snapshot().nodes == {}


def test_ingest_no_completed_run_degrades_honestly(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()

    def fake_get(url: str, params: Optional[dict[str, Any]] = None, timeout: Optional[float] = None):
        if url.endswith("/runs"):
            return _FakeResponse(200, {"items": [], "limit": 1, "offset": 0, "total": 0})
        raise AssertionError(f"unexpected URL: {url}")

    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "topic_foundry_fetch_failed"
    assert "topic_foundry_no_completed_run" in body["error"]


def test_ingest_substrate_store_unavailable_degrades_honestly(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import concept_atlas_routes

    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: None)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "substrate_store_unavailable"
    assert body["concepts_written"] == 0


def test_ingest_topic_foundry_base_url_not_configured_degrades_honestly(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    _patch_base_url(monkeypatch, "")
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "topic_foundry_base_url_not_configured"


# --- client-layer unit tests -------------------------------------------------


def test_client_fetch_run_topics_and_keywords_skips_outlier_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    fake_get, calls = _make_fake_get(topics_payload=_topics_payload_normal())
    monkeypatch.setattr(tfc.requests, "get", fake_get)

    result = tfc.fetch_run_topics_and_keywords(FAKE_BASE_URL)
    assert result["run_id"] == FAKE_RUN_ID
    assert -1 not in result["keywords_by_topic"]
    assert set(result["keywords_by_topic"].keys()) == {0, 1, 2}  # client fetches keywords before the adapter's min_doc_count filter


def test_client_keyword_fetch_failure_degrades_to_empty_list_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    def fake_get(url: str, params=None, timeout=None):
        if url.endswith("/runs"):
            return _FakeResponse(200, _runs_payload())
        if url.endswith("/topics"):
            return _FakeResponse(200, _topics_payload_normal())
        if url.endswith("/keywords"):
            raise requests.exceptions.Timeout("slow")
        raise AssertionError(url)

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    result = tfc.fetch_run_topics_and_keywords(FAKE_BASE_URL)
    assert result["keywords_by_topic"] == {0: [], 1: [], 2: []}


def test_client_no_completed_run_raises_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    def fake_get(url: str, params=None, timeout=None):
        assert url.endswith("/runs")
        return _FakeResponse(200, {"items": [], "limit": 1, "offset": 0, "total": 0})

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    with pytest.raises(tfc.TopicFoundryClientError):
        tfc.fetch_run_topics_and_keywords(FAKE_BASE_URL)
