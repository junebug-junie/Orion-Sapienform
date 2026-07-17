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
    # Not 4: identity-key merge upserts in place across repeated same-run ingest.
    assert len(concept_nodes) == 2


def test_ingest_cross_run_same_label_merges_to_one_durable_concept(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two completed runs with different run/topic IDs but the same semantic label
    must reconcile to one durable concept node (not two run-scoped orphans).

    Uses identical normalized labels (same keyword-derived label from the real
    adapter) because the HTTP client does not expose topic centroids; exact-label
    identity is the resolver contract that works without embeddings today.
    """
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    run_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    run_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    # Same keyword set → same adapter-derived label ("coherence / identity / merge").
    shared_keywords = ["coherence", "identity", "merge"]
    topic_a = 0
    topic_b = 7

    def _single_topic_payload(topic_id: int) -> dict[str, Any]:
        return {
            "items": [
                {"topic_id": -1, "count": 10, "outlier_pct": 1.0, "label": None},
                {"topic_id": topic_id, "count": 50, "outlier_pct": 0.0, "label": None},
            ],
            "limit": 200,
            "offset": 0,
            "total": 2,
        }

    def _ingest_run(*, run_id: str, topic_id: int) -> dict[str, Any]:
        def fake_get(url: str, params: Optional[dict[str, Any]] = None, timeout: Optional[float] = None):
            if url.endswith("/runs"):
                return _FakeResponse(200, _runs_payload(run_id))
            if url.endswith("/topics"):
                return _FakeResponse(200, _single_topic_payload(topic_id))
            if "/topics/" in url and url.endswith("/keywords"):
                fetched_topic_id = int(url.rsplit("/", 2)[1])
                assert fetched_topic_id == topic_id
                return _FakeResponse(200, {"topic_id": topic_id, "keywords": shared_keywords})
            raise AssertionError(f"unexpected URL in fake_get: {url}")

        _patch_topic_foundry_client(monkeypatch, fake_get)
        r = client.post("/api/substrate/concepts/ingest-topic-foundry")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["run_id"] == run_id
        assert body["concepts_written"] == 1
        return body

    _ingest_run(run_id=run_a, topic_id=topic_a)
    canonical_concept_id = f"sub-concept-topicfoundry-{run_a}-{topic_a}"
    run_b_concept_id = f"sub-concept-topicfoundry-{run_b}-{topic_b}"

    _ingest_run(run_id=run_b, topic_id=topic_b)

    snapshot = store.snapshot()
    concept_nodes = [n for n in snapshot.nodes.values() if n.node_kind == "concept"]
    assert len(concept_nodes) == 1, (
        f"expected one durable concept after cross-run ingest, got {len(concept_nodes)}: "
        f"{[n.node_id for n in concept_nodes]}"
    )
    durable = concept_nodes[0]
    assert durable.node_id == canonical_concept_id
    assert durable.node_id != run_b_concept_id
    assert run_b_concept_id not in snapshot.nodes

    supports_edges = [e for e in snapshot.edges.values() if e.predicate == "supports"]
    assert len(supports_edges) == 2  # one evidence support per run
    for edge in supports_edges:
        assert edge.target.node_id == canonical_concept_id, (
            f"supports edge must target durable concept {canonical_concept_id}, "
            f"got {edge.target.node_id}"
        )
        assert edge.target.node_id != run_b_concept_id


def test_ingest_cross_run_similar_embeddings_merge_paraphrased_labels(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paraphrased labels with similar embeddings must merge at the Hub ingest route.

    The Topic Foundry HTTP client does not expose centroids yet, so this test
    injects ``topic_embeddings`` into the adapter call while keeping the real
    route + materializer + store-backed resolver path.
    """
    import orion.substrate.adapters.topic_foundry as tf_adapter
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    run_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    run_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    topic_a = 0
    topic_b = 7
    # Cosine ~0.9986 — same pair as orion/substrate/tests/test_reconcile.py.
    embeddings_by_run = {
        run_a: {topic_a: [1.0, 1.0, 0.0]},
        run_b: {topic_b: [1.0, 0.9, 0.0]},
    }
    labels_by_run = {
        run_a: "surface encodings",
        run_b: "surface-level representations",
    }

    real_map = tf_adapter.map_topic_foundry_run_to_substrate

    def map_with_embeddings(*, run_id, topics, keywords_by_topic, segment_topic_map=None, **kwargs):
        return real_map(
            run_id=run_id,
            topics=topics,
            keywords_by_topic=keywords_by_topic,
            segment_topic_map=segment_topic_map or {},
            topic_embeddings=embeddings_by_run.get(str(run_id), {}),
            **kwargs,
        )

    monkeypatch.setattr(tf_adapter, "map_topic_foundry_run_to_substrate", map_with_embeddings)

    def _ingest_run(*, run_id: str, topic_id: int, label: str) -> None:
        def fake_get(url: str, params: Optional[dict[str, Any]] = None, timeout: Optional[float] = None):
            if url.endswith("/runs"):
                return _FakeResponse(200, _runs_payload(run_id))
            if url.endswith("/topics"):
                return _FakeResponse(
                    200,
                    {
                        "items": [
                            {"topic_id": -1, "count": 10, "outlier_pct": 1.0, "label": None},
                            {"topic_id": topic_id, "count": 50, "outlier_pct": 0.0, "label": label},
                        ],
                        "limit": 200,
                        "offset": 0,
                        "total": 2,
                    },
                )
            if "/topics/" in url and url.endswith("/keywords"):
                return _FakeResponse(200, {"topic_id": topic_id, "keywords": []})
            raise AssertionError(f"unexpected URL in fake_get: {url}")

        _patch_topic_foundry_client(monkeypatch, fake_get)
        r = client.post("/api/substrate/concepts/ingest-topic-foundry")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["run_id"] == run_id
        assert body["concepts_written"] == 1

    _ingest_run(run_id=run_a, topic_id=topic_a, label=labels_by_run[run_a])
    canonical_concept_id = f"sub-concept-topicfoundry-{run_a}-{topic_a}"
    run_b_concept_id = f"sub-concept-topicfoundry-{run_b}-{topic_b}"

    _ingest_run(run_id=run_b, topic_id=topic_b, label=labels_by_run[run_b])

    snapshot = store.snapshot()
    concept_nodes = [n for n in snapshot.nodes.values() if n.node_kind == "concept"]
    assert len(concept_nodes) == 1, (
        f"expected one durable concept after embedding merge, got {len(concept_nodes)}: "
        f"{[n.node_id for n in concept_nodes]}"
    )
    durable = concept_nodes[0]
    assert durable.node_id == canonical_concept_id
    assert run_b_concept_id not in snapshot.nodes

    supports_edges = [e for e in snapshot.edges.values() if e.predicate == "supports"]
    assert len(supports_edges) == 2
    for edge in supports_edges:
        assert edge.target.node_id == canonical_concept_id


# --- partial write honesty: counters must match durable upserts ---


class _FailAfterNUpsertsStore:
    """Delegating store that succeeds for the first N upserts, then raises.

    Used to prove the ingest route reports precise partial progress when
    ``SubstrateGraphMaterializer.apply_record`` writes incrementally and then
    raises mid-record (it does not roll back earlier upserts).
    """

    def __init__(self, inner: Any, *, fail_after: int) -> None:
        self._inner = inner
        self._fail_after = fail_after
        self._upserts = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def upsert_node(self, *, identity_key: str | None, node: Any) -> None:
        if self._upserts >= self._fail_after:
            raise RuntimeError("simulated upsert_node failure after partial write")
        self._inner.upsert_node(identity_key=identity_key, node=node)
        self._upserts += 1

    def upsert_edge(self, *, identity_key: str, edge: Any) -> None:
        if self._upserts >= self._fail_after:
            raise RuntimeError("simulated upsert_edge failure after partial write")
        self._inner.upsert_edge(identity_key=identity_key, edge=edge)
        self._upserts += 1


def test_ingest_partial_store_write_reports_precise_successful_counts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If materialization fails mid-record after some upserts succeeded, the
    response must report those successful counts — not lie with all zeros.
    """
    from orion.substrate.store import InMemorySubstrateGraphStore

    # Adapter order for the normal fixture: concept0, evidence0, concept1, ...
    # Fail after the first upsert so one concept node is durable in the store.
    inner = InMemorySubstrateGraphStore()
    store = _FailAfterNUpsertsStore(inner, fail_after=1)
    fake_get, _calls = _make_fake_get(topics_payload=_topics_payload_normal())
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()

    snapshot = store.snapshot()
    concept_nodes = [n for n in snapshot.nodes.values() if n.node_kind == "concept"]
    evidence_nodes = [n for n in snapshot.nodes.values() if n.node_kind == "evidence"]
    assert len(concept_nodes) == 1
    assert len(evidence_nodes) == 0
    assert snapshot.edges == {}

    assert body["available"] is False
    assert body["reason"] == "substrate_store_write_failed"
    assert body["run_id"] == FAKE_RUN_ID
    assert "error" in body
    # Precise partial progress — not the pre-fix lie of all zeros.
    assert body["concepts_written"] == 1
    assert body["evidence_nodes_written"] == 0
    assert body["edges_written"] == 0


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


# --- post-ingestion typed-relation classification step -----------------------
#
# Covers the additive post-ingestion step added to
# concept_atlas_ingest_topic_foundry(): for co_occurs_with edges among current
# concept nodes that clear the "count" worth-classifying threshold
# (orion.substrate.relation_classification.is_worth_classifying), call the
# injected classifier and write any resulting typed edge. The real LLM
# classifier (scripts/concept_relation_classifier.py) is never invoked here --
# build_llm_relation_classifier is monkeypatched to a fake so these tests stay
# fast, deterministic, and network-free.


def _seed_relation_pair_nodes_and_edges(store: Any) -> dict[str, str]:
    """Pre-seed the store with three concept nodes and two co_occurs_with
    edges: one crossing the default count threshold (5), one below it.
    Returns the seeded node ids so tests can assert against them.
    """
    from orion.core.schemas.cognitive_substrate import ConceptNodeV1, NodeRefV1, SubstrateEdgeV1
    from orion.substrate.adapters._common import make_provenance, make_temporal

    def _node(node_id: str, label: str) -> ConceptNodeV1:
        return ConceptNodeV1(
            node_id=node_id,
            anchor_scope="world",
            label=label,
            temporal=make_temporal(observed_at=None),
            provenance=make_provenance(source_kind="test", source_channel="test", producer="test"),
        )

    def _co_occurs_edge(source_id: str, target_id: str, *, count: int) -> SubstrateEdgeV1:
        return SubstrateEdgeV1(
            source=NodeRefV1(node_id=source_id, node_kind="concept"),
            target=NodeRefV1(node_id=target_id, node_kind="concept"),
            predicate="co_occurs_with",
            temporal=make_temporal(observed_at=None),
            provenance=make_provenance(source_kind="test", source_channel="test", producer="test"),
            metadata={"co_occurrence_count": count},
        )

    node_a = _node("sub-node-seed-a", "seed concept a")
    node_b = _node("sub-node-seed-b", "seed concept b")
    node_c = _node("sub-node-seed-c", "seed concept c")
    store.upsert_node(identity_key=None, node=node_a)
    store.upsert_node(identity_key=None, node=node_b)
    store.upsert_node(identity_key=None, node=node_c)

    # Default DEFAULT_COUNT_THRESHOLD is 5 (relation_classification.py) -- 10
    # clears it, 2 does not.
    edge_ab = _co_occurs_edge(node_a.node_id, node_b.node_id, count=10)
    edge_bc = _co_occurs_edge(node_b.node_id, node_c.node_id, count=2)
    store.upsert_edge(identity_key="seed-ab", edge=edge_ab)
    store.upsert_edge(identity_key="seed-bc", edge=edge_bc)

    return {
        "node_a": node_a.node_id,
        "node_b": node_b.node_id,
        "node_c": node_c.node_id,
        "edge_ab": edge_ab.edge_id,
        "edge_bc": edge_bc.edge_id,
    }


def _patch_fake_relation_classifier(monkeypatch: pytest.MonkeyPatch, *, predicate: Optional[str]):
    """Monkeypatch build_llm_relation_classifier so no real LLM/bus call ever
    happens. Returns the list of (source_id, target_id, edge_id) tuples the
    fake classifier was actually invoked with, for call-count assertions.
    """
    from scripts import concept_relation_classifier as crc

    calls: list[tuple[str, str, str]] = []

    def fake_build_llm_relation_classifier(pairs, *, settings, timeout_sec=None, route=None):
        def _classifier(node_a, node_b, edge):
            calls.append((node_a.node_id, node_b.node_id, edge.edge_id))
            return predicate

        return _classifier

    monkeypatch.setattr(crc, "build_llm_relation_classifier", fake_build_llm_relation_classifier)
    return calls


def test_ingest_classifies_only_pairs_crossing_count_threshold(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    seeded = _seed_relation_pair_nodes_and_edges(store)
    calls = _patch_fake_relation_classifier(monkeypatch, predicate="supports")

    fake_get, _calls = _make_fake_get(topics_payload=_topics_payload_normal())
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True

    # Only the A-B pair (count=10) clears the default threshold of 5; B-C
    # (count=2) must never reach the classifier.
    assert len(calls) == 1
    assert calls[0][0] == seeded["node_a"]
    assert calls[0][1] == seeded["node_b"]
    assert calls[0][2] == seeded["edge_ab"]

    assert body["typed_edges_written"] == 1

    # classify_relation() stamps metadata["source_edge_id"] with the
    # co_occurs_with edge it classified (relation_classification.py) -- use
    # that as the unique marker for "the typed edge our new step wrote",
    # since the normal ingest fixture also produces unrelated evidence-
    # >concept "supports" edges from the materializer itself.
    snapshot = store.snapshot()
    typed_edges = [
        e
        for e in snapshot.edges.values()
        if e.predicate == "supports" and e.metadata.get("source_edge_id") == seeded["edge_ab"]
    ]
    assert len(typed_edges) == 1
    assert typed_edges[0].source.node_id == seeded["node_a"]
    assert typed_edges[0].target.node_id == seeded["node_b"]


def test_ingest_classifier_none_result_writes_no_typed_edge(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A classifier that returns None (no confident relation) must not
    produce a typed edge, but must still be honestly reported as 0, not an
    error."""
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    seeded = _seed_relation_pair_nodes_and_edges(store)
    calls = _patch_fake_relation_classifier(monkeypatch, predicate=None)

    fake_get, _calls = _make_fake_get(topics_payload=_topics_payload_normal())
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True

    assert len(calls) == 1  # still invoked for the qualifying pair
    assert body["typed_edges_written"] == 0

    snapshot = store.snapshot()
    typed_edges = [
        e for e in snapshot.edges.values() if e.metadata.get("source_edge_id") == seeded["edge_ab"]
    ]
    assert typed_edges == []


def test_ingest_no_co_occurs_edges_reports_zero_typed_edges_without_classifier_call(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normal ingest fixture never produces co_occurs_with edges
    (segment_topic_map is empty by design) -- the classifier must never be
    invoked, and typed_edges_written must be an honest 0, not omitted."""
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    calls = _patch_fake_relation_classifier(monkeypatch, predicate="supports")

    fake_get, _calls = _make_fake_get(topics_payload=_topics_payload_normal())
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["typed_edges_written"] == 0
    assert calls == []


def test_ingest_second_call_does_not_duplicate_or_reclassify_already_typed_pair(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: calling the ingest route twice against the same store must
    not (a) write a second, duplicate typed edge for a pair already classified,
    or (b) spend another classifier call on it. Before the deterministic
    edge_id fix (relation_classification.py) and the already-classified
    filter (_typed_relation_classification_candidates), every call re-built a
    fresh-uuid edge_id for the same pair (accumulating unbounded duplicates in
    the store) and re-spent the LLM budget reclassifying pairs it already had
    an answer for."""
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    seeded = _seed_relation_pair_nodes_and_edges(store)
    calls = _patch_fake_relation_classifier(monkeypatch, predicate="supports")

    fake_get, _calls = _make_fake_get(topics_payload=_topics_payload_normal())
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r1 = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r1.status_code == 200
    assert r1.json()["typed_edges_written"] == 1
    assert len(calls) == 1

    r2 = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r2.status_code == 200
    # Second call must not reclassify the already-typed A-B pair.
    assert r2.json()["typed_edges_written"] == 0
    assert len(calls) == 1  # classifier not invoked again

    snapshot = store.snapshot()
    typed_edges = [
        e
        for e in snapshot.edges.values()
        if e.predicate == "supports" and e.metadata.get("source_edge_id") == seeded["edge_ab"]
    ]
    # Exactly one typed edge for this pair -- not two.
    assert len(typed_edges) == 1
