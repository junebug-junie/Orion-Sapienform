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


def _segments_payload_empty(run_id: str = FAKE_RUN_ID) -> dict[str, Any]:
    return {"run_id": run_id, "items": [], "limit": 1000, "offset": 0, "total": 0}


def _kg_edges_payload_empty(run_id: str = FAKE_RUN_ID) -> dict[str, Any]:
    # Shaped like KgEdgeListPage (services/orion-topic-foundry/app/models.py).
    return {"run_id": run_id, "items": [], "limit": 500, "offset": 0}


def _kg_edges_payload_mentions(run_id: str = FAKE_RUN_ID) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "items": [
            {
                "edge_id": "44444444-4444-4444-4444-444444444444",
                "segment_id": "seg-0a",
                "subject": "m",
                "predicate": "mentions",
                "object": "Juniper Feld",
                "confidence": 0.6,
                "created_at": "2026-07-15T09:00:01Z",
            }
        ],
        "limit": 500,
        "offset": 0,
    }


def _make_fake_get(
    *,
    topics_payload: dict[str, Any],
    run_id: str = FAKE_RUN_ID,
    unreachable: bool = False,
    segments_payload: Optional[dict[str, Any]] = None,
    segments_unreachable: bool = False,
    kg_edges_payload: Optional[dict[str, Any]] = None,
    kg_edges_unreachable: bool = False,
):
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
        if url.endswith("/kg/edges"):
            if kg_edges_unreachable:
                raise requests.exceptions.ConnectionError("connection refused")
            return _FakeResponse(200, kg_edges_payload if kg_edges_payload is not None else _kg_edges_payload_empty(run_id))
        if url.endswith("/segments"):
            if segments_unreachable:
                raise requests.exceptions.ConnectionError("connection refused")
            return _FakeResponse(200, segments_payload if segments_payload is not None else _segments_payload_empty(run_id))
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
    assert body["edges_written"] == 2  # supports edges only; no co_occurs_with (no segments fixture -> empty map)
    assert body["segments_fetched"] == 0
    assert body["segment_topic_map_buckets"] == 0

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

    # Code review 2026-08-18: the "latest completed run" lookup must be
    # scoped to this scheduler's own model, or ingestion can silently keep
    # reading a *different* model's runs (e.g. an old, unfiltered one).
    from scripts import concept_atlas_routes as car

    runs_calls = [params for url, params in calls if url.endswith("/runs")]
    assert runs_calls, "expected at least one /runs lookup"
    assert runs_calls[0] is not None
    assert runs_calls[0].get("model_name") == car._TOPIC_FOUNDRY_MODEL_NAME


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

    def upsert_node(
        self,
        *,
        identity_key: str | None,
        node: Any,
        skip_metadata_keys: Any = None,
    ) -> None:
        if self._upserts >= self._fail_after:
            raise RuntimeError("simulated upsert_node failure after partial write")
        self._inner.upsert_node(identity_key=identity_key, node=node, skip_metadata_keys=skip_metadata_keys)
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


# --- AI Town's own concept graph (2026-08-20) --------------------------------
# ingest-topic-foundry-aitown is the same _ingest_topic_foundry_run logic as
# the Orion route above, parameterized over which store/model it reads --
# these tests confirm it's actually wired to the AI Town store/model, not a
# copy-pasted duplicate that quietly points at the Orion ones.


def test_ingest_aitown_substrate_store_unavailable_degrades_honestly(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import concept_atlas_routes

    monkeypatch.setattr(concept_atlas_routes, "_get_aitown_substrate_store", lambda: None)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry-aitown")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["reason"] == "substrate_store_unavailable"
    assert body["concepts_written"] == 0


def test_ingest_aitown_route_writes_into_aitown_store_not_orion_store(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The load-bearing guarantee this whole feature exists for: AI Town
    ingestion must land in the AI Town store, not silently fall through to
    (or also write into) the Orion store -- and must scope its topic-foundry
    fetch to the AI Town model_name, not Orion's."""
    from orion.substrate.store import InMemorySubstrateGraphStore
    from scripts import concept_atlas_routes

    orion_store = InMemorySubstrateGraphStore()
    aitown_store = InMemorySubstrateGraphStore()
    monkeypatch.setattr(concept_atlas_routes, "_get_substrate_store", lambda: orion_store)
    monkeypatch.setattr(concept_atlas_routes, "_get_aitown_substrate_store", lambda: aitown_store)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)

    fake_get, calls = _make_fake_get(topics_payload=_topics_payload_normal())
    _patch_topic_foundry_client(monkeypatch, fake_get)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry-aitown")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["concepts_written"] > 0
    assert len(orion_store.snapshot().nodes) == 0
    assert len(aitown_store.snapshot().nodes) > 0

    runs_calls = [params for (url, params) in calls if url.endswith("/runs")]
    assert runs_calls, "expected at least one GET /runs call"
    assert all(
        (params or {}).get("model_name") == concept_atlas_routes._TOPIC_FOUNDRY_AITOWN_MODEL_NAME
        for params in runs_calls
    )


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


def test_client_fetch_mention_edges_filters_by_predicate_param(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    seen_params: dict[str, Any] = {}

    def fake_get(url: str, params=None, timeout=None):
        assert url.endswith("/kg/edges")
        seen_params.update(params or {})
        return _FakeResponse(200, _kg_edges_payload_mentions())

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    result = tfc.fetch_mention_edges_for_run(FAKE_BASE_URL, FAKE_RUN_ID)
    assert seen_params["predicate"] == "mentions"
    assert seen_params["run_id"] == FAKE_RUN_ID
    assert len(result) == 1
    assert result[0]["object"] == "Juniper Feld"


def test_client_fetch_mention_edges_empty_items_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    def fake_get(url: str, params=None, timeout=None):
        return _FakeResponse(200, _kg_edges_payload_empty())

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    assert tfc.fetch_mention_edges_for_run(FAKE_BASE_URL, FAKE_RUN_ID) == []


def test_client_fetch_mention_edges_malformed_response_raises_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import topic_foundry_client as tfc

    def fake_get(url: str, params=None, timeout=None):
        return _FakeResponse(200, {"run_id": FAKE_RUN_ID})  # missing "items"

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    with pytest.raises(tfc.TopicFoundryClientError):
        tfc.fetch_mention_edges_for_run(FAKE_BASE_URL, FAKE_RUN_ID)


def test_client_fetch_mention_edges_request_failure_raises_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import topic_foundry_client as tfc

    def fake_get(url: str, params=None, timeout=None):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    with pytest.raises(tfc.TopicFoundryClientError):
        tfc.fetch_mention_edges_for_run(FAKE_BASE_URL, FAKE_RUN_ID)


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


# --- segment_topic_map construction (co_occurs_with edges from real segments) ---
#
# Regression coverage for the bug found via live verification: the ingest route
# used to always pass segment_topic_map={} to map_topic_foundry_run_to_substrate(),
# so co_occurs_with edges were never produced, so _classify_typed_concept_relations
# (above) never had any real candidate pairs to classify -- despite being fully
# wired and tested against synthetic fixtures. Confirmed live: the running
# FalkorDB substrate graph had zero co_occurs_with edges despite real ingestion
# having run. Fixed by fetching GET /segments and grouping by UTC-day bucket of
# each segment's start_at (SegmentRecord has no direct session/conversation id).


def _segments_payload_same_day(run_id: str = FAKE_RUN_ID) -> dict[str, Any]:
    # Two segments for topic 0 and two for topic 1, all on 2026-07-15 (same UTC
    # day bucket) -- topics 0 and 1 co-occurring in that bucket should produce a
    # co_occurs_with edge between them. A third segment for topic -1 (outlier)
    # and a fourth with no start_at must both be excluded from the map entirely.
    return {
        "run_id": run_id,
        "items": [
            {"segment_id": "seg-0a", "topic_id": 0, "start_at": "2026-07-15T09:00:00Z"},
            {"segment_id": "seg-1a", "topic_id": 1, "start_at": "2026-07-15T14:30:00+00:00"},
            {"segment_id": "seg-outlier", "topic_id": -1, "start_at": "2026-07-15T10:00:00Z"},
            {"segment_id": "seg-no-ts", "topic_id": 0, "start_at": None},
        ],
        "limit": 1000,
        "offset": 0,
        "total": 4,
    }


def test_ingest_builds_segment_topic_map_and_produces_co_occurs_with_edges(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, calls = _make_fake_get(
        topics_payload=_topics_payload_normal(),
        segments_payload=_segments_payload_same_day(),
    )
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True

    segments_call_urls = [url for url, _params in calls if url.endswith("/segments")]
    assert len(segments_call_urls) == 1

    snapshot = store.snapshot()
    co_occurs_edges = [e for e in snapshot.edges.values() if e.predicate == "co_occurs_with"]
    assert len(co_occurs_edges) == 1
    edge = co_occurs_edges[0]
    node_ids = {edge.source.node_id, edge.target.node_id}
    assert node_ids == {
        f"sub-concept-topicfoundry-{FAKE_RUN_ID}-0",
        f"sub-concept-topicfoundry-{FAKE_RUN_ID}-1",
    }
    # edges_written in the response counts the co_occurs_with edge too (plus
    # the 2 supports edges from the normal-topics fixture).
    assert body["edges_written"] == 3
    assert body["segments_fetched"] == 4  # includes the excluded outlier + no-timestamp segments
    assert body["segment_topic_map_buckets"] == 1  # all real segments land in the same UTC day


def test_ingest_segments_fetch_failure_still_ingests_concepts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A segments-fetch failure must degrade to an empty segment_topic_map, not
    abort the route -- concept/evidence ingestion is independent of it."""
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, _calls = _make_fake_get(topics_payload=_topics_payload_normal(), segments_unreachable=True)
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["concepts_written"] == 2  # unaffected by the segments failure
    assert body["edges_written"] == 2  # supports edges only; no co_occurs_with
    assert body["segments_fetched"] == 0
    assert body["segment_topic_map_buckets"] == 0

    snapshot = store.snapshot()
    co_occurs_edges = [e for e in snapshot.edges.values() if e.predicate == "co_occurs_with"]
    assert co_occurs_edges == []


# --- mention edges (GET /kg/edges) -> EntityNodeV1 / associated_with, added 2026-07-28 ---


def test_ingest_mention_edges_produce_entity_nodes_and_associated_with_edges(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, calls = _make_fake_get(
        topics_payload=_topics_payload_normal(),
        segments_payload=_segments_payload_same_day(),
        kg_edges_payload=_kg_edges_payload_mentions(),
    )
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["mentions_fetched"] == 1
    assert body["entities_written"] == 1

    kg_edges_calls = [
        (url, params) for url, params in calls if url.endswith("/kg/edges")
    ]
    assert len(kg_edges_calls) == 1
    assert kg_edges_calls[0][1]["predicate"] == "mentions"

    snapshot = store.snapshot()
    entity_nodes = [n for n in snapshot.nodes.values() if n.node_kind == "entity"]
    assert len(entity_nodes) == 1
    assert entity_nodes[0].label == "Juniper Feld"

    associated_edges = [e for e in snapshot.edges.values() if e.predicate == "associated_with"]
    assert len(associated_edges) == 1
    assert associated_edges[0].source.node_id == f"sub-concept-topicfoundry-{FAKE_RUN_ID}-0"
    assert associated_edges[0].target.node_id == entity_nodes[0].node_id
    # "Juniper Feld" does not exact-match the "juniper" seed landmark label ->
    # no second (landmark) associated_with edge should appear here. Covered
    # explicitly (not just by the count above) by the landmark tests below.


# --- landmark connection (golden seed concepts), added 2026-08-20 ---
# docs/superpowers/specs/2026-08-20-concept-graph-landmark-connection-design.md


def _kg_edges_payload_mentions_landmark(object_text: str, run_id: str = FAKE_RUN_ID) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "items": [
            {
                "edge_id": "55555555-5555-5555-5555-555555555555",
                "segment_id": "seg-0a",
                "subject": "m",
                "predicate": "mentions",
                "object": object_text,
                "confidence": 0.7,
                "created_at": "2026-07-15T09:00:01Z",
            }
        ],
        "limit": 500,
        "offset": 0,
    }


def test_ingest_mention_exact_matching_seed_label_produces_landmark_edge(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mention whose text exact-matches a MENTION-RESOLVED seed concept's
    label (case-insensitively) gets a *second* associated_with edge straight
    to that seed node's real node_id, in addition to the normal topic-owned
    mention edge.

    Claude, not Orion, as of 2026-08-28: in `chat_history_log` Claude is a
    subject of conversation (20 rows name it, 1 row has it as the responder),
    so a mention edge is the right semantics. Orion and Juniper are speakers
    in 254/254 rows and now come from recorded segment provenance instead --
    see test_orion_is_no_longer_resolved_through_mentions below.
    """
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, _calls = _make_fake_get(
        topics_payload=_topics_payload_normal(),
        segments_payload=_segments_payload_same_day(),
        kg_edges_payload=_kg_edges_payload_mentions_landmark("claude"),  # lowercase -- must still match
    )
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["entities_written"] == 1

    snapshot = store.snapshot()
    entity_nodes = [n for n in snapshot.nodes.values() if n.node_kind == "entity"]
    assert len(entity_nodes) == 1

    associated_edges = [e for e in snapshot.edges.values() if e.predicate == "associated_with"]
    # One topic -> entity edge, one entity -> landmark edge.
    assert len(associated_edges) == 2
    landmark_edges = [e for e in associated_edges if e.target.node_id == "sub-concept-seed-claude"]
    assert len(landmark_edges) == 1
    assert landmark_edges[0].source.node_id == entity_nodes[0].node_id

    # The seed node itself was never written by this ingestion (it's not
    # part of the topic-foundry record) -- the edge points at its literal
    # node_id and would attach correctly once/if the seed fixture is loaded
    # into this same store; that's covered by
    # test_landmark_concept_ids_matches_real_seed_fixture_labels below and
    # the network-route hydration tests in test_concept_atlas_routes.py.
    assert store.get_node_by_id("sub-concept-seed-claude") is None


def test_ingest_mention_not_matching_any_seed_label_produces_no_landmark_edge(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, _calls = _make_fake_get(
        topics_payload=_topics_payload_normal(),
        segments_payload=_segments_payload_same_day(),
        kg_edges_payload=_kg_edges_payload_mentions_landmark("some random entity"),
    )
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    snapshot = store.snapshot()
    associated_edges = [e for e in snapshot.edges.values() if e.predicate == "associated_with"]
    assert len(associated_edges) == 1  # topic -> entity only, no landmark edge


def test_ingest_aitown_route_never_wires_landmark_concept_ids(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AI Town has no golden seed concepts written into its own store (see
    the design doc's non-goals) -- confirm the AI Town ingestion route never
    produces a landmark edge even for a mention that would match one of
    Orion's seed labels, since it never passes landmark_concept_ids."""
    from orion.substrate.store import InMemorySubstrateGraphStore

    aitown_store = InMemorySubstrateGraphStore()
    fake_get, _calls = _make_fake_get(
        topics_payload=_topics_payload_normal(),
        segments_payload=_segments_payload_same_day(),
        kg_edges_payload=_kg_edges_payload_mentions_landmark("orion"),
    )
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    from scripts import concept_atlas_routes

    monkeypatch.setattr(concept_atlas_routes, "_get_aitown_substrate_store", lambda: aitown_store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry-aitown")
    assert r.status_code == 200
    snapshot = aitown_store.snapshot()
    associated_edges = [e for e in snapshot.edges.values() if e.predicate == "associated_with"]
    assert len(associated_edges) == 1  # topic -> entity only, no landmark edge


def test_seed_ids_read_the_real_fixture_and_split_by_resolution_method() -> None:
    """Both id maps must read the real seed_concepts.yaml fixture (not a
    hardcoded pair) -- that is what makes adding a 5th seed 'just work'. And
    each seed must appear in exactly ONE of them: a speaker resolved from
    recorded provenance must NOT also be resolvable through entity mentions,
    or the retired path is not retired, it is a fallback (CLAUDE.md 0A).
    """
    from scripts.concept_atlas_routes import (
        _landmark_concept_ids,
        _seed_concept_ids,
        _speaker_concept_ids,
    )

    seeds = _seed_concept_ids()
    assert seeds.get("orion") == "sub-concept-seed-orion"
    assert seeds.get("juniper") == "sub-concept-seed-juniper"
    assert seeds.get("claude") == "sub-concept-seed-claude"

    speakers = _speaker_concept_ids()
    landmarks = _landmark_concept_ids()

    # Orion and Juniper: participation only.
    assert speakers.get("orion") == "sub-concept-seed-orion"
    assert speakers.get("juniper") == "sub-concept-seed-juniper"
    assert "orion" not in landmarks
    assert "juniper" not in landmarks

    # Claude: mentions only.
    assert landmarks.get("claude") == "sub-concept-seed-claude"
    assert "claude" not in speakers

    # No seed in both, none dropped entirely.
    assert not (set(speakers) & set(landmarks))
    assert set(speakers) | set(landmarks) == set(seeds)


def test_orion_is_no_longer_resolved_through_mentions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retired path must actually be gone, not merely unused.

    A mention whose text is exactly "orion" must NOT produce a landmark edge
    to the Orion seed. Orion's connection comes from recorded segment
    provenance now; leaving the mention route live for the same subject would
    mean two producers for one fact, with the weaker one (28% ceiling, 0%
    actual) silently filling in.
    """
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, _calls = _make_fake_get(
        topics_payload=_topics_payload_normal(),
        segments_payload=_segments_payload_same_day(),
        kg_edges_payload=_kg_edges_payload_mentions_landmark("orion"),
    )
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200

    snapshot = store.snapshot()
    # The entity node itself is still created (it is a real mentioned thing);
    # what must not exist is a MENTION-derived edge to the Orion seed.
    # Deliberately not "no edge to the seed at all": a participation edge to
    # that same seed is the correct replacement, and would make a broader
    # assertion fail on right behavior.
    assert [n for n in snapshot.nodes.values() if n.node_kind == "entity"]
    mention_seed_edges = [
        e
        for e in snapshot.edges.values()
        if e.target.node_id == "sub-concept-seed-orion"
        and e.provenance.source_kind == "topic_foundry.mention_landmark"
    ]
    assert mention_seed_edges == []


def test_ingest_cross_run_same_entity_label_merges_to_one_durable_entity(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the identity-resolution gap found in review 2026-07-28:
    orion/substrate/reconcile.py::canonical_node_key had no branch for
    node_kind == "entity", so EntityNodeV1's own node_id (run-scoped, a hash
    of run_id + label) meant every ingestion tick created a brand-new entity
    node for the same real-world mentioned entity -- unbounded duplication on
    a scheduler that runs daily by default. Mirrors
    test_ingest_cross_run_same_label_merges_to_one_durable_concept's shape.
    """
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    run_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    run_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    def _segments_payload(run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "items": [{"segment_id": "seg-a", "topic_id": 0, "start_at": "2026-07-15T09:00:00Z"}],
            "limit": 1000,
            "offset": 0,
            "total": 1,
        }

    def _kg_edges_payload(run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "items": [
                {
                    "edge_id": "44444444-4444-4444-4444-444444444444",
                    "segment_id": "seg-a",
                    "subject": "m",
                    "predicate": "mentions",
                    # Deliberately different case/whitespace across runs --
                    # normalization must still collapse this to one entity.
                    "object": "Juniper Feld" if run_id == run_a else "  juniper feld ",
                    "confidence": 0.6,
                    "created_at": "2026-07-15T09:00:01Z",
                }
            ],
            "limit": 500,
            "offset": 0,
        }

    def _ingest_run(run_id: str) -> dict[str, Any]:
        fake_get, _calls = _make_fake_get(
            topics_payload=_topics_payload_normal(),
            run_id=run_id,
            segments_payload=_segments_payload(run_id),
            kg_edges_payload=_kg_edges_payload(run_id),
        )
        _patch_topic_foundry_client(monkeypatch, fake_get)
        r = client.post("/api/substrate/concepts/ingest-topic-foundry")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["entities_written"] == 1
        return body

    _ingest_run(run_a)
    _ingest_run(run_b)

    snapshot = store.snapshot()
    entity_nodes = [n for n in snapshot.nodes.values() if n.node_kind == "entity"]
    assert len(entity_nodes) == 1, (
        f"expected one durable entity after cross-run ingest, got {len(entity_nodes)}: "
        f"{[n.node_id for n in entity_nodes]}"
    )

    associated_edges = [e for e in snapshot.edges.values() if e.predicate == "associated_with"]
    assert len(associated_edges) == 1, (
        f"expected one durable associated_with edge after cross-run ingest, got "
        f"{len(associated_edges)}"
    )
    assert associated_edges[0].target.node_id == entity_nodes[0].node_id


def test_ingest_mentions_fetch_failure_still_ingests_concepts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mentions-fetch failure must degrade to no entity nodes/edges, not
    abort the route -- concept/evidence/co_occurs_with ingestion above is
    independent of it (same contract as the segments-fetch failure test)."""
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, _calls = _make_fake_get(
        topics_payload=_topics_payload_normal(),
        segments_payload=_segments_payload_same_day(),
        kg_edges_unreachable=True,
    )
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["mentions_fetched"] == 0
    assert body["entities_written"] == 0
    # co_occurs_with (unaffected by the mentions failure) still produced.
    snapshot = store.snapshot()
    co_occurs_edges = [e for e in snapshot.edges.values() if e.predicate == "co_occurs_with"]
    assert len(co_occurs_edges) == 1


def test_ingest_co_occurs_with_edge_clears_classification_threshold(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove the fix actually unblocks _classify_typed_concept_relations, not
    just that a co_occurs_with edge exists: a single ingestion call with enough
    same-day segment overlap must produce an edge whose co_occurrence_count
    clears orion.substrate.relation_classification's DEFAULT_COUNT_THRESHOLD (5),
    and is_worth_classifying() must say yes for it."""
    from orion.substrate.relation_classification import is_worth_classifying
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()

    # 6 distinct day-buckets, each with one segment for topic 0 and one for
    # topic 1 -- 6 co-occurrences of the (0, 1) pair, clearing threshold=5.
    items = []
    for day in range(15, 21):  # 2026-07-15 .. 2026-07-20
        items.append({"segment_id": f"seg-0-{day}", "topic_id": 0, "start_at": f"2026-07-{day}T09:00:00Z"})
        items.append({"segment_id": f"seg-1-{day}", "topic_id": 1, "start_at": f"2026-07-{day}T14:00:00Z"})
    segments_payload = {"run_id": FAKE_RUN_ID, "items": items, "limit": 1000, "offset": 0, "total": len(items)}

    fake_get, _calls = _make_fake_get(topics_payload=_topics_payload_normal(), segments_payload=segments_payload)
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)

    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200

    snapshot = store.snapshot()
    co_occurs_edges = [e for e in snapshot.edges.values() if e.predicate == "co_occurs_with"]
    assert len(co_occurs_edges) == 1
    edge = co_occurs_edges[0]
    assert edge.metadata.get("co_occurrence_count") == 6

    concept_nodes = {n.node_id: n for n in snapshot.nodes.values() if n.node_kind == "concept"}
    node_a = concept_nodes[edge.source.node_id]
    node_b = concept_nodes[edge.target.node_id]
    assert is_worth_classifying(node_a, node_b, edge, strategy="count") is True


# --- _day_bucket_from_timestamp -----------------------------------------------


def test_day_bucket_from_timestamp_parses_z_suffixed_iso() -> None:
    from scripts.concept_atlas_routes import _day_bucket_from_timestamp

    assert _day_bucket_from_timestamp("2026-07-15T10:23:00Z") == "2026-07-15"


def test_day_bucket_from_timestamp_parses_offset_iso() -> None:
    from scripts.concept_atlas_routes import _day_bucket_from_timestamp

    assert _day_bucket_from_timestamp("2026-07-15T10:23:00+00:00") == "2026-07-15"


def test_day_bucket_from_timestamp_garbage_returns_none() -> None:
    from scripts.concept_atlas_routes import _day_bucket_from_timestamp

    assert _day_bucket_from_timestamp("not-a-timestamp") is None


def test_day_bucket_from_timestamp_none_returns_none() -> None:
    from scripts.concept_atlas_routes import _day_bucket_from_timestamp

    assert _day_bucket_from_timestamp(None) is None


# ---------------------------------------------------------------------------
# Hub-side participation wiring (branch 2). The adapter tests hand-build
# segment_speakers, so ONLY these exercise the actual read of
# seg["provenance"]["speakers"] -- the single seam the whole feature rides on.
# ---------------------------------------------------------------------------


def _segments_payload_with_speakers(run_id: str = FAKE_RUN_ID) -> dict[str, Any]:
    """Topic 0: 2 orion + 1 juniper. Topic 1: 1 juniper. Hand-counted shares
    below are 2/3, 1/3 and 1/1."""
    return {
        "run_id": run_id,
        "items": [
            {
                "segment_id": "seg-0a",
                "topic_id": 0,
                "start_at": "2026-08-28T09:00:00Z",
                "provenance": {"row_ids": ["r1"], "speakers": ["orion"]},
            },
            {
                "segment_id": "seg-0b",
                "topic_id": 0,
                "start_at": "2026-08-28T09:05:00Z",
                "provenance": {"row_ids": ["r2"], "speakers": ["Orion"]},  # case-insensitive
            },
            {
                "segment_id": "seg-0c",
                "topic_id": 0,
                "start_at": "2026-08-28T09:10:00Z",
                "provenance": {"row_ids": ["r3"], "speakers": ["juniper"]},
            },
            {
                "segment_id": "seg-1a",
                "topic_id": 1,
                "start_at": "2026-08-28T14:30:00Z",
                "provenance": {"row_ids": ["r4"], "speakers": ["juniper"]},
            },
        ],
        "limit": 1000,
        "offset": 0,
        "total": 4,
    }


def _participation_edges_in(store) -> list:
    return [
        e
        for e in store.snapshot().edges.values()
        if e.provenance.source_kind == "topic_foundry.participation"
    ]


def _ingest_with_speakers(client, monkeypatch, segments_payload):
    from orion.substrate.store import InMemorySubstrateGraphStore

    store = InMemorySubstrateGraphStore()
    fake_get, _calls = _make_fake_get(
        topics_payload=_topics_payload_normal(),
        segments_payload=segments_payload,
    )
    _patch_topic_foundry_client(monkeypatch, fake_get)
    _patch_base_url(monkeypatch, FAKE_BASE_URL)
    _patch_store(monkeypatch, store)
    r = client.post("/api/substrate/concepts/ingest-topic-foundry")
    assert r.status_code == 200
    return store, r.json()


def test_hub_reads_provenance_speakers_and_writes_participation_edges(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole feature rides on this one read. Without it segment_speakers
    stays empty, the adapter no-ops, and -- because the mention path was
    retired for these speakers in the same commit -- Orion and Juniper connect
    to nothing while every other test stays green."""
    store, body = _ingest_with_speakers(client, monkeypatch, _segments_payload_with_speakers())

    edges = _participation_edges_in(store)
    assert len(edges) == 3
    by_pair = {(e.source.node_id, e.target.node_id): e for e in edges}
    orion_t0 = by_pair[(f"sub-concept-topicfoundry-{FAKE_RUN_ID}-0", "sub-concept-seed-orion")]
    juniper_t0 = by_pair[(f"sub-concept-topicfoundry-{FAKE_RUN_ID}-0", "sub-concept-seed-juniper")]
    juniper_t1 = by_pair[(f"sub-concept-topicfoundry-{FAKE_RUN_ID}-1", "sub-concept-seed-juniper")]

    assert orion_t0.salience == pytest.approx(2 / 3)  # "Orion" matched case-insensitively
    assert juniper_t0.salience == pytest.approx(1 / 3)
    assert juniper_t1.salience == pytest.approx(1.0)
    assert orion_t0.metadata["share_is_partial"] is False

    assert body["segments_with_speakers"] == 4
    assert body["participation_edges"] == 3


def test_run_without_recorded_speakers_reports_zero_rather_than_looking_healthy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`provenance.speakers` only exists on runs trained after 2026-08-28. An
    older run produces no participation edges AND no mention-landmark edges
    for these speakers (that path is retired), so the response must make the
    zero visible instead of reporting a healthy-looking ingest."""
    store, body = _ingest_with_speakers(client, monkeypatch, _segments_payload_same_day())

    assert _participation_edges_in(store) == []
    assert body["available"] is True  # the ingest itself really did succeed
    assert body["segments_with_speakers"] == 0
    assert body["participation_edges"] == 0
    assert body["segments_fetched"] > 0  # ...on a run that did have segments


def test_participation_share_is_flagged_partial_when_the_segment_page_is_full(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """salience reads as "N% of this topic", which is only true over the
    segments actually fetched. The fetch is one un-paginated page."""
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car, "_SEGMENTS_FETCH_LIMIT", 4)  # our fixture has exactly 4
    store, _body = _ingest_with_speakers(client, monkeypatch, _segments_payload_with_speakers())

    edges = _participation_edges_in(store)
    assert edges
    for edge in edges:
        assert edge.metadata["share_is_partial"] is True


def test_speaker_set_cannot_drift_from_the_columns_that_produce_it() -> None:
    """A name in _PARTICIPATION_RESOLVED_SPEAKERS but not in
    _TOPIC_FOUNDRY_COLUMN_SPEAKERS is excluded from the mention path AND never
    appears in any segment's speakers -- zero edges from either route, silently."""
    from scripts.concept_atlas_routes import (
        _PARTICIPATION_RESOLVED_SPEAKERS,
        _TOPIC_FOUNDRY_COLUMN_SPEAKERS,
    )

    assert _PARTICIPATION_RESOLVED_SPEAKERS == frozenset(_TOPIC_FOUNDRY_COLUMN_SPEAKERS.values())
