"""Tests for Self Atlas's second self_concept_history producer
(self-model rebuild arc, Patch 3 follow-up, 2026-09-05):
services/orion-hub/scripts/self_atlas_cluster_history.py.

Mirrors test_topic_foundry_self_atlas_scheduler.py's import-path/env
bootstrap and _FakeResponse HTTP-faking conventions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import requests

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

FAKE_BASE_URL = "http://fake-topic-foundry:8615"
FAKE_RUN_ID = "66666666-6666-6666-6666-666666666666"


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self) -> Any:
        return self._payload


@pytest.fixture(autouse=True)
def _clean_import_path():
    _ensure_hub_scripts_import_path()
    yield


# ---------------------------------------------------------------------------
# Pure helpers: concept_id / content / event-building
# ---------------------------------------------------------------------------


def test_concept_id_uses_slugified_label_when_present() -> None:
    from scripts import self_atlas_cluster_history as sach

    assert sach._self_atlas_concept_id(3, "GPU Thermal Behavior!") == "self-atlas-cluster-gpu-thermal-behavior"


def test_concept_id_falls_back_to_topic_id_when_label_missing() -> None:
    from scripts import self_atlas_cluster_history as sach

    assert sach._self_atlas_concept_id(7, None) == "self-atlas-cluster-topic-7"


def test_concept_id_falls_back_when_label_is_only_punctuation() -> None:
    """A label that slugifies to an empty string (e.g. all punctuation)
    must not silently produce "self-atlas-cluster-" -- falls back to the
    topic_id form same as a genuinely missing label."""
    from scripts import self_atlas_cluster_history as sach

    assert sach._self_atlas_concept_id(9, "!!!") == "self-atlas-cluster-topic-9"


def test_content_includes_label_and_keywords() -> None:
    from scripts import self_atlas_cluster_history as sach

    content = sach._self_atlas_content("GPU thermal behavior", 3, ["temperature", "throttle", "fan curve"])
    assert "GPU thermal behavior" in content
    assert "temperature" in content
    assert "throttle" in content


def test_content_falls_back_to_topic_id_label_and_notes_missing_keywords() -> None:
    from scripts import self_atlas_cluster_history as sach

    content = sach._self_atlas_content(None, 4, [])
    assert "topic_4" in content
    assert "no keywords available" in content


def test_content_caps_keyword_count() -> None:
    from scripts import self_atlas_cluster_history as sach

    many_keywords = [f"kw{i}" for i in range(30)]
    content = sach._self_atlas_content("label", 1, many_keywords)
    assert content.count("kw") == sach._MAX_KEYWORDS_IN_CONTENT


def test_build_events_skips_outlier_topic() -> None:
    from scripts import self_atlas_cluster_history as sach

    topics = [{"topic_id": -1, "label": "noise", "count": 5}, {"topic_id": 0, "label": "real topic", "count": 10}]
    events = sach._build_self_atlas_cluster_events(topics=topics, keywords_by_topic={}, segments=[])

    assert len(events) == 1
    assert events[0]["concept_id"] == "self-atlas-cluster-real-topic"


def test_build_events_collects_dedupes_sorts_and_caps_evidence_refs() -> None:
    from scripts import self_atlas_cluster_history as sach

    topics = [{"topic_id": 0, "label": "cluster a", "count": 3}]
    segments = [
        {"topic_id": 0, "provenance": {"row_ids": ["item-3", "item-1"]}},
        {"topic_id": 0, "provenance": {"row_ids": ["item-1", "item-2"]}},  # duplicate item-1
        {"topic_id": -1, "provenance": {"row_ids": ["item-noise"]}},  # outlier bucket, excluded
    ]

    events = sach._build_self_atlas_cluster_events(topics=topics, keywords_by_topic={}, segments=segments)

    assert len(events) == 1
    assert events[0]["evidence_refs"] == ["item-1", "item-2", "item-3"]


def test_build_events_caps_evidence_refs_at_max() -> None:
    from scripts import self_atlas_cluster_history as sach

    topics = [{"topic_id": 0, "label": "big cluster", "count": 1000}]
    segments = [{"topic_id": 0, "provenance": {"row_ids": [f"item-{i:04d}" for i in range(200)]}}]

    events = sach._build_self_atlas_cluster_events(topics=topics, keywords_by_topic={}, segments=segments)

    assert len(events[0]["evidence_refs"]) == sach._MAX_EVIDENCE_REFS_PER_CLUSTER


def test_build_events_returns_one_event_per_real_topic() -> None:
    from scripts import self_atlas_cluster_history as sach

    topics = [
        {"topic_id": 0, "label": "cluster a", "count": 3},
        {"topic_id": 1, "label": "cluster b", "count": 4},
    ]
    events = sach._build_self_atlas_cluster_events(
        topics=topics, keywords_by_topic={0: ["x"], 1: ["y"]}, segments=[]
    )

    assert {e["concept_id"] for e in events} == {
        "self-atlas-cluster-cluster-a",
        "self-atlas-cluster-cluster-b",
    }


# ---------------------------------------------------------------------------
# Version/dedup logic
# ---------------------------------------------------------------------------


def test_publish_writes_version_one_when_no_existing_row(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import self_atlas_cluster_history as sach

    monkeypatch.setattr(sach, "_latest_self_concept_history_row", lambda concept_id: None)
    published_events = []
    monkeypatch.setattr(
        sach, "_publish_events", lambda events, *, correlation_id: (published_events.extend(events) or (len(events), 0))
    )
    monkeypatch.setattr(
        sach,
        "_build_self_atlas_cluster_events",
        lambda **kwargs: [{"concept_id": "self-atlas-cluster-a", "content": "new content", "evidence_refs": []}],
    )

    result = _run_publish_with_fake_fetch(monkeypatch, sach)

    assert result["published_count"] == 1
    assert result["skipped_unchanged_count"] == 0
    assert published_events[0]["version"] == 1


def test_publish_skips_concept_with_unchanged_content(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import self_atlas_cluster_history as sach

    monkeypatch.setattr(sach, "_latest_self_concept_history_row", lambda concept_id: (2, "same content", []))
    published_events = []
    monkeypatch.setattr(
        sach, "_publish_events", lambda events, *, correlation_id: (published_events.extend(events) or (len(events), 0))
    )
    monkeypatch.setattr(
        sach,
        "_build_self_atlas_cluster_events",
        lambda **kwargs: [{"concept_id": "self-atlas-cluster-a", "content": "same content", "evidence_refs": []}],
    )

    result = _run_publish_with_fake_fetch(monkeypatch, sach)

    assert result["published_count"] == 0
    assert result["skipped_unchanged_count"] == 1
    assert published_events == []


def test_publish_bumps_version_when_content_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import self_atlas_cluster_history as sach

    monkeypatch.setattr(sach, "_latest_self_concept_history_row", lambda concept_id: (2, "old content", []))
    published_events = []
    monkeypatch.setattr(
        sach, "_publish_events", lambda events, *, correlation_id: (published_events.extend(events) or (len(events), 0))
    )
    monkeypatch.setattr(
        sach,
        "_build_self_atlas_cluster_events",
        lambda **kwargs: [{"concept_id": "self-atlas-cluster-a", "content": "new content", "evidence_refs": []}],
    )

    result = _run_publish_with_fake_fetch(monkeypatch, sach)

    assert result["published_count"] == 1
    assert published_events[0]["version"] == 3


def test_publish_bumps_version_when_only_evidence_refs_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review finding (2026-09-05): a cluster whose label/keywords are
    stable but has accumulated new self_knowledge_items evidence must still
    get a fresh version -- comparing content text alone would freeze this
    cluster's history forever after its first publish."""
    from scripts import self_atlas_cluster_history as sach

    monkeypatch.setattr(sach, "_latest_self_concept_history_row", lambda concept_id: (1, "same content", ["item-1"]))
    published_events = []
    monkeypatch.setattr(
        sach, "_publish_events", lambda events, *, correlation_id: (published_events.extend(events) or (len(events), 0))
    )
    monkeypatch.setattr(
        sach,
        "_build_self_atlas_cluster_events",
        lambda **kwargs: [
            {"concept_id": "self-atlas-cluster-a", "content": "same content", "evidence_refs": ["item-1", "item-2"]}
        ],
    )

    result = _run_publish_with_fake_fetch(monkeypatch, sach)

    assert result["published_count"] == 1
    assert result["skipped_unchanged_count"] == 0
    assert published_events[0]["version"] == 2


def test_build_events_disambiguates_concept_id_collision_within_a_batch() -> None:
    """Review finding (2026-09-05): two distinct clusters whose labels
    slugify to the same string must not silently collide onto one
    concept_id -- each candidate in a batch must end up with a unique id."""
    from scripts import self_atlas_cluster_history as sach

    topics = [
        {"topic_id": 0, "label": "GPU Thermal Behavior!", "count": 5},
        {"topic_id": 1, "label": "GPU Thermal Behavior?", "count": 7},
    ]

    events = sach._build_self_atlas_cluster_events(topics=topics, keywords_by_topic={}, segments=[])

    concept_ids = [e["concept_id"] for e in events]
    assert len(concept_ids) == len(set(concept_ids)), "collided concept_ids must be disambiguated"
    assert concept_ids[0] == "self-atlas-cluster-gpu-thermal-behavior"
    assert concept_ids[1] == "self-atlas-cluster-gpu-thermal-behavior-topic-1"


def test_publish_end_to_end_handles_colliding_concept_ids_without_dropping_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full orchestration path (not just the pure builder): two colliding
    candidates must both reach a distinct _latest_self_concept_history_row
    lookup and both get published, rather than one silently overwriting the
    other's version decision."""
    from scripts import self_atlas_cluster_history as sach

    monkeypatch.setattr(sach, "_latest_self_concept_history_row", lambda concept_id: None)
    published_events = []
    monkeypatch.setattr(
        sach, "_publish_events", lambda events, *, correlation_id: (published_events.extend(events) or (len(events), 0))
    )
    monkeypatch.setattr(
        sach,
        "_build_self_atlas_cluster_events",
        lambda **kwargs: [
            {"concept_id": "self-atlas-cluster-a", "content": "content A", "evidence_refs": []},
            {"concept_id": "self-atlas-cluster-a-topic-1", "content": "content B", "evidence_refs": []},
        ],
    )

    result = _run_publish_with_fake_fetch(monkeypatch, sach)

    assert result["published_count"] == 2
    assert {e["concept_id"] for e in published_events} == {"self-atlas-cluster-a", "self-atlas-cluster-a-topic-1"}


def _run_publish_with_fake_fetch(monkeypatch: pytest.MonkeyPatch, sach) -> dict:
    """Shared plumbing for the three version/dedup tests above: fake out the
    real `settings.TOPIC_FOUNDRY_BASE_URL` attribute (not the whole settings
    object -- concept_atlas_routes.py computes an HDBSCAN fingerprint off
    several other real settings attributes at import time, so a bare stand-in
    object blows up on missing attributes) and the HTTP fetch calls, so only
    the dedup logic under test actually runs, without hand-writing a full
    run/topics/segments fixture for each one (that full round trip is covered
    separately by test_publish_end_to_end_builds_and_publishes_real_shaped_envelopes)."""
    from scripts.settings import settings as real_settings

    monkeypatch.setattr(real_settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL, raising=False)

    from scripts import topic_foundry_client as tfc

    monkeypatch.setattr(
        tfc,
        "fetch_run_topics_and_keywords",
        lambda base_url, *, model_name=None, **kwargs: {
            "run_id": FAKE_RUN_ID,
            "run": {"run_id": FAKE_RUN_ID},
            "topics": [{"topic_id": 0, "label": "a", "count": 1}],
            "keywords_by_topic": {},
        },
    )
    monkeypatch.setattr(tfc, "fetch_segments_for_run", lambda base_url, run_id, **kwargs: [])

    return sach.publish_self_atlas_cluster_history()


# ---------------------------------------------------------------------------
# End-to-end: real fetch + build + version-lookup path, fake bus
# ---------------------------------------------------------------------------


def test_publish_end_to_end_builds_and_publishes_real_shaped_envelopes(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import self_atlas_cluster_history as sach
    from scripts import topic_foundry_client as tfc
    from scripts.settings import settings as real_settings

    # Patch only the specific attributes this path reads (not the whole
    # settings object -- see _run_publish_with_fake_fetch's docstring for why).
    monkeypatch.setattr(real_settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL, raising=False)
    monkeypatch.setattr(real_settings, "ORION_BUS_URL", "redis://fake:6379/0", raising=False)

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/runs"):
            return _FakeResponse(200, {"items": [{"run_id": FAKE_RUN_ID, "status": "complete"}]})
        if url.endswith("/topics"):
            return _FakeResponse(
                200,
                {
                    "items": [
                        {"topic_id": -1, "label": None, "count": 2, "outlier_pct": 1.0},
                        {"topic_id": 0, "label": "gpu thermal behavior", "count": 5, "outlier_pct": 0.0},
                    ]
                },
            )
        if url.endswith("/keywords"):
            return _FakeResponse(200, {"keywords": ["temperature", "throttle"]})
        if url.endswith("/segments"):
            return _FakeResponse(
                200,
                {
                    "items": [
                        {"topic_id": 0, "start_at": "2026-09-05T00:00:00Z", "provenance": {"row_ids": ["item-1", "item-2"]}},
                    ],
                    "total": 1,
                },
            )
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(tfc.requests, "get", fake_get)

    from scripts import self_atlas_cluster_history as sach_module

    monkeypatch.setattr(sach_module, "_latest_self_concept_history_row", lambda concept_id: None)

    published = []

    class _FakeBus:
        async def connect(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def publish(self, channel, envelope) -> None:
            published.append((channel, envelope))

    monkeypatch.setattr("orion.core.bus.async_service.OrionBusAsync", lambda *a, **k: _FakeBus())

    result = sach.publish_self_atlas_cluster_history()

    assert result["available"] is True
    assert result["run_id"] == FAKE_RUN_ID
    assert result["published_count"] == 1
    assert result["concept_ids"] == ["self-atlas-cluster-gpu-thermal-behavior"]

    assert len(published) == 1
    channel, envelope = published[0]
    assert channel == sach.CHANNEL_SELF_CONCEPT_HISTORY_WRITE
    assert envelope.payload["produced_by"] == "self_atlas_cluster"
    assert envelope.payload["concept_id"] == "self-atlas-cluster-gpu-thermal-behavior"
    assert envelope.payload["evidence_refs"] == ["item-1", "item-2"]
    assert "gpu thermal behavior" in envelope.payload["content"]
    assert "temperature" in envelope.payload["content"]


def test_publish_missing_base_url_returns_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import self_atlas_cluster_history as sach
    from scripts.settings import settings as real_settings

    monkeypatch.setattr(real_settings, "TOPIC_FOUNDRY_BASE_URL", "", raising=False)

    result = sach.publish_self_atlas_cluster_history()

    assert result["available"] is False
    assert result["reason"] == "topic_foundry_base_url_not_configured"
