"""Tests for the Self Atlas -- topic-foundry's clustering pipeline pointed
at Orion's own self_knowledge_items instead of chat (self-model rebuild
arc, Patch 3, 2026-09-05). Mirrors test_topic_foundry_scheduler.py's
conventions for the sibling AI Town dataset/model pair.
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
FAKE_DATASET_ID = "44444444-4444-4444-4444-444444444444"
FAKE_MODEL_ID = "55555555-5555-5555-5555-555555555555"
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


def test_self_constants_use_the_self_knowledge_items_table_and_columns() -> None:
    from scripts import concept_atlas_routes as car

    assert car._TOPIC_FOUNDRY_SELF_SOURCE_TABLE == "self_knowledge_items"
    assert car._TOPIC_FOUNDRY_SELF_ID_COLUMN == "item_id"
    assert car._TOPIC_FOUNDRY_SELF_TIME_COLUMN == "created_at"
    assert car._TOPIC_FOUNDRY_SELF_TEXT_COLUMNS == ["name", "symbol_name", "metadata_text"]
    # Distinct dataset/model names from both Orion and AI Town -- no
    # accidental get-or-create collision with either existing pair.
    assert car._TOPIC_FOUNDRY_SELF_DATASET_NAME not in {
        car._TOPIC_FOUNDRY_DATASET_NAME,
        car._TOPIC_FOUNDRY_AITOWN_DATASET_NAME,
    }


def test_ensure_dataset_and_model_sends_the_self_specific_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review-relevant regression: id_column/time_column/text_columns used
    to be hardcoded module globals inside _ensure_topic_foundry_dataset_and_
    model's body (the chat datasets' correlation_id/created_at/prompt+
    response shape) -- confirms the new parameters actually reach the real
    POST /datasets payload for a source table with a different shape."""
    from scripts import concept_atlas_routes as car
    from scripts import topic_foundry_client as tfc

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/datasets"):
            return _FakeResponse(200, {"datasets": []})
        if url.endswith("/models"):
            return _FakeResponse(200, {"models": []})
        raise AssertionError(f"unexpected GET {url}")

    posted_dataset_payload = {}

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/datasets"):
            posted_dataset_payload.update(json)
            return _FakeResponse(200, {"dataset_id": FAKE_DATASET_ID, "created_at": "2026-09-05T00:00:00Z"})
        if url.endswith("/models"):
            return _FakeResponse(200, {"model_id": FAKE_MODEL_ID, "created_at": "2026-09-05T00:00:00Z"})
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    monkeypatch.setattr(tfc.requests, "post", fake_post)

    result = car._ensure_topic_foundry_dataset_and_model(
        FAKE_BASE_URL,
        dataset_name=car._TOPIC_FOUNDRY_SELF_DATASET_NAME,
        model_name=car._TOPIC_FOUNDRY_SELF_MODEL_NAME,
        source_table=car._TOPIC_FOUNDRY_SELF_SOURCE_TABLE,
        where_sql=None,
        id_column=car._TOPIC_FOUNDRY_SELF_ID_COLUMN,
        time_column=car._TOPIC_FOUNDRY_SELF_TIME_COLUMN,
        text_columns=car._TOPIC_FOUNDRY_SELF_TEXT_COLUMNS,
        windowing_spec=car._TOPIC_FOUNDRY_SELF_WINDOWING_SPEC,
    )

    assert result == (FAKE_DATASET_ID, FAKE_MODEL_ID)
    assert posted_dataset_payload["source_table"] == "self_knowledge_items"
    assert posted_dataset_payload["id_column"] == "item_id"
    assert posted_dataset_payload["time_column"] == "created_at"
    assert posted_dataset_payload["text_columns"] == ["name", "symbol_name", "metadata_text"]
    assert posted_dataset_payload["where_sql"] is None


def test_ensure_dataset_and_model_default_columns_unchanged_for_existing_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirms the id_column/time_column/text_columns parameterization is
    purely additive: a zero-arg-for-those-three call (Orion's existing call
    shape) still sends the original chat-dataset column names."""
    from scripts import concept_atlas_routes as car
    from scripts import topic_foundry_client as tfc

    monkeypatch.setattr(tfc.requests, "get", lambda *a, **k: _FakeResponse(200, {"datasets": [], "models": []}))
    posted = {}

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/datasets"):
            posted.update(json)
            return _FakeResponse(200, {"dataset_id": FAKE_DATASET_ID, "created_at": "2026-09-05T00:00:00Z"})
        return _FakeResponse(200, {"model_id": FAKE_MODEL_ID, "created_at": "2026-09-05T00:00:00Z"})

    monkeypatch.setattr(tfc.requests, "post", fake_post)

    car._ensure_topic_foundry_dataset_and_model(FAKE_BASE_URL, windowing_spec=car._TOPIC_FOUNDRY_WINDOWING_SPEC)

    assert posted["id_column"] == "correlation_id"
    assert posted["time_column"] == "created_at"
    assert posted["text_columns"] == ["prompt", "response"]


def test_trigger_topic_foundry_self_training_run_uses_self_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    captured = {}

    def fake_trigger(**kwargs):
        captured.update(kwargs)
        return {"triggered": True, "run_id": FAKE_RUN_ID}

    monkeypatch.setattr(car, "trigger_topic_foundry_training_run", fake_trigger)

    result = car.trigger_topic_foundry_self_training_run()

    assert result == {"triggered": True, "run_id": FAKE_RUN_ID}
    assert captured["dataset_name"] == car._TOPIC_FOUNDRY_SELF_DATASET_NAME
    assert captured["model_name"] == car._TOPIC_FOUNDRY_SELF_MODEL_NAME
    assert captured["source_table"] == "self_knowledge_items"
    assert captured["where_sql"] is None
    assert captured["id_column"] == "item_id"
    assert captured["time_column"] == "created_at"
    assert captured["text_columns"] == ["name", "symbol_name", "metadata_text"]
    assert captured["log_prefix"] == "topic_foundry_self"


def test_trigger_topic_foundry_self_enrichment_uses_self_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    captured = {}

    def fake_enrich(**kwargs):
        captured.update(kwargs)
        return {"triggered": True}

    monkeypatch.setattr(car, "trigger_topic_foundry_enrichment", fake_enrich)

    car.trigger_topic_foundry_self_enrichment()

    assert captured["model_name"] == car._TOPIC_FOUNDRY_SELF_MODEL_NAME
    assert captured["log_prefix"] == "topic_foundry_self"


def test_resolve_store_for_graph_param_self(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    sentinel = object()
    monkeypatch.setattr(car, "_get_self_substrate_store", lambda: sentinel)

    store, label = car._resolve_store_for_graph_param("self")

    assert store is sentinel
    assert label == "self"


def test_get_self_substrate_store_resolves_the_real_attr_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    class _FakeApiRoutes:
        SUBSTRATE_SEMANTIC_STORE_SELF = object()

    monkeypatch.setitem(sys.modules, "scripts.api_routes", _FakeApiRoutes())

    store = car._get_self_substrate_store()

    assert store is _FakeApiRoutes.SUBSTRATE_SEMANTIC_STORE_SELF


def test_ingest_topic_foundry_self_route_uses_self_store_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    captured = {}

    def fake_ingest(**kwargs):
        captured.update(kwargs)
        return {"available": True}

    sentinel_store = object()
    monkeypatch.setattr(car, "_ingest_topic_foundry_run", fake_ingest)
    monkeypatch.setattr(car, "_get_self_substrate_store", lambda: sentinel_store)

    result = car.concept_atlas_ingest_topic_foundry_self()

    assert result == {"available": True}
    assert captured["store"] is sentinel_store
    assert captured["model_name"] == car._TOPIC_FOUNDRY_SELF_MODEL_NAME
    assert captured["log_prefix"] == "concept_atlas_self"
    assert "landmark_concept_ids" not in captured
    assert "speaker_concept_ids" not in captured
