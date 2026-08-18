"""Tests for the autonomous topic-foundry training + ingestion scheduler
(Gap 5 of the concept-graph-pipeline design).

Covers the new client functions in
``services/orion-hub/scripts/topic_foundry_client.py``
(``list_datasets``/``list_models``/``create_dataset``/``create_model``/
``trigger_training_run``) and the scheduler entry points in
``services/orion-hub/scripts/concept_atlas_routes.py``
(``_ensure_topic_foundry_dataset_and_model``/
``trigger_topic_foundry_training_run``). All HTTP calls are mocked at the
``requests.get``/``requests.post`` boundary inside ``scripts.topic_foundry_client``
-- no real topic-foundry service, no network.

The actual `main.py` scheduler loop (the ``asyncio.create_task`` wiring) is
intentionally not unit-tested here, mirroring the established convention for
the sibling decay scheduler (PR #1131): the loop itself is a thin
sleep-then-call wrapper with no independently testable logic; what matters
is that the two functions it calls (tested here) behave correctly.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

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
FAKE_DATASET_ID = "33333333-3333-3333-3333-333333333333"
FAKE_MODEL_ID = "22222222-2222-2222-2222-222222222222"
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


@pytest.fixture(autouse=True)
def _clean_import_path():
    _ensure_hub_scripts_import_path()
    yield


# --- topic_foundry_client.py: list/create dataset/model, trigger run -------


def test_list_datasets_returns_items(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    def fake_get(url, params=None, timeout=None):
        assert url == f"{FAKE_BASE_URL}/datasets"
        return _FakeResponse(200, {"datasets": [{"dataset_id": FAKE_DATASET_ID, "name": "d"}]})

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    result = tfc.list_datasets(FAKE_BASE_URL)
    assert result == [{"dataset_id": FAKE_DATASET_ID, "name": "d"}]


def test_list_datasets_malformed_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    monkeypatch.setattr(tfc.requests, "get", lambda *a, **k: _FakeResponse(200, {"not_datasets": []}))
    with pytest.raises(tfc.TopicFoundryClientError):
        tfc.list_datasets(FAKE_BASE_URL)


def test_list_models_returns_items(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    def fake_get(url, params=None, timeout=None):
        assert url == f"{FAKE_BASE_URL}/models"
        return _FakeResponse(200, {"models": [{"model_id": FAKE_MODEL_ID, "name": "m"}]})

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    result = tfc.list_models(FAKE_BASE_URL)
    assert result == [{"model_id": FAKE_MODEL_ID, "name": "m"}]


def test_fetch_segments_for_run_returns_items(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse(
            200,
            {
                "run_id": FAKE_RUN_ID,
                "items": [{"segment_id": "s1", "topic_id": 0, "start_at": "2026-07-15T10:00:00Z"}],
                "limit": 1000,
                "offset": 0,
                "total": 1,
            },
        )

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    result = tfc.fetch_segments_for_run(FAKE_BASE_URL, FAKE_RUN_ID)
    assert captured["url"] == f"{FAKE_BASE_URL}/segments"
    assert captured["params"]["run_id"] == FAKE_RUN_ID
    assert captured["params"]["format"] == "wrapped"
    assert captured["params"]["include_bounds"] is True
    assert result == [{"segment_id": "s1", "topic_id": 0, "start_at": "2026-07-15T10:00:00Z"}]


def test_fetch_segments_for_run_malformed_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    monkeypatch.setattr(tfc.requests, "get", lambda *a, **k: _FakeResponse(200, {"not_items": []}))
    with pytest.raises(tfc.TopicFoundryClientError):
        tfc.fetch_segments_for_run(FAKE_BASE_URL, FAKE_RUN_ID)


def test_fetch_segments_for_run_logs_warning_when_total_exceeds_fetched(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression for silent truncation at the 1000-segment fetch ceiling: when
    the response's own total exceeds what was actually fetched, this must be
    logged (not silently dropped) even though no pagination loop backfills it."""
    from scripts import topic_foundry_client as tfc

    monkeypatch.setattr(
        tfc.requests,
        "get",
        lambda *a, **k: _FakeResponse(
            200, {"run_id": FAKE_RUN_ID, "items": [{"segment_id": "s1"}], "limit": 1, "offset": 0, "total": 5}
        ),
    )
    with caplog.at_level("WARNING", logger="orion-hub.topic_foundry_client"):
        result = tfc.fetch_segments_for_run(FAKE_BASE_URL, FAKE_RUN_ID)
    assert len(result) == 1
    assert any("topic_foundry_segments_truncated" in rec.message for rec in caplog.records)


def test_create_dataset_posts_payload_and_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    posted = {}

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        return _FakeResponse(200, {"dataset_id": FAKE_DATASET_ID, "created_at": "2026-07-17T00:00:00Z"})

    monkeypatch.setattr(tfc.requests, "post", fake_post)
    result = tfc.create_dataset(FAKE_BASE_URL, {"name": "d"})
    assert posted["url"] == f"{FAKE_BASE_URL}/datasets"
    assert posted["json"] == {"name": "d"}
    assert result["dataset_id"] == FAKE_DATASET_ID


def test_create_model_posts_payload_and_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    def fake_post(url, json=None, timeout=None):
        assert url == f"{FAKE_BASE_URL}/models"
        return _FakeResponse(200, {"model_id": FAKE_MODEL_ID, "created_at": "2026-07-17T00:00:00Z"})

    monkeypatch.setattr(tfc.requests, "post", fake_post)
    result = tfc.create_model(FAKE_BASE_URL, {"name": "m"})
    assert result["model_id"] == FAKE_MODEL_ID


def test_trigger_training_run_posts_run_train_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    posted = {}

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        return _FakeResponse(200, {"run_id": FAKE_RUN_ID, "status": "queued"})

    monkeypatch.setattr(tfc.requests, "post", fake_post)
    result = tfc.trigger_training_run(
        FAKE_BASE_URL,
        model_id=FAKE_MODEL_ID,
        dataset_id=FAKE_DATASET_ID,
        start_at="2026-06-17T00:00:00+00:00",
        end_at="2026-07-17T00:00:00+00:00",
    )
    assert posted["url"] == f"{FAKE_BASE_URL}/runs/train"
    assert posted["json"] == {
        "model_id": FAKE_MODEL_ID,
        "dataset_id": FAKE_DATASET_ID,
        "start_at": "2026-06-17T00:00:00+00:00",
        "end_at": "2026-07-17T00:00:00+00:00",
    }
    assert result == {"run_id": FAKE_RUN_ID, "status": "queued"}


def test_trigger_training_run_connection_error_raises_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    def fake_post(url, json=None, timeout=None):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(tfc.requests, "post", fake_post)
    with pytest.raises(tfc.TopicFoundryClientError):
        tfc.trigger_training_run(
            FAKE_BASE_URL, model_id=FAKE_MODEL_ID, dataset_id=FAKE_DATASET_ID, start_at="x", end_at="y"
        )


# --- concept_atlas_routes.py: dataset/model ensure + trigger entry point ---


def test_ensure_dataset_and_model_finds_existing_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car
    from scripts import topic_foundry_client as tfc

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/datasets"):
            return _FakeResponse(
                200, {"datasets": [{"dataset_id": FAKE_DATASET_ID, "name": car._TOPIC_FOUNDRY_DATASET_NAME}]}
            )
        if url.endswith("/models"):
            return _FakeResponse(
                200, {"models": [{"model_id": FAKE_MODEL_ID, "name": car._TOPIC_FOUNDRY_MODEL_NAME}]}
            )
        raise AssertionError(f"unexpected GET {url}")

    create_calls = []

    def fake_post(url, json=None, timeout=None):
        create_calls.append(url)
        raise AssertionError("should not create when an existing dataset/model is found")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    monkeypatch.setattr(tfc.requests, "post", fake_post)

    result = car._ensure_topic_foundry_dataset_and_model(FAKE_BASE_URL)
    assert result == (FAKE_DATASET_ID, FAKE_MODEL_ID)
    assert create_calls == []


def test_ensure_dataset_and_model_creates_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car
    from scripts import topic_foundry_client as tfc

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/datasets"):
            return _FakeResponse(200, {"datasets": []})
        if url.endswith("/models"):
            return _FakeResponse(200, {"models": []})
        raise AssertionError(f"unexpected GET {url}")

    create_calls = []

    def fake_post(url, json=None, timeout=None):
        create_calls.append(url)
        if url.endswith("/datasets"):
            assert json["name"] == car._TOPIC_FOUNDRY_DATASET_NAME
            # AI Town rows (client_meta.external_room.platform == "aitown")
            # must be excluded from Orion's own concept-graph dataset -- see
            # docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md.
            assert json["where_sql"] == car._TOPIC_FOUNDRY_WHERE_SQL
            assert "aitown" in json["where_sql"]
            return _FakeResponse(200, {"dataset_id": FAKE_DATASET_ID, "created_at": "2026-07-17T00:00:00Z"})
        if url.endswith("/models"):
            assert json["name"] == car._TOPIC_FOUNDRY_MODEL_NAME
            assert json["dataset_id"] == FAKE_DATASET_ID
            return _FakeResponse(200, {"model_id": FAKE_MODEL_ID, "created_at": "2026-07-17T00:00:00Z"})
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    monkeypatch.setattr(tfc.requests, "post", fake_post)

    result = car._ensure_topic_foundry_dataset_and_model(FAKE_BASE_URL)
    assert result == (FAKE_DATASET_ID, FAKE_MODEL_ID)
    assert set(create_calls) == {f"{FAKE_BASE_URL}/datasets", f"{FAKE_BASE_URL}/models"}


def test_ensure_dataset_and_model_degrades_to_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car
    from scripts import topic_foundry_client as tfc

    def fake_get(url, params=None, timeout=None):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    assert car._ensure_topic_foundry_dataset_and_model(FAKE_BASE_URL) is None


def test_trigger_topic_foundry_training_run_no_base_url_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", "")
    result = car.trigger_topic_foundry_training_run()
    assert result == {"triggered": False, "reason": "topic_foundry_base_url_not_configured"}


def test_trigger_topic_foundry_training_run_dataset_model_resolution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(car, "_ensure_topic_foundry_dataset_and_model", lambda base_url: None)
    result = car.trigger_topic_foundry_training_run()
    assert result == {"triggered": False, "reason": "dataset_or_model_resolution_failed"}


def test_trigger_topic_foundry_training_run_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_WINDOW_DAYS", 30)
    monkeypatch.setattr(
        car, "_ensure_topic_foundry_dataset_and_model", lambda base_url: (FAKE_DATASET_ID, FAKE_MODEL_ID)
    )

    trigger_calls = []

    def fake_trigger_training_run(base_url, *, model_id, dataset_id, start_at, end_at, timeout=None):
        trigger_calls.append((base_url, model_id, dataset_id, start_at, end_at))
        return {"run_id": FAKE_RUN_ID, "status": "queued"}

    monkeypatch.setattr(car, "trigger_training_run", fake_trigger_training_run)

    result = car.trigger_topic_foundry_training_run()
    assert result["triggered"] is True
    assert result["run_id"] == FAKE_RUN_ID
    assert result["status"] == "queued"
    assert result["dataset_id"] == FAKE_DATASET_ID
    assert result["model_id"] == FAKE_MODEL_ID
    assert result["window_days"] == 30
    assert len(trigger_calls) == 1
    assert trigger_calls[0][0] == FAKE_BASE_URL
    assert trigger_calls[0][1] == FAKE_MODEL_ID
    assert trigger_calls[0][2] == FAKE_DATASET_ID


def test_trigger_topic_foundry_training_run_windows_are_day_floored_and_repeatable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the spec_hash-dedup-never-fires bug caught in review:
    start_at/end_at must be floored to a UTC day boundary, NOT
    datetime.now(timezone.utc) verbatim -- otherwise every tick computes a
    microsecond-unique window, topic-foundry's spec_hash dedup (keyed on the
    exact start_at/end_at it receives) never matches a prior run, and every
    single tick trains a brand-new HDBSCAN model regardless of interval.
    Two calls within the same UTC day must produce byte-identical
    start_at/end_at strings."""
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_WINDOW_DAYS", 30)
    monkeypatch.setattr(
        car, "_ensure_topic_foundry_dataset_and_model", lambda base_url: (FAKE_DATASET_ID, FAKE_MODEL_ID)
    )

    windows_seen = []

    def fake_trigger_training_run(base_url, *, model_id, dataset_id, start_at, end_at, timeout=None):
        windows_seen.append((start_at, end_at))
        return {"run_id": FAKE_RUN_ID, "status": "queued"}

    monkeypatch.setattr(car, "trigger_training_run", fake_trigger_training_run)

    car.trigger_topic_foundry_training_run()
    car.trigger_topic_foundry_training_run()

    assert len(windows_seen) == 2
    assert windows_seen[0] == windows_seen[1]

    start_at_str, end_at_str = windows_seen[0]
    end_at = datetime.fromisoformat(end_at_str)
    start_at = datetime.fromisoformat(start_at_str)
    assert end_at.hour == 0 and end_at.minute == 0 and end_at.second == 0 and end_at.microsecond == 0
    assert start_at == end_at - timedelta(days=30)


def test_trigger_topic_foundry_training_run_client_error_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car
    from scripts.topic_foundry_client import TopicFoundryClientError

    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(
        car, "_ensure_topic_foundry_dataset_and_model", lambda base_url: (FAKE_DATASET_ID, FAKE_MODEL_ID)
    )

    def fake_trigger_training_run(base_url, *, model_id, dataset_id, start_at, end_at, timeout=None):
        raise TopicFoundryClientError("boom")

    monkeypatch.setattr(car, "trigger_training_run", fake_trigger_training_run)

    result = car.trigger_topic_foundry_training_run()
    assert result["triggered"] is False
    assert result["reason"] == "train_trigger_failed"


# --- topic_foundry_client.py: trigger_enrichment_for_run, added 2026-07-28 --


def test_trigger_enrichment_for_run_posts_run_enrich_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    posted = {}

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        return _FakeResponse(
            200, {"run_id": FAKE_RUN_ID, "status": "running", "enriched_count": 0, "failed_count": 0}
        )

    monkeypatch.setattr(tfc.requests, "post", fake_post)
    result = tfc.trigger_enrichment_for_run(FAKE_BASE_URL, FAKE_RUN_ID, limit=200, force=False)
    assert posted["url"] == f"{FAKE_BASE_URL}/runs/{FAKE_RUN_ID}/enrich"
    assert posted["json"] == {"force": False, "limit": 200}
    assert result["status"] == "running"


def test_trigger_enrichment_for_run_omits_limit_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    posted = {}

    def fake_post(url, json=None, timeout=None):
        posted["json"] = json
        return _FakeResponse(200, {"run_id": FAKE_RUN_ID, "status": "running"})

    monkeypatch.setattr(tfc.requests, "post", fake_post)
    tfc.trigger_enrichment_for_run(FAKE_BASE_URL, FAKE_RUN_ID, limit=None)
    assert posted["json"] == {"force": False}


def test_trigger_enrichment_for_run_connection_error_raises_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import topic_foundry_client as tfc

    def fake_post(url, json=None, timeout=None):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(tfc.requests, "post", fake_post)
    with pytest.raises(tfc.TopicFoundryClientError):
        tfc.trigger_enrichment_for_run(FAKE_BASE_URL, FAKE_RUN_ID)


# --- concept_atlas_routes.py: trigger_topic_foundry_enrichment, added 2026-07-28 --


def test_trigger_topic_foundry_enrichment_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE", False)
    result = car.trigger_topic_foundry_enrichment()
    assert result == {"triggered": False, "reason": "enrich_disabled"}


def test_trigger_topic_foundry_enrichment_no_base_url_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE", True)
    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", "")
    result = car.trigger_topic_foundry_enrichment()
    assert result == {"triggered": False, "reason": "topic_foundry_base_url_not_configured"}


def test_trigger_topic_foundry_enrichment_no_completed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car
    from scripts.topic_foundry_client import TopicFoundryClientError

    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE", True)
    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)

    def fake_fetch_latest_completed_run(base_url, timeout=None):
        raise TopicFoundryClientError("no completed run")

    monkeypatch.setattr(car, "fetch_latest_completed_run", fake_fetch_latest_completed_run)

    result = car.trigger_topic_foundry_enrichment()
    assert result == {"triggered": False, "reason": "no_completed_run"}


def test_trigger_topic_foundry_enrichment_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE", True)
    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_LIMIT", 150)

    monkeypatch.setattr(
        car, "fetch_latest_completed_run", lambda base_url, timeout=None: {"run_id": FAKE_RUN_ID}
    )

    seen_kwargs = {}

    def fake_trigger_enrichment_for_run(base_url, run_id, *, limit=None, force=False, timeout=None):
        seen_kwargs["base_url"] = base_url
        seen_kwargs["run_id"] = run_id
        seen_kwargs["limit"] = limit
        seen_kwargs["force"] = force
        return {"run_id": run_id, "status": "running", "enriched_count": 0, "failed_count": 0}

    monkeypatch.setattr(car, "trigger_enrichment_for_run", fake_trigger_enrichment_for_run)

    result = car.trigger_topic_foundry_enrichment()
    assert result["triggered"] is True
    assert result["run_id"] == FAKE_RUN_ID
    assert seen_kwargs == {
        "base_url": FAKE_BASE_URL,
        "run_id": FAKE_RUN_ID,
        "limit": 150,
        "force": False,
    }


@pytest.mark.parametrize("configured_limit", [0, -5])
def test_trigger_topic_foundry_enrichment_non_positive_limit_does_not_call_endpoint(
    monkeypatch: pytest.MonkeyPatch, configured_limit: int
) -> None:
    """Review-caught 2026-07-28: the naive `int(x or 0) or None` computation
    turned a configured limit of 0 into "no cap sent" (unlimited) --
    backwards from what an operator setting it to 0 almost certainly means.
    A configured limit of 0 or negative must skip calling the enrichment
    endpoint at all, not silently fall through to unlimited.
    """
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE", True)
    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_LIMIT", configured_limit)
    monkeypatch.setattr(
        car, "fetch_latest_completed_run", lambda base_url, timeout=None: {"run_id": FAKE_RUN_ID}
    )

    called = []

    def fake_trigger_enrichment_for_run(base_url, run_id, *, limit=None, force=False, timeout=None):
        called.append((run_id, limit))
        return {"run_id": run_id, "status": "running", "enriched_count": 0, "failed_count": 0}

    monkeypatch.setattr(car, "trigger_enrichment_for_run", fake_trigger_enrichment_for_run)

    result = car.trigger_topic_foundry_enrichment()
    assert result["triggered"] is False
    assert result["reason"] == "enrich_limit_non_positive"
    assert called == [], "a non-positive configured limit must never reach the endpoint"


def test_trigger_topic_foundry_enrichment_client_error_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car
    from scripts.topic_foundry_client import TopicFoundryClientError

    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE", True)
    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(
        car, "fetch_latest_completed_run", lambda base_url, timeout=None: {"run_id": FAKE_RUN_ID}
    )

    def fake_trigger_enrichment_for_run(base_url, run_id, *, limit=None, force=False, timeout=None):
        raise TopicFoundryClientError("boom")

    monkeypatch.setattr(car, "trigger_enrichment_for_run", fake_trigger_enrichment_for_run)

    result = car.trigger_topic_foundry_enrichment()
    assert result["triggered"] is False
    assert result["reason"] == "enrich_trigger_failed"
