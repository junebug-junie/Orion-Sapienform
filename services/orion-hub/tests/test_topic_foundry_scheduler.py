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

    result = car._ensure_topic_foundry_dataset_and_model(
        FAKE_BASE_URL, windowing_spec=car._TOPIC_FOUNDRY_WINDOWING_SPEC
    )
    assert result == (FAKE_DATASET_ID, FAKE_MODEL_ID)
    assert create_calls == []


def test_ensure_dataset_and_model_warns_on_where_sql_drift(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Code review 2026-08-18: get-or-create matches purely by name, so a
    where_sql edited under an already-used name would silently keep training
    on the stale filter forever. Not fixable without an update endpoint
    topic-foundry doesn't have -- but it must at least be loud (logged), not
    silent, so drift shows up on the very next scheduler tick.
    """
    from scripts import concept_atlas_routes as car
    from scripts import topic_foundry_client as tfc

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/datasets"):
            return _FakeResponse(
                200,
                {
                    "datasets": [
                        {
                            "dataset_id": FAKE_DATASET_ID,
                            "name": car._TOPIC_FOUNDRY_DATASET_NAME,
                            "where_sql": "some stale filter that no longer matches",
                            "source_table": car._TOPIC_FOUNDRY_SOURCE_TABLE,
                        }
                    ]
                },
            )
        if url.endswith("/models"):
            return _FakeResponse(
                200, {"models": [{"model_id": FAKE_MODEL_ID, "name": car._TOPIC_FOUNDRY_MODEL_NAME}]}
            )
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    caplog.set_level("WARNING")

    result = car._ensure_topic_foundry_dataset_and_model(
        FAKE_BASE_URL, windowing_spec=car._TOPIC_FOUNDRY_WINDOWING_SPEC
    )

    assert result == (FAKE_DATASET_ID, FAKE_MODEL_ID)
    assert "topic_foundry_dataset_where_sql_drift" in caplog.text
    assert "topic_foundry_dataset_source_table_drift" not in caplog.text


def test_ensure_dataset_and_model_warns_on_source_table_drift(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Code review 2026-08-20: this function's source_table parameterization
    (added for AI Town's own dataset) introduced a second drifting field
    with no analogous drift check -- an edit to source_table under an
    already-used dataset name would silently keep training against the OLD
    table forever, same failure shape as the where_sql case above."""
    from scripts import concept_atlas_routes as car
    from scripts import topic_foundry_client as tfc

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/datasets"):
            return _FakeResponse(
                200,
                {
                    "datasets": [
                        {
                            "dataset_id": FAKE_DATASET_ID,
                            "name": car._TOPIC_FOUNDRY_AITOWN_DATASET_NAME,
                            "where_sql": None,
                            "source_table": "some_stale_table_no_longer_correct",
                        }
                    ]
                },
            )
        if url.endswith("/models"):
            return _FakeResponse(
                200, {"models": [{"model_id": FAKE_MODEL_ID, "name": car._TOPIC_FOUNDRY_AITOWN_MODEL_NAME}]}
            )
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    caplog.set_level("WARNING")

    result = car._ensure_topic_foundry_dataset_and_model(
        FAKE_BASE_URL,
        dataset_name=car._TOPIC_FOUNDRY_AITOWN_DATASET_NAME,
        model_name=car._TOPIC_FOUNDRY_AITOWN_MODEL_NAME,
        source_table=car._TOPIC_FOUNDRY_AITOWN_SOURCE_TABLE,
        where_sql=None,
        windowing_spec=car._TOPIC_FOUNDRY_AITOWN_WINDOWING_SPEC,
    )

    assert result == (FAKE_DATASET_ID, FAKE_MODEL_ID)
    assert "topic_foundry_dataset_source_table_drift" in caplog.text
    assert "topic_foundry_dataset_where_sql_drift" not in caplog.text


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
            # min_cluster_size=15/metric="euclidean" (this file's hardcoded
            # values until 2026-08-19) is the exact combination flagged by
            # topic-foundry's own 2026-07-21 incident note as producing
            # degenerate clusters, and produced 0 clusters on the real
            # AI-Town-filtered corpus. Must come from settings, not a literal,
            # so it can be retuned via env without a code change.
            model_spec = json["model_spec"]
            assert model_spec["min_cluster_size"] == car.settings.SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_MIN_CLUSTER_SIZE
            assert model_spec["metric"] == car.settings.SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_METRIC
            # Pin the actual default values too -- the settings-consistency
            # assertion above would pass even if both regressed together.
            # metric is "euclidean", NOT "cosine" -- confirmed live 2026-08-19
            # that the installed hdbscan library's real clusterer rejects
            # "cosine" outright (ValueError("Unrecognized metric 'cosine'")),
            # despite ModelSpec's own field default disagreeing.
            assert model_spec["min_cluster_size"] == 8
            assert model_spec["metric"] == "euclidean"
            return _FakeResponse(200, {"model_id": FAKE_MODEL_ID, "created_at": "2026-07-17T00:00:00Z"})
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    monkeypatch.setattr(tfc.requests, "post", fake_post)

    result = car._ensure_topic_foundry_dataset_and_model(
        FAKE_BASE_URL, windowing_spec=car._TOPIC_FOUNDRY_WINDOWING_SPEC
    )
    assert result == (FAKE_DATASET_ID, FAKE_MODEL_ID)
    assert set(create_calls) == {f"{FAKE_BASE_URL}/datasets", f"{FAKE_BASE_URL}/models"}


def test_ensure_dataset_and_model_degrades_to_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car
    from scripts import topic_foundry_client as tfc

    def fake_get(url, params=None, timeout=None):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    assert car._ensure_topic_foundry_dataset_and_model(
        FAKE_BASE_URL, windowing_spec=car._TOPIC_FOUNDRY_WINDOWING_SPEC
    ) is None


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
    monkeypatch.setattr(car, "_ensure_topic_foundry_dataset_and_model", lambda base_url, **kwargs: None)
    result = car.trigger_topic_foundry_training_run()
    assert result == {"triggered": False, "reason": "dataset_or_model_resolution_failed"}


def test_trigger_topic_foundry_training_run_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_WINDOW_DAYS", 30)
    monkeypatch.setattr(
        car,
        "_ensure_topic_foundry_dataset_and_model",
        lambda base_url, **kwargs: (FAKE_DATASET_ID, FAKE_MODEL_ID),
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
        car,
        "_ensure_topic_foundry_dataset_and_model",
        lambda base_url, **kwargs: (FAKE_DATASET_ID, FAKE_MODEL_ID),
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
        car,
        "_ensure_topic_foundry_dataset_and_model",
        lambda base_url, **kwargs: (FAKE_DATASET_ID, FAKE_MODEL_ID),
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

    def fake_fetch_latest_completed_run(base_url, model_name=None, timeout=None):
        raise TopicFoundryClientError("no completed run")

    monkeypatch.setattr(car, "fetch_latest_completed_run", fake_fetch_latest_completed_run)

    result = car.trigger_topic_foundry_enrichment()
    assert result == {"triggered": False, "reason": "no_completed_run"}


def test_trigger_topic_foundry_enrichment_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE", True)
    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_LIMIT", 150)

    seen_fetch_kwargs = {}

    def fake_fetch_latest_completed_run(base_url, model_name=None, timeout=None):
        seen_fetch_kwargs["model_name"] = model_name
        return {"run_id": FAKE_RUN_ID}

    monkeypatch.setattr(car, "fetch_latest_completed_run", fake_fetch_latest_completed_run)

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
    # Code review 2026-08-18: fetch_latest_completed_run must be scoped to
    # this scheduler's own model, or it can silently resolve to a
    # *different* model's latest run (e.g. the old, unfiltered one).
    assert seen_fetch_kwargs["model_name"] == car._TOPIC_FOUNDRY_MODEL_NAME


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
        car, "fetch_latest_completed_run", lambda base_url, model_name=None, timeout=None: {"run_id": FAKE_RUN_ID}
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
        car, "fetch_latest_completed_run", lambda base_url, model_name=None, timeout=None: {"run_id": FAKE_RUN_ID}
    )

    def fake_trigger_enrichment_for_run(base_url, run_id, *, limit=None, force=False, timeout=None):
        raise TopicFoundryClientError("boom")

    monkeypatch.setattr(car, "trigger_enrichment_for_run", fake_trigger_enrichment_for_run)

    result = car.trigger_topic_foundry_enrichment()
    assert result["triggered"] is False
    assert result["reason"] == "enrich_trigger_failed"


def test_topic_foundry_hdbscan_metric_validator_rejects_unrecognized_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live incident 2026-08-19: metric="cosine" (topic-foundry's own
    ModelSpec field default) creates a model successfully and only fails
    deep inside a background training task -- ValueError("Unrecognized
    metric 'cosine'") from the installed hdbscan library. The validator
    must catch this (and any other typo/unsupported value) at Settings
    construction time, not leave it to be discovered live again.
    """
    from app.settings import Settings

    monkeypatch.setenv("SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_METRIC", "cosine")
    with pytest.raises(Exception, match="cosine"):
        Settings()


def test_topic_foundry_hdbscan_metric_validator_accepts_euclidean(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.settings import Settings

    monkeypatch.setenv("SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_METRIC", "euclidean")
    settings_instance = Settings()
    assert settings_instance.SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_METRIC == "euclidean"


# --- AI Town's own concept graph (2026-08-20) --------------------------------


def test_ensure_dataset_and_model_creates_aitown_dataset_with_no_where_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AI Town's dataset reads aitown_chat_history_log directly, which is
    already AI-Town-only by construction (table-split routing, not a
    filter) -- unlike Orion's dataset, it needs no where_sql at all."""
    from scripts import concept_atlas_routes as car
    from scripts import topic_foundry_client as tfc

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/datasets"):
            return _FakeResponse(200, {"datasets": []})
        if url.endswith("/models"):
            return _FakeResponse(200, {"models": []})
        raise AssertionError(f"unexpected GET {url}")

    posted = {}

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/datasets"):
            posted["dataset"] = json
            return _FakeResponse(200, {"dataset_id": FAKE_DATASET_ID, "created_at": "2026-08-20T00:00:00Z"})
        if url.endswith("/models"):
            posted["model"] = json
            return _FakeResponse(200, {"model_id": FAKE_MODEL_ID, "created_at": "2026-08-20T00:00:00Z"})
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    monkeypatch.setattr(tfc.requests, "post", fake_post)

    result = car._ensure_topic_foundry_dataset_and_model(
        FAKE_BASE_URL,
        dataset_name=car._TOPIC_FOUNDRY_AITOWN_DATASET_NAME,
        model_name=car._TOPIC_FOUNDRY_AITOWN_MODEL_NAME,
        source_table=car._TOPIC_FOUNDRY_AITOWN_SOURCE_TABLE,
        where_sql=None,
        windowing_spec=car._TOPIC_FOUNDRY_AITOWN_WINDOWING_SPEC,
    )

    assert result == (FAKE_DATASET_ID, FAKE_MODEL_ID)
    assert posted["dataset"]["name"] == car._TOPIC_FOUNDRY_AITOWN_DATASET_NAME
    assert posted["dataset"]["source_table"] == "aitown_chat_history_log"
    assert posted["dataset"]["where_sql"] is None
    assert posted["model"]["name"] == car._TOPIC_FOUNDRY_AITOWN_MODEL_NAME
    # Distinct names from Orion's own dataset/model -- the whole point of a
    # second dataset is that it never collides with or overwrites Orion's.
    assert car._TOPIC_FOUNDRY_AITOWN_DATASET_NAME != car._TOPIC_FOUNDRY_DATASET_NAME
    assert car._TOPIC_FOUNDRY_AITOWN_MODEL_NAME != car._TOPIC_FOUNDRY_MODEL_NAME


def test_trigger_topic_foundry_aitown_training_run_uses_aitown_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_WINDOW_DAYS", 30)

    seen_kwargs = {}

    def fake_ensure(base_url, **kwargs):
        seen_kwargs.update(kwargs)
        return (FAKE_DATASET_ID, FAKE_MODEL_ID)

    monkeypatch.setattr(car, "_ensure_topic_foundry_dataset_and_model", fake_ensure)
    monkeypatch.setattr(
        car,
        "trigger_training_run",
        lambda base_url, **kwargs: {"run_id": FAKE_RUN_ID, "status": "queued"},
    )

    result = car.trigger_topic_foundry_aitown_training_run()

    assert result["triggered"] is True
    assert seen_kwargs["dataset_name"] == car._TOPIC_FOUNDRY_AITOWN_DATASET_NAME
    assert seen_kwargs["model_name"] == car._TOPIC_FOUNDRY_AITOWN_MODEL_NAME
    assert seen_kwargs["source_table"] == "aitown_chat_history_log"
    assert seen_kwargs["where_sql"] is None


def test_trigger_topic_foundry_aitown_enrichment_uses_aitown_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE", True)
    monkeypatch.setattr(car.settings, "TOPIC_FOUNDRY_BASE_URL", FAKE_BASE_URL)
    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_ENRICH_LIMIT", 150)

    seen = {}

    def fake_fetch_latest_completed_run(base_url, model_name=None, timeout=None):
        seen["model_name"] = model_name
        return {"run_id": FAKE_RUN_ID}

    monkeypatch.setattr(car, "fetch_latest_completed_run", fake_fetch_latest_completed_run)
    monkeypatch.setattr(
        car,
        "trigger_enrichment_for_run",
        lambda base_url, run_id, **kwargs: {"run_id": run_id, "status": "running"},
    )

    result = car.trigger_topic_foundry_aitown_enrichment()

    assert result["triggered"] is True
    assert seen["model_name"] == car._TOPIC_FOUNDRY_AITOWN_MODEL_NAME
    assert seen["model_name"] != car._TOPIC_FOUNDRY_MODEL_NAME


def test_topic_foundry_model_spec_fingerprint_changes_with_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model name is suffixed with a fingerprint of the settings that
    feed model_spec (min_cluster_size/metric/embedding_source_url) instead
    of a hand-bumped "-v3"/"-v4" version suffix -- code review on this
    patch flagged that a hand-bumped suffix silently reproduces the exact
    bug class it exists to fix (forget to bump it, get-or-create keeps
    training on the OLD model_spec forever, with no drift warning possible
    on the model side unlike the dataset's where_sql case). This test
    pins: same settings -> same fingerprint (deterministic, required for
    get-or-create idempotency); different settings -> different fingerprint
    (required for the failure mode to actually be closed).
    """
    from scripts import concept_atlas_routes as car

    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_MIN_CLUSTER_SIZE", 8)
    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_METRIC", "euclidean")
    fingerprint_a = car._topic_foundry_model_spec_fingerprint()
    fingerprint_a_again = car._topic_foundry_model_spec_fingerprint()
    assert fingerprint_a == fingerprint_a_again

    monkeypatch.setattr(car.settings, "SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_MIN_CLUSTER_SIZE", 12)
    fingerprint_b = car._topic_foundry_model_spec_fingerprint()
    assert fingerprint_b != fingerprint_a


# ---------------------------------------------------------------------------
# Windowing spec + model-name fingerprint (2026-08-28 concept-induction rebuild)
# docs/superpowers/specs/2026-08-28-concept-induction-topic-model-rebuild-design.md
# ---------------------------------------------------------------------------


def test_created_model_carries_split_windowing_and_real_speakers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model row freezes its windowing_spec at creation, so what is sent
    here is what every future run trains on. Before 2026-08-28 this payload
    hardcoded ``block_mode="turn_pairs"`` against a source table whose every
    row already holds a full prompt+response exchange -- so it paired two
    whole exchanges and labelled one "User:" and the other "Assistant:".
    """
    from scripts import concept_atlas_routes as car
    from scripts import topic_foundry_client as tfc

    seen: dict[str, Any] = {}

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/datasets"):
            return _FakeResponse(200, {"datasets": []})
        if url.endswith("/models"):
            return _FakeResponse(200, {"models": []})
        raise AssertionError(f"unexpected GET {url}")

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/datasets"):
            return _FakeResponse(200, {"dataset_id": FAKE_DATASET_ID, "created_at": "2026-08-28T00:00:00Z"})
        if url.endswith("/models"):
            seen["windowing_spec"] = json["windowing_spec"]
            return _FakeResponse(200, {"model_id": FAKE_MODEL_ID, "created_at": "2026-08-28T00:00:00Z"})
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    monkeypatch.setattr(tfc.requests, "post", fake_post)

    assert car._ensure_topic_foundry_dataset_and_model(
        FAKE_BASE_URL, windowing_spec=car._TOPIC_FOUNDRY_WINDOWING_SPEC
    ) == (
        FAKE_DATASET_ID,
        FAKE_MODEL_ID,
    )

    ws = seen["windowing_spec"]
    assert ws["block_mode"] == "rows"
    assert ws["block_mode"] != "turn_pairs"
    assert ws["split_text_columns"] is True
    # `prompt` is always Juniper and `response` is always Orion on
    # chat_history_log -- recorded fact, not inference.
    assert ws["column_speakers"] == {"prompt": "juniper", "response": "orion"}
    # Derived from column_speakers, never the old ["user", "assistant"]
    # literal, which would match neither speaker and drop every block under
    # turn_pairs.
    assert ws["include_roles"] == ["juniper", "orion"]


def test_aitown_model_records_no_speakers_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import concept_atlas_routes as car
    from scripts import topic_foundry_client as tfc

    seen: dict[str, Any] = {}

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/datasets"):
            return _FakeResponse(200, {"datasets": []})
        if url.endswith("/models"):
            return _FakeResponse(200, {"models": []})
        raise AssertionError(f"unexpected GET {url}")

    def fake_post(url, json=None, timeout=None):
        if url.endswith("/datasets"):
            return _FakeResponse(200, {"dataset_id": FAKE_DATASET_ID, "created_at": "2026-08-28T00:00:00Z"})
        if url.endswith("/models"):
            seen["windowing_spec"] = json["windowing_spec"]
            return _FakeResponse(200, {"model_id": FAKE_MODEL_ID, "created_at": "2026-08-28T00:00:00Z"})
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr(tfc.requests, "get", fake_get)
    monkeypatch.setattr(tfc.requests, "post", fake_post)

    car._ensure_topic_foundry_dataset_and_model(
        FAKE_BASE_URL,
        dataset_name=car._TOPIC_FOUNDRY_AITOWN_DATASET_NAME,
        model_name=car._TOPIC_FOUNDRY_AITOWN_MODEL_NAME,
        source_table=car._TOPIC_FOUNDRY_AITOWN_SOURCE_TABLE,
        where_sql=None,
        windowing_spec=car._TOPIC_FOUNDRY_AITOWN_WINDOWING_SPEC,
    )

    ws = seen["windowing_spec"]
    # AI Town's prompt/response authors are agents, not Juniper and Orion.
    # Splitting is still right (two different speakers); naming them is not.
    assert ws["split_text_columns"] is True
    assert ws["column_speakers"] == {}
    # Empty is falsy, so the role filter short-circuits and drops nothing --
    # an unknown speaker must never silently delete the corpus.
    assert ws["include_roles"] == []


def test_model_name_fingerprint_changes_when_windowing_changes() -> None:
    """The structural drift closure. A model row freezes windowing_spec at
    creation and get-or-create matches purely by name, so if the fingerprint
    ignored windowing, editing it here would keep training on the OLD
    windowing forever with nothing to diff against (GET /models returns
    ModelSummary, which omits both specs). Confirmed live 2026-08-28: the
    model then in service still carried block_mode=turn_pairs in its row.
    """
    from scripts import concept_atlas_routes as car

    base = car._topic_foundry_windowing_spec({"prompt": "juniper", "response": "orion"})
    changed = dict(base, block_mode="turn_pairs")

    assert car._topic_foundry_model_spec_fingerprint(base) != car._topic_foundry_model_spec_fingerprint(
        changed
    )
    # Key order must not matter -- the fingerprint has to be stable across
    # equivalent dicts or every Hub restart could mint a new model.
    reordered = {k: base[k] for k in sorted(base, reverse=True)}
    assert car._topic_foundry_model_spec_fingerprint(reordered) == car._topic_foundry_model_spec_fingerprint(
        base
    )


def test_orion_and_aitown_models_have_distinct_names() -> None:
    """They now carry different windowing (column_speakers differs), so they
    must not collide on one model row."""
    from scripts import concept_atlas_routes as car

    assert car._TOPIC_FOUNDRY_MODEL_NAME != car._TOPIC_FOUNDRY_AITOWN_MODEL_NAME
    assert car._TOPIC_FOUNDRY_WINDOWING_SPEC != car._TOPIC_FOUNDRY_AITOWN_WINDOWING_SPEC
    # The names also differ by their base constant, so comparing full names
    # would pass even if the fingerprint ignored windowing entirely (review
    # finding, 2026-08-28: the original assertion was vacuous w.r.t. its own
    # docstring). Compare the fingerprint SUFFIXES.
    orion_suffix = car._TOPIC_FOUNDRY_MODEL_NAME.rsplit("-", 1)[-1]
    aitown_suffix = car._TOPIC_FOUNDRY_AITOWN_MODEL_NAME.rsplit("-", 1)[-1]
    assert orion_suffix != aitown_suffix
    assert orion_suffix == car._topic_foundry_model_spec_fingerprint(
        car._TOPIC_FOUNDRY_WINDOWING_SPEC
    )
