"""Regression tests for repeated topic-foundry work (2026-10-30).

Bug A: POST /runs/train deduped only queued/running runs, so a finished run
did not stop an identical new one. Hub sends a train request on every
restart; spec_hash 569e0f36f1 trained six times on 2026-10-29.

Bug B: enrichment was keyed per segment_id, and every run makes new
segment_ids, so the same text went back through the LLM after every run.
Replay of 2026-10-29: 2,365 segments enriched; the chat model's runs shared
407 of 407 distinct texts with the run before them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from app.models import RunTrainRequest
from app.routers import runs as runs_module
from app.services import enrichment as enrichment_module
from app.settings import settings
from app.storage import repository as repository_module

RUN_ID = "11111111-1111-1111-1111-111111111111"
SPECS = {
    "dataset": {
        "dataset_id": "22222222-2222-2222-2222-222222222222",
        "name": "ds",
        "source_table": "chat_history_log",
        "id_column": "correlation_id",
        "time_column": "created_at",
        "text_columns": ["prompt", "response"],
        "created_at": "2026-08-29T00:00:00+00:00",
    },
    "windowing": {},
    "model": {"embedding_source_url": "http://vector-host:8320/embedding"},
}


# --- Bug A: a recent complete run blocks an identical train -----------------


class _FakeCursor:
    def __init__(self, sink):
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sink.append((sql, params))

    def fetchone(self):
        return None


class _FakeConn:
    def __init__(self, sink):
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self.sink)


def test_spec_hash_dedup_query_admits_a_recent_complete_run(monkeypatch):
    sink = []
    monkeypatch.setattr(repository_module, "pg_conn", lambda: _FakeConn(sink))
    repository_module.fetch_run_by_spec_hash("569e0f36f1", complete_within_secs=86400)
    sql, params = sink[0]
    normalized = " ".join(sql.split())
    # The old query was `status IN ('queued', 'running')` and nothing else.
    assert "status = 'complete'" in normalized
    assert "completed_at >= NOW() - make_interval(secs => %s)" in normalized
    assert params == ("569e0f36f1", 86400, 86400)


def test_spec_hash_dedup_window_zero_keeps_old_behaviour(monkeypatch):
    sink = []
    monkeypatch.setattr(repository_module, "pg_conn", lambda: _FakeConn(sink))
    repository_module.fetch_run_by_spec_hash("abc")
    assert sink[0][1] == ("abc", 0, 0)


def _stub_train(monkeypatch, existing):
    calls = {"lookup": [], "created": []}
    model_row = {
        "windowing_spec": {},
        "model_spec": {"embedding_source_url": "http://vector-host:8320/embedding"},
        "enrichment_spec": None,
    }
    monkeypatch.setattr(runs_module, "fetch_model", lambda model_id: model_row)
    monkeypatch.setattr(runs_module, "fetch_dataset", lambda dataset_id: SPECS["dataset"] | {"dataset_id": str(dataset_id)})

    def _lookup(spec_hash, *, complete_within_secs=0):
        calls["lookup"].append(complete_within_secs)
        return existing

    monkeypatch.setattr(runs_module, "fetch_run_by_spec_hash", _lookup)
    monkeypatch.setattr(runs_module, "create_run", lambda run: calls["created"].append(run))
    monkeypatch.setattr(runs_module, "enqueue_training", lambda *a, **k: None)
    return calls


def _train_request(**kw):
    return RunTrainRequest(
        model_id=uuid4(),
        dataset_id=UUID(SPECS["dataset"]["dataset_id"]),
        start_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
        end_at=datetime(2026, 10, 29, tzinfo=timezone.utc),
        **kw,
    )


def test_train_returns_recent_complete_run_instead_of_training(monkeypatch):
    monkeypatch.setattr(settings, "topic_foundry_train_dedup_complete_window_secs", 86400)
    calls = _stub_train(monkeypatch, {"run_id": RUN_ID, "status": "complete"})
    resp = runs_module.train_run_endpoint(_train_request(), background_tasks=None)
    assert str(resp.run_id) == RUN_ID
    assert resp.status == "complete"
    assert calls["created"] == []
    assert calls["lookup"] == [86400]


def test_train_force_only_dedups_in_flight_runs(monkeypatch):
    monkeypatch.setattr(settings, "topic_foundry_train_dedup_complete_window_secs", 86400)
    calls = _stub_train(monkeypatch, None)
    resp = runs_module.train_run_endpoint(_train_request(force=True), background_tasks=None)
    assert resp.status == "queued"
    assert len(calls["created"]) == 1
    assert calls["lookup"] == [0]


# --- Bug B: enrichment reused across runs by text hash ----------------------


def _run_row(run_id):
    return {
        "run_id": run_id,
        "model_id": str(uuid4()),
        "dataset_id": str(uuid4()),
        "specs": SPECS,
        "spec_hash": "abc",
        "status": "complete",
        "stage": "trained",
        "stats": {},
        "artifact_paths": {},
        "created_at": datetime(2026, 10, 29, tzinfo=timezone.utc),
        "started_at": None,
        "completed_at": None,
        "error": None,
    }


class _Store:
    """In-memory stand-in for topic_foundry_segments + the enrichment cache."""

    def __init__(self):
        self.cache = {}
        self.segments = {}  # run_id -> list of segment dicts
        self.texts = {}  # run_id -> {segment_id: text}
        self.runs = {}
        self.llm_calls = []
        self.last_stats = {}

    def add_run(self, texts):
        run_id = str(uuid4())
        segs = [{"segment_id": str(uuid4()), "enriched_at": None} for _ in texts]
        self.segments[run_id] = segs
        self.texts[run_id] = {s["segment_id"]: t for s, t in zip(segs, texts)}
        self.runs[run_id] = _run_row(run_id)
        return run_id


@pytest.fixture
def store(monkeypatch):
    st = _Store()
    monkeypatch.setattr(settings, "topic_foundry_llm_enable", True)
    monkeypatch.setattr(enrichment_module, "fetch_run", lambda run_id: st.runs[str(run_id)])
    monkeypatch.setattr(
        enrichment_module, "update_run", lambda record: st.last_stats.__setitem__(str(record.run_id), dict(record.stats))
    )
    monkeypatch.setattr(enrichment_module, "load_taxonomy", lambda spec: ["memory", "identity"])
    monkeypatch.setattr(enrichment_module, "_load_segment_text_map", lambda row: st.texts[row["run_id"]])
    monkeypatch.setattr(enrichment_module, "_write_enrichment_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(enrichment_module, "_publish_enrich_complete", lambda *a, **k: None)
    monkeypatch.setattr(enrichment_module, "_generate_edges", lambda row: None)
    monkeypatch.setattr(
        enrichment_module, "fetch_segments", lambda run_id, has_enrichment=None: st.segments[str(run_id)]
    )

    def _update(segment_id, *, enrichment, enrichment_version):
        enrichment["enriched_at"] = "now"  # same mutation the real write does
        for segs in st.segments.values():
            for seg in segs:
                if seg["segment_id"] == str(segment_id):
                    seg["enriched_at"] = "now"
                    seg["enrichment"] = enrichment

    monkeypatch.setattr(enrichment_module, "update_segment_enrichment", _update)
    monkeypatch.setattr(
        enrichment_module, "fetch_enrichment_cache", lambda keys: {k: st.cache[k] for k in keys if k in st.cache}
    )

    def _insert(key, *, enrichment, enricher):
        st.cache[key] = enrichment  # real SQL: ON CONFLICT DO UPDATE (overwrite)

    monkeypatch.setattr(enrichment_module, "insert_enrichment_cache", _insert)

    def _llm(text, taxonomy):
        st.llm_calls.append(text)
        return {"title": f"t:{text[:10]}", "aspects": ["memory"], "meaning": {}, "sentiment": {}}

    monkeypatch.setattr(enrichment_module, "_llm_enrich", _llm)
    return st


def _enrich(run_id, *, limit=None, force=False, enricher="llm"):
    enrichment_module._run_enrichment(UUID(run_id), force=force, enricher=enricher, limit=limit)


def test_second_run_with_same_texts_makes_no_llm_calls(store):
    texts = ["alpha segment text", "beta segment text", "gamma segment text"]
    first = store.add_run(texts)
    _enrich(first)
    assert len(store.llm_calls) == 3

    second = store.add_run(texts)  # new run_id, new segment_ids, same text
    _enrich(second)
    assert len(store.llm_calls) == 3, "unchanged text must not be re-enriched"
    assert all(seg["enriched_at"] for seg in store.segments[second])
    assert store.segments[second][0]["enrichment"]["title"] == "t:alpha segm"
    assert store.last_stats[second]["enrichment_cache_hits"] == 3


def test_only_changed_text_goes_to_the_llm(store):
    _enrich(store.add_run(["one", "two"]))
    _enrich(store.add_run(["one", "two", "three"]))
    assert store.llm_calls == ["one", "two", "three"]


def test_duplicate_texts_within_one_run_call_the_llm_once(store):
    # Self Atlas: 14,553 segments, only 1,984 distinct texts.
    _enrich(store.add_run(["same"] * 5))
    assert store.llm_calls == ["same"]


def test_limit_caps_fresh_calls_not_cache_hits(store):
    _enrich(store.add_run(["a", "b"]))
    run = store.add_run(["a", "b", "c", "d", "e"])
    _enrich(run, limit=2)
    assert store.llm_calls == ["a", "b", "c", "d"]
    enriched = [s for s in store.segments[run] if s["enriched_at"]]
    assert len(enriched) == 4  # 2 cache hits + 2 fresh; "e" waits for the next pass


def test_heuristic_fallback_is_not_cached(store, monkeypatch):
    monkeypatch.setattr(enrichment_module, "_llm_enrich", lambda text, tax: store.llm_calls.append(text))
    _enrich(store.add_run(["flaky"]))
    assert store.cache == {}


def test_force_bypasses_the_cache_and_overwrites_it(store, monkeypatch):
    texts = ["x", "y"]
    _enrich(store.add_run(texts))

    def _llm_v2(text, taxonomy):
        store.llm_calls.append(text)
        return {"title": f"v2:{text}", "aspects": ["memory"], "meaning": {}, "sentiment": {}}

    monkeypatch.setattr(enrichment_module, "_llm_enrich", _llm_v2)
    _enrich(store.add_run(texts), force=True)
    assert store.llm_calls == ["x", "y", "x", "y"]

    third = store.add_run(texts)
    _enrich(third)  # unforced: must get the forced (new) answer from the cache
    assert store.llm_calls == ["x", "y", "x", "y"]
    assert store.segments[third][0]["enrichment"]["title"] == "v2:x"


def test_empty_llm_answer_is_not_cached(store, monkeypatch):
    def _empty(text, taxonomy):
        store.llm_calls.append(text)
        return {}

    monkeypatch.setattr(enrichment_module, "_llm_enrich", _empty)
    _enrich(store.add_run(["blank"]))
    assert store.cache == {}


def test_duplicate_texts_get_separate_dicts(store):
    run = store.add_run(["dup"] * 3)
    _enrich(run)
    enrichments = [seg["enrichment"] for seg in store.segments[run]]
    assert len({id(e) for e in enrichments}) == 3
    assert all("enriched_at" not in value for value in store.cache.values())


def test_cached_enrichment_does_not_carry_a_stale_enriched_at(store):
    _enrich(store.add_run(["text"]))
    assert all("enriched_at" not in value for value in store.cache.values())


def test_cache_key_tracks_prompt_inputs(monkeypatch):
    monkeypatch.setattr(settings, "topic_foundry_llm_enable", True)
    key = enrichment_module.enrichment_cache_key
    assert key("t", ["a"], "llm") == key("t", ["a"], "llm")
    assert key("t", ["a"], "llm") != key("u", ["a"], "llm")
    assert key("t", ["a"], "llm") != key("t", ["b"], "llm")
    assert key("t", ["a"], "heuristic") is None
    assert key("", ["a"], "llm") is None
    monkeypatch.setattr(settings, "topic_foundry_llm_bus_route", "other-route")
    other = key("t", ["a"], "llm")
    monkeypatch.setattr(settings, "topic_foundry_llm_bus_route", None)
    assert other != key("t", ["a"], "llm")
