"""Body economy A5: a training run opens one `maintain` act and closes it once, on every path."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from app.services import act_emit, training


class _Seg:
    def __init__(self):
        self.doc_id = str(uuid4())
        self.text = "a segment about the cabinet getting warm"
        self.row_ids = ["r1"]
        self.timestamps = []
        self.speakers = []


@pytest.fixture
def harness(monkeypatch):
    events: list = []
    state = {"segments": [_Seg(), _Seg()], "raise_in": None}

    def run_record(run_id, payload, model_row, dataset, spec_hash):
        return SimpleNamespace(
            run_id=run_id, model_id=uuid4(), dataset_id=uuid4(), status="queued", stage=None,
            started_at=None, completed_at=None, error=None, stats=None, artifact_paths=None,
            specs=SimpleNamespace(model=SimpleNamespace(embedding_source_url="http://x"),
                                  enrichment=SimpleNamespace(enable_enrichment=False, enricher=None)),
        )

    class _Embedder:
        def __init__(self, url):
            pass

        def embed_texts(self, texts):
            return [[0.1, 0.2] for _ in texts]

    class _Clusterer:
        def fit_predict(self, x):
            return np.zeros(len(x), dtype=int)

    def write_artifacts(*a, **k):
        if state["raise_in"] == "artifacts":
            raise OSError("disk full")
        return {"topics_summary": "/tmp/x"}

    monkeypatch.setattr(training, "_build_run_record", run_record)
    monkeypatch.setattr(training, "update_run", lambda run: None)
    monkeypatch.setattr(training, "_prepare_segments", lambda run, payload: (state["segments"], 0))
    monkeypatch.setattr(training, "VectorHostEmbeddingProvider", _Embedder)
    monkeypatch.setattr(training, "_build_reducer", lambda spec, n_samples: None)
    monkeypatch.setattr(training, "_build_clusterer", lambda spec: _Clusterer())
    monkeypatch.setattr(training, "_compute_stats", lambda labels, segments: {})
    monkeypatch.setattr(training, "_write_artifacts", write_artifacts)
    monkeypatch.setattr(training, "insert_segments", lambda records: None)
    monkeypatch.setattr(training, "_publish_run_complete", lambda run, model_row: None)
    monkeypatch.setattr(act_emit, "default_publisher", lambda: events.append)
    return SimpleNamespace(events=events, state=state)


def _run(run_id=None):
    run_id = run_id or uuid4()
    training._run_training(run_id, SimpleNamespace(), {"name": "orion", "version": "v1"}, None, "h")
    return run_id


def test_complete_run_opens_and_closes_one_maintain_act(harness):
    run_id = _run()
    assert [e.event_kind for e in harness.events] == ["act_opened", "act_closed"]
    opened, closed = harness.events
    from orion.grammar.acts import stable_act_id

    assert opened.act.act_id == closed.act.act_id == stable_act_id("topic_foundry.run", "maintain", str(run_id))
    assert opened.act.actor == "topic_foundry.run" and opened.act.verb == "maintain"
    assert opened.provenance.source_service == "orion-topic-foundry"
    assert closed.act.outcome.value == 1.0 and "status=complete" in closed.act.outcome.note
    assert type(closed.act.cost).__name__ == "ActCostV1" and closed.act.cost.seconds >= 0.0
    corr = act_emit.run_correlation_id(run_id)
    assert opened.correlation_id == closed.correlation_id == corr


def test_no_documents_closes_zero(harness):
    harness.state["segments"] = []
    _run()
    assert [e.event_kind for e in harness.events] == ["act_opened", "act_closed"]
    assert harness.events[1].act.outcome.value == 0.0
    assert "No documents available" in harness.events[1].act.outcome.note


def test_exception_mid_run_closes_zero(harness):
    harness.state["raise_in"] = "artifacts"
    _run()
    assert [e.event_kind for e in harness.events] == ["act_opened", "act_closed"]
    assert harness.events[1].act.outcome.note.startswith("raised:OSError")


def test_base_exception_still_closes_once(harness, monkeypatch):
    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(training, "insert_segments", boom)
    with pytest.raises(KeyboardInterrupt):
        _run()
    assert [e.event_kind for e in harness.events] == ["act_opened", "act_closed"]
    assert harness.events[1].act.outcome.note == "exited_without_outcome"


def test_flag_off_no_act(harness, monkeypatch):
    monkeypatch.setattr(act_emit, "default_publisher", lambda: None)
    _run()
    assert harness.events == []


def test_failing_publish_never_fails_the_run(harness, monkeypatch):
    attempts = []

    def bad(event):
        attempts.append(event.event_kind)
        raise ConnectionError("bus gone")

    monkeypatch.setattr(act_emit, "default_publisher", lambda: bad)
    _run()  # no raise
    assert attempts == ["act_opened", "act_closed"]


def test_a_broken_act_build_never_strands_the_run(harness, monkeypatch):
    def broken(*a, **k):
        raise ValueError("bad act")

    monkeypatch.setattr(act_emit, "run_act", broken)
    statuses = []
    monkeypatch.setattr(training, "update_run", lambda run: statuses.append(run.status))
    _run()
    assert statuses[-1] == "complete" and harness.events == []


def test_same_run_id_same_act_and_correlation():
    rid = uuid4()
    assert act_emit.run_act(rid).act_id == act_emit.run_act(rid).act_id
    assert act_emit.run_correlation_id(rid) == act_emit.run_correlation_id(str(rid))
