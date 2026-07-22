from __future__ import annotations

import json
from uuid import uuid4


def test_load_topic_labels_reads_from_topics_summary_artifact(monkeypatch, tmp_path):
    """Live incident 2026-07-21: list_topics_endpoint hardcoded label=None
    unconditionally -- topic_foundry_segments (what list_topics() queries)
    has no topic-level label column at all, and real labels only ever land
    in the topics_summary.json artifact file. This locks in the fix: labels
    get read from that file, mirroring topic_keywords_endpoint's existing
    artifact_paths pattern."""
    from app.routers import topics as topics_router

    run_id = uuid4()
    summary_path = tmp_path / "topics_summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {"topic_id": -1, "count": 5, "outlier_pct": 1.0, "label": None},
                {"topic_id": 0, "count": 10, "outlier_pct": 0.0, "label": "Cat Persona"},
                {"topic_id": 1, "count": 8, "outlier_pct": 0.0, "label": "Hardware Setup"},
            ]
        )
    )

    monkeypatch.setattr(
        topics_router,
        "fetch_run",
        lambda rid: {"artifact_paths": {"topics_summary": str(summary_path)}},
    )

    labels = topics_router._load_topic_labels(run_id)
    assert labels == {"0": "Cat Persona", "1": "Hardware Setup"}
    assert "-1" not in labels  # null/outlier labels excluded, not "null" strings


def test_load_topic_labels_returns_empty_when_run_missing(monkeypatch):
    from app.routers import topics as topics_router

    monkeypatch.setattr(topics_router, "fetch_run", lambda rid: None)
    assert topics_router._load_topic_labels(uuid4()) == {}


def test_load_topic_labels_returns_empty_when_artifact_missing(monkeypatch):
    from app.routers import topics as topics_router

    monkeypatch.setattr(
        topics_router, "fetch_run", lambda rid: {"artifact_paths": {"topics_summary": "/nonexistent/path.json"}}
    )
    assert topics_router._load_topic_labels(uuid4()) == {}


def test_list_topics_endpoint_applies_real_labels(monkeypatch, tmp_path):
    """End-to-end: the actual endpoint function must thread real labels
    into TopicSummaryItem.label, not leave every item's label hardcoded
    None regardless of what _load_topic_labels found."""
    from app.routers import topics as topics_router

    summary_path = tmp_path / "topics_summary.json"
    summary_path.write_text(json.dumps([{"topic_id": 0, "count": 10, "outlier_pct": 0.0, "label": "Cat Persona"}]))

    monkeypatch.setattr(
        topics_router,
        "fetch_run",
        lambda rid: {"artifact_paths": {"topics_summary": str(summary_path)}},
    )
    monkeypatch.setattr(
        topics_router,
        "list_topics",
        lambda run_id, *, limit, offset: ([{"topic_id": 0, "count": 10, "outliers": 0}], 1),
    )

    page = topics_router.list_topics_endpoint(run_id=uuid4(), limit=200, offset=0)
    assert len(page.items) == 1
    assert page.items[0].label == "Cat Persona"
