from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.digest import (
    DigestSummary,
    TopicsSnapshot,
    build_digest_content,
    fetch_topics_snapshot,
    window_bounds,
)


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_fetch_topics_snapshot_maps_foundry_success(monkeypatch):
    run_id = "00000000-0000-0000-0000-000000000099"
    model_name = "digest-model"

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/runs"):
            return _FakeResponse(
                {
                    "items": [
                        {
                            "run_id": run_id,
                            "status": "complete",
                            "model": {"name": model_name, "version": "v1"},
                        }
                    ]
                }
            )
        if url.endswith("/topics"):
            assert params["run_id"] == run_id
            return _FakeResponse(
                {
                    "items": [
                        {"topic_id": 3, "label": "recall", "count": 12},
                        {"topic_id": 7, "count": 4},
                    ]
                }
            )
        if url.endswith("/drift"):
            assert params["model_name"] == model_name
            return _FakeResponse(
                {
                    "model_name": model_name,
                    "records": [
                        {
                            "drift_id": "00000000-0000-0000-0000-0000000000aa",
                            "js_divergence": 0.82,
                            "window_end": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                }
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("app.digest.requests.get", fake_get)

    snapshot = fetch_topics_snapshot(
        topic_foundry_url="http://topic-foundry:8615",
        model_name=model_name,
        window_minutes=60,
        max_topics=5,
        drift_max_records=3,
    )

    assert snapshot.summary_error is None
    assert snapshot.drift_error is None
    assert [item.label for item in snapshot.summary_items] == ["recall", "Topic 7"]
    assert snapshot.summary_items[0].value == 12.0
    assert snapshot.drift_items[0].score == pytest.approx(0.82)
    assert model_name in snapshot.drift_items[0].label


def test_parse_foundry_drift_filters_by_window_minutes():
    from app.digest import _parse_foundry_topic_drift

    now = datetime.now(timezone.utc)
    recent = (now - timedelta(minutes=30)).isoformat()
    stale = (now - timedelta(days=7)).isoformat()
    items = _parse_foundry_topic_drift(
        {
            "records": [
                {"js_divergence": 0.95, "window_end": stale},
                {"js_divergence": 0.55, "window_end": recent},
            ]
        },
        "digest-model",
        window_minutes=60,
    )
    assert len(items) == 1
    assert items[0].score == pytest.approx(0.55)


def test_fetch_topics_snapshot_unavailable_is_graceful():
    snapshot = fetch_topics_snapshot(
        topic_foundry_url=None,
        model_name=None,
        window_minutes=30,
        max_topics=5,
        drift_max_records=3,
    )
    assert snapshot.summary_items == []
    assert snapshot.drift_items == []
    assert "TOPIC_FOUNDRY_URL not set" in (snapshot.summary_error or "")
    assert "TOPIC_FOUNDRY_URL not set" in (snapshot.drift_error or "")


def test_build_digest_content_includes_topics_without_crashing():
    window_end = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    window_start, window_end = window_bounds(window_end, 1)
    summary = DigestSummary(
        window_start=window_start,
        window_end=window_end,
        severity_counts={},
        top_event_kinds=[],
        top_source_services=[],
        critical_events=[],
        warning_events=[],
        failed_attempts=[],
        throttled_count=0,
        deduped_count=0,
    )
    topics = TopicsSnapshot(
        window_minutes=60,
        summary_items=[],
        drift_items=[],
        summary_error="connection refused",
        drift_error="connection refused",
    )
    body_text, body_md = build_digest_content(summary, topics_snapshot=topics, drift_max_items=3)
    assert "Topics:" in body_text
    assert "Top topics unavailable" in body_text
    assert "## Topics" in body_md
