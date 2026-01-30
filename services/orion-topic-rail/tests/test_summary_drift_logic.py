from datetime import datetime, timedelta, timezone

from app.topic_rail.analytics import compute_summary_rows, compute_drift_rows


def test_compute_summary_rows_respects_min_docs_and_pct():
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=60)
    window_end = now
    counts = [(1, 10), (2, 3)]

    def labeler(topic_id, keywords):
        return f"topic-{topic_id}"

    def keywords_fn(topic_id):
        return [f"kw-{topic_id}"]

    rows = compute_summary_rows(
        model_version="v1",
        window_start=window_start,
        window_end=window_end,
        counts=counts,
        total_docs=13,
        outlier_count=1,
        outlier_pct=1 / 13,
        topic_labeler=labeler,
        topic_keywords_fn=keywords_fn,
        max_topics=10,
        min_docs=5,
    )

    assert len(rows) == 1
    assert rows[0]["topic_id"] == 1
    assert rows[0]["pct_of_window"] == 10 / 13


def test_compute_drift_rows_basic_stats():
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=60)
    window_end = now
    rows = [
        {"session_id": "s1", "topic_id": 1, "created_at": now - timedelta(minutes=3)},
        {"session_id": "s1", "topic_id": 1, "created_at": now - timedelta(minutes=2)},
        {"session_id": "s1", "topic_id": 2, "created_at": now - timedelta(minutes=1)},
    ]

    drift = compute_drift_rows(
        model_version="v1",
        window_start=window_start,
        window_end=window_end,
        rows=rows,
        min_turns=2,
    )

    assert len(drift) == 1
    item = drift[0]
    assert item["turns"] == 3
    assert item["unique_topics"] == 2
    assert item["dominant_topic_id"] == 1
    assert item["dominant_pct"] == 2 / 3
    assert item["switch_rate"] == 1 / 3
