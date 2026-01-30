from app.topic_rail.db import writer as writer_module


def test_upsert_idempotency_sql_contains_on_conflict(monkeypatch):
    captured = {"query": None}

    def fake_execute_values(cur, query, rows):
        captured["query"] = query
        captured["rows"] = rows

    class DummyCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    class DummyConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def cursor(self):
            return DummyCursor()

        def close(self):
            return None

    monkeypatch.setattr(writer_module, "execute_values", fake_execute_values)
    monkeypatch.setattr(writer_module.psycopg2, "connect", lambda dsn: DummyConn())

    writer = writer_module.TopicRailWriter("dsn")
    assignments = [
        {
            "chat_id": "chat-1",
            "correlation_id": "corr-1",
            "trace_id": "corr-1",
            "session_id": "session-1",
            "topic_id": 1,
            "topic_label": "label",
            "topic_keywords": ["a"],
            "topic_confidence": 0.9,
            "model_version": "v1",
        }
    ]

    writer.upsert_assignments(assignments)
    assert "ON CONFLICT (chat_id, model_version)" in captured["query"]
