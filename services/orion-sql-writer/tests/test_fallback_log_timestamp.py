"""Tests for bus_fallback_log created_at_ts population and truth windows."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))


def test_write_fallback_sets_created_at_ts(monkeypatch) -> None:
    captured: dict = {}

    class _FakeLog:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_sess = MagicMock()
    monkeypatch.setattr("app.worker.BusFallbackLog", _FakeLog)
    monkeypatch.setattr("app.worker.get_session", lambda: fake_sess)
    monkeypatch.setattr("app.worker.remove_session", lambda: None)
    monkeypatch.setattr("app.worker._json_sanitize", lambda x: x)

    from app.worker import _write_fallback

    _write_fallback("spark.telemetry", "corr-1", {"x": 1}, "duplicate key")

    assert captured["kind"] == "spark.telemetry"
    assert captured["correlation_id"] == "corr-1"
    assert isinstance(captured["created_at_ts"], datetime)
    assert captured["created_at_ts"].tzinfo is not None
    fake_sess.commit.assert_called_once()


def test_fallback_counts_use_created_at_ts_windows(monkeypatch) -> None:
    conn = MagicMock()
    conn.execute.side_effect = [
        MagicMock(scalar_one=lambda: 10),
        MagicMock(scalar_one=lambda: 1),
        MagicMock(scalar_one=lambda: 3),
        MagicMock(scalar_one=lambda: 5),
    ]
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    engine = MagicMock()
    engine.connect.return_value = conn
    monkeypatch.setattr("app.grammar_truth.grammar_engine", engine)

    from app.grammar_truth import _fallback_counts

    counts = _fallback_counts()
    assert counts == {"total": 10, "last_5m": 1, "last_30m": 3, "last_60m": 5}

    sql_texts = [str(c.args[0]) for c in conn.execute.call_args_list]
    assert all("created_at_ts" in sql for sql in sql_texts[1:])
    assert "INTERVAL '5 minutes'" in sql_texts[1]
    assert "INTERVAL '30 minutes'" in sql_texts[2]
    assert "INTERVAL '60 minutes'" in sql_texts[3]
