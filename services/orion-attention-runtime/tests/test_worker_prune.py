from __future__ import annotations

from unittest.mock import MagicMock

from app.store import PRUNE_ATTENTION_FRAMES_SQL
from app.worker import AttentionRuntimeWorker


def _make_worker(monkeypatch, *, retention_hours: str) -> AttentionRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ATTENTION_FRAME_RETENTION_HOURS", retention_hours)
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = AttentionRuntimeWorker.__new__(AttentionRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    return worker


def test_prune_tick_calls_store_with_configured_retention(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="48.0")
    worker._store.prune_attention_frames.return_value = 3

    worker._prune_tick()

    worker._store.prune_attention_frames.assert_called_once_with(retention_hours=48.0)


def test_prune_tick_disabled_when_retention_zero(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="0")

    worker._prune_tick()

    worker._store.prune_attention_frames.assert_not_called()


def test_prune_sql_is_batched_and_guards_latest_frame():
    assert "DELETE FROM substrate_attention_frames" in PRUNE_ATTENTION_FRAMES_SQL
    assert "LIMIT :batch_size" in PRUNE_ATTENTION_FRAMES_SQL
    assert "frame_id <>" in PRUNE_ATTENTION_FRAMES_SQL
    assert "ORDER BY generated_at DESC" in PRUNE_ATTENTION_FRAMES_SQL
