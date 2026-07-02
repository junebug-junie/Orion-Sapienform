from __future__ import annotations

from unittest.mock import MagicMock

from app.store import PRUNE_FIELD_STATE_SQL
from app.worker import FieldDigesterWorker


def _make_worker(monkeypatch, *, retention_hours: str) -> FieldDigesterWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("FIELD_STATE_RETENTION_HOURS", retention_hours)
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = FieldDigesterWorker.__new__(FieldDigesterWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    return worker


def test_prune_tick_calls_store_with_configured_retention(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="48.0")
    worker._store.prune_field_state.return_value = 3

    worker._prune_tick()

    worker._store.prune_field_state.assert_called_once_with(retention_hours=48.0)


def test_prune_tick_disabled_when_retention_zero(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="0")

    worker._prune_tick()

    worker._store.prune_field_state.assert_not_called()


def test_prune_sql_is_batched_and_guards_latest_tick():
    # The newest row must never be deletable, so load_latest_field() can never
    # observe an empty table even if the writer is paused (idle tick flag off).
    assert "DELETE FROM substrate_field_state" in PRUNE_FIELD_STATE_SQL
    assert "LIMIT :batch_size" in PRUNE_FIELD_STATE_SQL
    assert "tick_id <>" in PRUNE_FIELD_STATE_SQL
    assert "ORDER BY generated_at DESC" in PRUNE_FIELD_STATE_SQL
