from __future__ import annotations

from unittest.mock import MagicMock

from app.store import PRUNE_HISTORY_SQL
from app.worker import SelfStateRuntimeWorker


def _make_worker(monkeypatch, *, retention_hours: str) -> SelfStateRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("SELF_STATE_RETENTION_HOURS", retention_hours)
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = SelfStateRuntimeWorker.__new__(SelfStateRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    return worker


def test_prune_tick_calls_store_with_configured_retention(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="48.0")
    worker._store.prune_history.return_value = 5

    worker._prune_tick()

    worker._store.prune_history.assert_called_once_with(retention_hours=48.0)


def test_prune_tick_disabled_when_retention_zero(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="0")

    worker._prune_tick()

    worker._store.prune_history.assert_not_called()


def test_prune_sql_covers_all_three_tables_with_guards():
    assert set(PRUNE_HISTORY_SQL) == {
        "substrate_self_state",
        "self_state_predictions",
        "identity_snapshots",
    }
    guards = {
        "substrate_self_state": "self_state_id <>",
        "self_state_predictions": "prediction_id <>",
        "identity_snapshots": "snapshot_id <>",
    }
    for table, sql in PRUNE_HISTORY_SQL.items():
        assert f"DELETE FROM {table}" in sql
        assert "LIMIT :batch_size" in sql
        assert guards[table] in sql
        assert "ORDER BY generated_at DESC" in sql
