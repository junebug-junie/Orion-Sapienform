from __future__ import annotations

from unittest.mock import MagicMock

from app.store import PRUNE_APPLIED_DELTAS_SQL, PRUNE_FIELD_STATE_SQL
from app.worker import FieldDigesterWorker


def _make_worker(
    monkeypatch, *, retention_hours: str, applied_deltas_min_age_hours: str = "1.0"
) -> FieldDigesterWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("FIELD_STATE_RETENTION_HOURS", retention_hours)
    monkeypatch.setenv("FIELD_APPLIED_DELTAS_PRUNE_MIN_AGE_HOURS", applied_deltas_min_age_hours)
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = FieldDigesterWorker.__new__(FieldDigesterWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    return worker


def test_prune_tick_calls_store_with_configured_retention(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="48.0")
    worker._store.prune_field_state.return_value = 3
    worker._store.prune_applied_deltas.return_value = 0

    worker._prune_tick()

    worker._store.prune_field_state.assert_called_once_with(retention_hours=48.0)


def test_prune_tick_disabled_when_retention_zero(monkeypatch):
    worker = _make_worker(monkeypatch, retention_hours="0")
    worker._store.prune_applied_deltas.return_value = 0

    worker._prune_tick()

    worker._store.prune_field_state.assert_not_called()


def test_prune_tick_always_prunes_applied_deltas_even_when_field_state_retention_is_zero(
    monkeypatch,
):
    # Applied-deltas correctness is tied to receipt existence, not to field_state's
    # retention window, so it must not be gated by FIELD_STATE_RETENTION_HOURS.
    worker = _make_worker(monkeypatch, retention_hours="0", applied_deltas_min_age_hours="2.0")
    worker._store.prune_applied_deltas.return_value = 7

    worker._prune_tick()

    worker._store.prune_applied_deltas.assert_called_once_with(min_age_hours=2.0)


def test_prune_sql_is_batched_and_guards_latest_tick():
    # The newest row must never be deletable, so load_latest_field() can never
    # observe an empty table even if the writer is paused (idle tick flag off).
    assert "DELETE FROM substrate_field_state" in PRUNE_FIELD_STATE_SQL
    assert "LIMIT :batch_size" in PRUNE_FIELD_STATE_SQL
    assert "tick_id <>" in PRUNE_FIELD_STATE_SQL
    assert "ORDER BY generated_at DESC" in PRUNE_FIELD_STATE_SQL


def test_prune_applied_deltas_sql_only_deletes_rows_whose_receipt_is_gone():
    # Correctness anchor: a dedup row must never be removed while its source
    # receipt could still be replayed (e.g. via a manual cursor reset).
    assert "DELETE FROM substrate_field_applied_deltas" in PRUNE_APPLIED_DELTAS_SQL
    assert "NOT EXISTS" in PRUNE_APPLIED_DELTAS_SQL
    assert "substrate_reduction_receipts" in PRUNE_APPLIED_DELTAS_SQL
    assert "LIMIT :batch_size" in PRUNE_APPLIED_DELTAS_SQL
