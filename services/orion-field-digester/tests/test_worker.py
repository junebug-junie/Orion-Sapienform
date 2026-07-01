from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.worker import FieldDigesterWorker
from app.tensor.field_state import empty_field_state, new_tick_id


def _make_worker(monkeypatch, *, idle_tick_enabled: bool = True) -> FieldDigesterWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("FIELD_DIGESTER_IDLE_TICK_ENABLED", "true" if idle_tick_enabled else "false")
    import app.settings as settings_mod
    settings_mod._settings = None  # get_settings() caches a module-level singleton; reset for isolation

    worker = FieldDigesterWorker.__new__(FieldDigesterWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._lattice = MagicMock(nodes=["n1"], capabilities=["c1"], edges=[])
    return worker


def test_idle_tick_mints_new_tick_and_does_not_touch_receipt_cursor(monkeypatch):
    worker = _make_worker(monkeypatch, idle_tick_enabled=True)
    worker._store.fetch_new_receipts.return_value = []
    existing = empty_field_state(lattice=worker._lattice, now=datetime.now(timezone.utc), tick_id="tick_old")
    worker._store.load_latest_field.return_value = existing

    worker._tick()

    worker._store.save_field.assert_called_once()
    saved_state = worker._store.save_field.call_args.args[0]
    assert saved_state.tick_id != "tick_old"
    worker._store.commit_digest_tick.assert_not_called()


def test_idle_tick_disabled_reverts_to_silent_early_return(monkeypatch):
    worker = _make_worker(monkeypatch, idle_tick_enabled=False)
    worker._store.fetch_new_receipts.return_value = []

    worker._tick()

    worker._store.load_latest_field.assert_not_called()
    worker._store.save_field.assert_not_called()
    worker._store.commit_digest_tick.assert_not_called()


def test_non_idle_tick_still_uses_commit_digest_tick(monkeypatch):
    worker = _make_worker(monkeypatch, idle_tick_enabled=True)
    fetched_item = MagicMock()
    fetched_item.receipt.receipt_id = "r1"
    fetched_item.receipt.state_deltas = []
    fetched_item.created_at = datetime.now(timezone.utc)
    worker._store.fetch_new_receipts.return_value = [fetched_item]
    existing = empty_field_state(lattice=worker._lattice, now=datetime.now(timezone.utc), tick_id="tick_old")
    worker._store.load_latest_field.return_value = existing

    worker._tick()

    worker._store.commit_digest_tick.assert_called_once()
    worker._store.save_field.assert_not_called()
