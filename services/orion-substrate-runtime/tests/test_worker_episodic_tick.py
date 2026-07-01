"""Unit tests for the episodic consolidation tick (self-modeling loop rung 4).

Verifies the worker's `_episodic_tick` is a no-op when disabled, consolidates
the last completed clock-aligned window into an idempotent proposal-marked
episode, and fails open on store errors.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker
from orion.schemas.reduction_receipt import ReductionReceiptV1


def _make_worker(monkeypatch, *, episodic_tick_enabled: bool = True) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv(
        "SUBSTRATE_EPISODIC_TICK_ENABLED", "true" if episodic_tick_enabled else "false"
    )
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    return worker


def _receipt(receipt_id: str, created_at: datetime) -> ReductionReceiptV1:
    return ReductionReceiptV1(receipt_id=receipt_id, created_at=created_at)


def test_episodic_tick_disabled_is_noop(monkeypatch):
    worker = _make_worker(monkeypatch, episodic_tick_enabled=False)
    worker._episodic_tick()
    worker._store.fetch_receipts_between.assert_not_called()


def test_episodic_tick_saves_proposal_marked_episode(monkeypatch):
    worker = _make_worker(monkeypatch, episodic_tick_enabled=True)
    window = int(worker._settings.episodic_window_seconds)
    now = datetime.now(timezone.utc)
    end = datetime.fromtimestamp((int(now.timestamp()) // window) * window, tz=timezone.utc)
    worker._store.fetch_receipts_between.return_value = [
        _receipt("receipt:a", end - timedelta(seconds=30)),
        _receipt("receipt:b", end - timedelta(seconds=10)),
    ]
    worker._store.save_episode_summary.return_value = True
    worker._store.prune_episode_summaries.return_value = 0

    worker._episodic_tick()

    fetch_kwargs = worker._store.fetch_receipts_between.call_args.kwargs
    assert fetch_kwargs["end"] == end
    assert fetch_kwargs["start"] == end - timedelta(seconds=window)

    episode = worker._store.save_episode_summary.call_args.args[0]
    assert episode.status == "proposal"
    assert episode.receipt_refs == ["receipt:a", "receipt:b"]
    assert episode.window_end == end


def test_episodic_tick_is_idempotent_within_a_window(monkeypatch):
    worker = _make_worker(monkeypatch, episodic_tick_enabled=True)
    window = int(worker._settings.episodic_window_seconds)
    now = datetime.now(timezone.utc)
    end = datetime.fromtimestamp((int(now.timestamp()) // window) * window, tz=timezone.utc)
    worker._store.fetch_receipts_between.return_value = [
        _receipt("receipt:a", end - timedelta(seconds=30)),
    ]
    worker._store.save_episode_summary.return_value = True
    worker._store.prune_episode_summaries.return_value = 0

    worker._episodic_tick()
    worker._episodic_tick()

    first = worker._store.save_episode_summary.call_args_list[0].args[0]
    second = worker._store.save_episode_summary.call_args_list[1].args[0]
    assert first.episode_id == second.episode_id


def test_episodic_tick_skips_save_on_empty_window(monkeypatch):
    worker = _make_worker(monkeypatch, episodic_tick_enabled=True)
    worker._store.fetch_receipts_between.return_value = []
    worker._episodic_tick()
    worker._store.save_episode_summary.assert_not_called()


def test_episodic_tick_fails_open_on_store_error(monkeypatch):
    worker = _make_worker(monkeypatch, episodic_tick_enabled=True)
    worker._store.fetch_receipts_between.side_effect = RuntimeError("db down")
    worker._episodic_tick()  # must not raise
