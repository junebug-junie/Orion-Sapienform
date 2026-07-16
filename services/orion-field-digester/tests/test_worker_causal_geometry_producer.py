from __future__ import annotations

from unittest.mock import MagicMock

import app.worker as worker_module
from app.worker import FieldDigesterWorker


def _make_worker(monkeypatch, *, producer_enabled: str = "false") -> FieldDigesterWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("FIELD_PLASTICITY_PRODUCER_ENABLED", producer_enabled)
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = FieldDigesterWorker.__new__(FieldDigesterWorker)
    worker._settings = settings_mod.get_settings()
    worker._lattice = MagicMock(edges=["fake-edge-1", "fake-edge-2"])
    return worker


def test_producer_tick_is_a_noop_when_disabled(monkeypatch) -> None:
    worker = _make_worker(monkeypatch, producer_enabled="false")
    mock_cycle = MagicMock()
    monkeypatch.setattr(worker_module, "run_causal_geometry_production_cycle", mock_cycle)
    monkeypatch.setattr(worker_module, "get_learned_store", MagicMock())

    worker._causal_geometry_producer_tick()

    mock_cycle.assert_not_called()


def test_producer_tick_calls_production_cycle_with_expected_args_when_enabled(monkeypatch) -> None:
    worker = _make_worker(monkeypatch, producer_enabled="true")
    mock_cycle = MagicMock(
        return_value={
            "ok": True,
            "snapshot_id": "snap-1",
            "insufficient_data": False,
            "candidates_found": 2,
            "proposals_created": 1,
            "proposals_skipped_pending_duplicate": 1,
        }
    )
    monkeypatch.setattr(worker_module, "run_causal_geometry_production_cycle", mock_cycle)
    mock_store = MagicMock(name="learned-store")
    monkeypatch.setattr(worker_module, "get_learned_store", lambda: mock_store)

    worker._causal_geometry_producer_tick()

    mock_cycle.assert_called_once()
    kwargs = mock_cycle.call_args.kwargs
    assert kwargs["postgres_uri"] == worker._settings.postgres_uri
    assert kwargs["topology_path"] == worker._settings.lattice_path
    assert kwargs["field_edges"] == worker._lattice.edges
    assert kwargs["store"] is mock_store
    assert kwargs["window_hours"] == worker._settings.field_plasticity_producer_window_hours


def test_producer_tick_never_raises_on_a_failed_cycle(monkeypatch) -> None:
    worker = _make_worker(monkeypatch, producer_enabled="true")
    monkeypatch.setattr(
        worker_module,
        "run_causal_geometry_production_cycle",
        MagicMock(
            return_value={
                "ok": False,
                "stage": "measurement",
                "error": "postgres unreachable",
                "snapshot_id": None,
                "insufficient_data": None,
                "candidates_found": 0,
                "proposals_created": 0,
                "proposals_skipped_pending_duplicate": 0,
            }
        ),
    )
    monkeypatch.setattr(worker_module, "get_learned_store", MagicMock())

    # Must not raise.
    worker._causal_geometry_producer_tick()
