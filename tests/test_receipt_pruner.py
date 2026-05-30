from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/orion-substrate-runtime"))

import app.receipt_pruner as pruner
from app.receipt_pruner import (
    SAFE_PRUNE_DELETE_SQL_FRAGMENT,
    maybe_run_emergency_prune,
    refresh_pressure_cache,
)


@dataclass
class _PrunerSettings:
    receipt_critical_table_gb: float = 20.0
    receipt_max_table_gb: float = 25.0
    receipt_prune_interval_sec: float = 900.0
    receipt_postgres_data_path: str = "/nonexistent"
    receipt_disk_critical_pct: float = 85.0
    receipt_warn_table_gb: float = 15.0


def test_safe_prune_sql_requires_expires_and_applied_or_error():
    assert "expires_at < now()" in SAFE_PRUNE_DELETE_SQL_FRAGMENT
    assert "substrate_field_applied_deltas" in SAFE_PRUNE_DELETE_SQL_FRAGMENT


def test_emergency_success_delete_requires_applied_deltas_guard():
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    pruner.run_emergency_prune(engine)
    calls = [str(c.args[0]) for c in conn.execute.call_args_list]
    success_sql = calls[0]
    debug_sql = calls[1]
    assert "receipt_kind = 'success'" in success_sql
    assert "substrate_field_applied_deltas" in success_sql
    assert "NOT EXISTS" in success_sql
    assert "receipt_kind = 'debug_sample'" in debug_sql
    assert "substrate_field_applied_deltas" in debug_sql


def test_maybe_run_emergency_prune_respects_cooldown():
    pruner._last_emergency_prune_monotonic = 0.0
    settings = _PrunerSettings(receipt_critical_table_gb=0.001)
    engine = MagicMock()

    with patch("app.receipt_pruner.table_size_gb", return_value=100.0):
        with patch("app.receipt_pruner.run_emergency_prune") as mock_emergency:
            assert maybe_run_emergency_prune(engine, settings) is True
            assert mock_emergency.call_count == 1
            assert maybe_run_emergency_prune(engine, settings) is False
            assert mock_emergency.call_count == 1


def test_refresh_pressure_cache_updates_cached_state():
    settings = _PrunerSettings(receipt_critical_table_gb=100.0)
    engine = MagicMock()

    with patch("app.receipt_pruner.table_size_gb", return_value=50.0):
        refresh_pressure_cache(engine, settings)
        disk_critical, table_critical = pruner.get_cached_pressure_state()
        assert disk_critical is False
        assert table_critical is False
