"""Store methods for substrate_node_prediction_error_history
(manual_migration_node_prediction_error_history_v1.sql)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

TS = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_node_prediction_error_history_v1.sql"


class _Engine:
    def __init__(self, rows=None, rowcount=0):
        self.executed = []
        self._rows = rows or []
        self._rowcount = rowcount

    def _conn(self):
        conn = MagicMock()

        def execute(stmt, params=None):
            self.executed.append((str(stmt), params))
            result = MagicMock()
            result.fetchall.return_value = self._rows
            result.rowcount = self._rowcount
            return result

        conn.execute.side_effect = execute
        return conn

    @contextmanager
    def begin(self):
        yield self._conn()

    @contextmanager
    def connect(self):
        yield self._conn()


def _store(engine):
    from app.store import BiometricsSubstrateStore

    store = BiometricsSubstrateStore.__new__(BiometricsSubstrateStore)
    store._engine = engine
    return store


def test_save_samples_is_idempotent_insert():
    eng = _Engine()
    n = _store(eng).save_prediction_error_history_samples(
        [("node:substrate.chat", TS, 0.14), ("node:substrate.route", TS, 0.0)]
    )
    assert n == 2
    sql, params = eng.executed[0]
    assert "INSERT INTO substrate_node_prediction_error_history" in sql
    assert "ON CONFLICT (node_id, observed_at) DO NOTHING" in sql
    assert params[1] == {"node_id": "node:substrate.route", "observed_at": TS, "value": 0.0}


def test_save_empty_samples_skips_db():
    eng = _Engine()
    assert _store(eng).save_prediction_error_history_samples([]) == 0
    assert eng.executed == []


def test_fetch_history_maps_rows():
    eng = _Engine(rows=[("node:substrate.chat", TS, 0.25)])
    rows = _store(eng).fetch_prediction_error_history(since=TS)
    assert rows == [("node:substrate.chat", TS, 0.25)]
    assert eng.executed[0][1] == {"since": TS}


def test_prune_returns_rowcount():
    eng = _Engine(rowcount=7)
    assert _store(eng).prune_prediction_error_history(older_than=TS) == 7
    sql, params = eng.executed[0]
    assert "DELETE FROM substrate_node_prediction_error_history" in sql
    assert params == {"older_than": TS}


def test_migration_matches_store_columns_and_pk():
    sql = MIGRATION.read_text().lower()
    assert "create table if not exists substrate_node_prediction_error_history" in sql
    for col in ("node_id text", "observed_at timestamptz", "value double precision", "recorded_at timestamptz"):
        assert col in sql
    assert "primary key (node_id, observed_at)" in sql
