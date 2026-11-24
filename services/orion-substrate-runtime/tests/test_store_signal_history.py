"""Store methods for substrate_node_signal_history (PR #3781 Patch 1)."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TS = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class _Engine:
    def __init__(self, rows=None, one=None, rowcount=0):
        self.executed, self._rows, self._one, self._rowcount = [], rows or [], one, rowcount

    def _conn(self):
        conn = MagicMock()

        def execute(stmt, params=None):
            self.executed.append((str(stmt), params))
            result = MagicMock()
            result.fetchall.return_value = self._rows
            result.fetchone.return_value = self._one
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


def test_save_signal_samples_inserts_kind_idempotently():
    eng = _Engine()
    n = _store(eng).save_signal_history_samples([("node:substrate.strain", TS, 1.0, "alarm")])
    assert n == 1
    sql, params = eng.executed[0]
    assert "INSERT INTO substrate_node_signal_history" in sql
    assert "ON CONFLICT (node_id, observed_at) DO NOTHING" in sql
    assert params == [{"node_id": "node:substrate.strain", "observed_at": TS, "value": 1.0, "kind": "alarm"}]


def test_save_empty_skips_db():
    eng = _Engine()
    assert _store(eng).save_signal_history_samples([]) == 0
    assert eng.executed == []


def test_prune_signal_history_returns_rowcount():
    eng = _Engine(rowcount=4)
    assert _store(eng).prune_signal_history(older_than=TS) == 4
    sql, params = eng.executed[0]
    assert "DELETE FROM substrate_node_signal_history" in sql and params == {"older_than": TS}


def test_read_both_defaults_to_the_read_both_ids_and_makes_naive_utc():
    eng = _Engine(rows=[("node:substrate.cabinet", TS.replace(tzinfo=None), 0.0)])
    rows = _store(eng).fetch_signal_history_read_both(since=TS)
    assert rows == [("node:substrate.cabinet", TS, 0.0)]
    sql, params = eng.executed[0]
    assert "UNION ALL" in sql
    # Only the per-node read_both ids (the three alarms); never the cameras (design rev 3, #3827).
    assert params["since"] == TS
    assert params["node_ids"] == ["node:substrate.cabinet", "node:substrate.strain", "node:substrate.street"]
    _store(eng).fetch_signal_history_read_both(since=TS, node_ids=["node:substrate.perception"])
    assert eng.executed[-1][1]["node_ids"] == []


def test_count_and_event_times_read_both_scope_to_one_node():
    eng = _Engine(one=(42,))
    assert _store(eng).count_signal_history_read_both(node_id="node:substrate.strain", since=TS) == 42
    assert eng.executed[0][1] == {"since": TS, "node_ids": ["node:substrate.strain"]}
    eng2 = _Engine(rows=[(TS.replace(tzinfo=None),)])
    assert _store(eng2).fetch_signal_event_times_read_both(node_id="node:substrate.strain", since=TS) == [TS]
    assert "value > 0" in eng2.executed[0][0]
