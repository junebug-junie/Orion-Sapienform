"""Unit tests for P2's two new store methods (docs/superpowers/specs/2026-08-
12-perception-frontier-design.md): `get_latest_perception_embedding_baseline`
(cold-start-tolerant, per-stream read) and
`save_perception_embedding_baseline` (this stream's sole write path, into a
table nothing else in the repo touches).

Mocked here, same pattern as test_store_codebase_mass_baseline.py -- the real
round trip against Postgres is not re-verified in this unit test (no live DB
in this environment); the SQL shape and cold-start/error-handling behavior
are what's under test.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from orion.substrate.prediction_error import PerceptionEmbeddingBaseline


class _RecordingEngine:
    def __init__(self, first_row=None):
        self.executed = []
        self._first_row = first_row

    @contextmanager
    def begin(self):
        conn = MagicMock()

        def execute(stmt, params=None):
            self.executed.append((str(stmt), params))
            return MagicMock()

        conn.execute.side_effect = execute
        yield conn

    @contextmanager
    def connect(self):
        conn = MagicMock()

        def execute(stmt, params=None):
            self.executed.append((str(stmt), params))
            m = MagicMock()
            m.mappings.return_value.first.return_value = self._first_row
            return m

        conn.execute.side_effect = execute
        yield conn


def _store_with(engine):
    from app.store import BiometricsSubstrateStore

    store = BiometricsSubstrateStore.__new__(BiometricsSubstrateStore)
    store._engine = engine
    return store


def test_get_latest_perception_embedding_baseline_parses_row():
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(0.1, 0.2, 0.3), n=4)
    row = {"baseline_json": baseline.to_json_dict()}
    eng = _RecordingEngine(first_row=row)
    store = _store_with(eng)

    restored = store.get_latest_perception_embedding_baseline("cam0")

    assert restored == baseline
    sql, params = eng.executed[0]
    assert "SELECT baseline_json FROM substrate_perception_embedding_baseline" in sql
    assert "WHERE stream_id = :stream_id" in sql
    assert "ORDER BY generated_at DESC" in sql
    assert params == {"stream_id": "cam0"}


def test_get_latest_perception_embedding_baseline_returns_fresh_when_empty():
    eng = _RecordingEngine(first_row=None)
    store = _store_with(eng)
    assert store.get_latest_perception_embedding_baseline("cam0") == PerceptionEmbeddingBaseline()


def test_get_latest_perception_embedding_baseline_fails_open_on_error():
    eng = MagicMock()
    eng.connect.side_effect = RuntimeError("db down")
    store = _store_with(eng)
    assert store.get_latest_perception_embedding_baseline("cam0") == PerceptionEmbeddingBaseline()  # must not raise


def test_get_latest_perception_embedding_baseline_handles_string_encoded_json():
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0,), n=1)
    import json as _json

    row = {"baseline_json": _json.dumps(baseline.to_json_dict())}
    eng = _RecordingEngine(first_row=row)
    store = _store_with(eng)
    assert store.get_latest_perception_embedding_baseline("cam0") == baseline


def test_save_perception_embedding_baseline_inserts_and_prunes_scoped_to_stream():
    """Review finding: the retention DELETE must be scoped by stream_id, not
    global -- an unscoped DELETE would let a busy stream's save silently
    prune an unrelated, infrequently-active stream's only row."""
    eng = _RecordingEngine()
    store = _store_with(eng)
    baseline = PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=2)

    store.save_perception_embedding_baseline("cam0", baseline, retention_days=30.0)

    sqls = " ".join(s for s, _ in eng.executed)
    assert "INSERT INTO substrate_perception_embedding_baseline" in sqls
    assert "DELETE FROM substrate_perception_embedding_baseline" in sqls
    assert "30.0 days" in sqls
    insert_params = eng.executed[0][1]
    assert insert_params["stream_id"] == "cam0"
    assert insert_params["baseline_id"].startswith("perception-embedding-baseline-")
    assert insert_params["baseline_json"].adapted == baseline.to_json_dict()

    delete_sql, delete_params = eng.executed[1]
    assert "DELETE FROM substrate_perception_embedding_baseline" in delete_sql
    assert "WHERE stream_id = :stream_id" in delete_sql
    assert delete_params == {"stream_id": "cam0"}
