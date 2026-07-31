"""Unit tests for the codebase-mass consumer patch's two new store methods
(docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md, "Producer
+ consumer patch design"): `get_latest_codebase_mass_baseline` (cold-start-
tolerant read) and `save_codebase_mass_baseline` (this tick's sole write
path, into a table nothing else in the repo touches).

Mocked here (same pattern as the sibling test_store_attention_self_model.py
uses); the round trip against a real Postgres instance was verified directly
during development -- see this patch's PR report.
"""

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

from orion.substrate.prediction_error import CodebaseMassBaseline, _DomainEwmaBaseline


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


def test_get_latest_codebase_mass_baseline_parses_row():
    baseline = CodebaseMassBaseline(
        git=_DomainEwmaBaseline(ewma=1.0, variance=2.0, n=3),
        pr=_DomainEwmaBaseline(ewma=4.0, variance=5.0, n=6),
        graph=_DomainEwmaBaseline(ewma=7.0, variance=8.0, n=9),
    )
    row = {"baseline_json": baseline.to_json_dict()}
    eng = _RecordingEngine(first_row=row)
    store = _store_with(eng)

    restored = store.get_latest_codebase_mass_baseline()

    assert restored == baseline
    sql = eng.executed[0][0]
    assert "SELECT baseline_json FROM substrate_codebase_mass_baseline" in sql
    assert "ORDER BY generated_at DESC" in sql


def test_get_latest_codebase_mass_baseline_returns_fresh_when_empty():
    """No row yet is a real, honest cold-start state -- a fresh baseline, not
    None and not an error (matches every domain's own cold-start handling in
    orion/substrate/prediction_error.py, unlike get_latest_field_attention_
    frame's None-on-empty, since CodebaseMassBaseline's own default IS the
    correct cold-start value, not an absence to special-case downstream)."""
    eng = _RecordingEngine(first_row=None)
    store = _store_with(eng)
    assert store.get_latest_codebase_mass_baseline() == CodebaseMassBaseline()


def test_get_latest_codebase_mass_baseline_fails_open_on_error():
    eng = MagicMock()
    eng.connect.side_effect = RuntimeError("db down")
    store = _store_with(eng)
    assert store.get_latest_codebase_mass_baseline() == CodebaseMassBaseline()  # must not raise


def test_get_latest_codebase_mass_baseline_handles_string_encoded_json():
    """Some drivers return jsonb as a raw string, not a pre-parsed dict --
    matches get_latest_field_attention_frame's own defensive handling."""
    baseline = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=1.0, variance=2.0, n=3))
    import json as _json

    row = {"baseline_json": _json.dumps(baseline.to_json_dict())}
    eng = _RecordingEngine(first_row=row)
    store = _store_with(eng)
    assert store.get_latest_codebase_mass_baseline() == baseline


def test_save_codebase_mass_baseline_inserts_and_prunes():
    eng = _RecordingEngine()
    store = _store_with(eng)
    baseline = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=1.0, variance=2.0, n=3))

    store.save_codebase_mass_baseline(baseline, retention_days=30.0)

    sqls = " ".join(s for s, _ in eng.executed)
    assert "INSERT INTO substrate_codebase_mass_baseline" in sqls
    assert "DELETE FROM substrate_codebase_mass_baseline" in sqls
    assert "30.0 days" in sqls
    insert_params = eng.executed[0][1]
    assert insert_params["baseline_id"].startswith("codebase-mass-baseline-")
    assert insert_params["baseline_json"].adapted == baseline.to_json_dict()
