"""Unit tests for `BiometricsSubstrateStore.save_codebase_delta_log` (docs/
superpowers/specs/2026-07-30-codebase-mass-signal-design.md follow-on: raw
per-tick event history for a future Hub "cocreation signals" analytics tab).

Mocked here (same pattern as the sibling test_store_codebase_mass_baseline.py
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

from orion.schemas.codebase_delta import CodebaseDeltaV1, GitDeltaPayloadV1

_NOW = datetime(2026, 7, 31, tzinfo=timezone.utc)


class _RecordingEngine:
    def __init__(self):
        self.executed = []

    @contextmanager
    def begin(self):
        conn = MagicMock()

        def execute(stmt, params=None):
            self.executed.append((str(stmt), params))
            return MagicMock()

        conn.execute.side_effect = execute
        yield conn


def _store_with(engine):
    from app.store import BiometricsSubstrateStore

    store = BiometricsSubstrateStore.__new__(BiometricsSubstrateStore)
    store._engine = engine
    return store


def _event() -> CodebaseDeltaV1:
    return CodebaseDeltaV1(
        domain="git",
        observed_at=_NOW,
        git=GitDeltaPayloadV1(
            prev_sha="a" * 40, head_sha="b" * 40, commit_count=1,
            files_added=0, files_deleted=0, files_modified=1,
            lines_added=520, lines_removed=0,
        ),
    )


def test_save_codebase_delta_log_inserts_and_prunes():
    eng = _RecordingEngine()
    store = _store_with(eng)

    store.save_codebase_delta_log(_event(), score=0.5, retention_days=180.0)

    sqls = " ".join(s for s, _ in eng.executed)
    assert "INSERT INTO substrate_codebase_delta_log" in sqls
    assert "DELETE FROM substrate_codebase_delta_log" in sqls
    assert "180.0 days" in sqls
    insert_params = eng.executed[0][1]
    assert insert_params["event_id"].startswith("codebase-delta-")
    assert insert_params["domain"] == "git"
    assert insert_params["observed_at"] == _NOW
    assert insert_params["score"] == 0.5
    assert insert_params["event_json"].adapted == _event().model_dump(mode="json")


def test_save_codebase_delta_log_is_idempotent_on_identical_event():
    """Same event content + same observed_at -> same digest -> ON CONFLICT
    DO NOTHING makes redelivery safe (matches save_codebase_mass_baseline's
    own idempotency reasoning)."""
    eng = _RecordingEngine()
    store = _store_with(eng)
    event = _event()

    store.save_codebase_delta_log(event, score=0.5, retention_days=180.0)
    store.save_codebase_delta_log(event, score=0.5, retention_days=180.0)

    first_id = eng.executed[0][1]["event_id"]
    second_id = eng.executed[2][1]["event_id"]  # insert, delete, insert, delete
    assert first_id == second_id
