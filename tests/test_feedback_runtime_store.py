from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-feedback-runtime"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_store_class():
    spec = importlib.util.spec_from_file_location(
        "feedback_runtime_store",
        SVC / "app" / "store.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FeedbackRuntimeStore


FeedbackRuntimeStore = _load_store_class()

from orion.schemas.feedback_frame import FeedbackFrameV1  # noqa: E402

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)


def _frame() -> FeedbackFrameV1:
    return FeedbackFrameV1(
        frame_id="feedback.frame:execution.dispatch.frame:pf1:feedback_policy.v1",
        generated_at=NOW,
        source_execution_dispatch_frame_id="execution.dispatch.frame:pf1:execution_dispatch_policy.v1",
        source_policy_frame_id="policy.frame:pf1:substrate_policy.v1",
        outcome_status="dry_run_only",
        outcome_score=0.5,
        confidence_score=0.9,
    )


def test_save_and_load_latest(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = FeedbackRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "INSERT INTO substrate_feedback_frames" in sql:
            result.rowcount = 1
        elif "source_execution_dispatch_frame_id" in sql and "LEFT JOIN" in sql:
            result.mappings.return_value.first.return_value = None
        else:
            result.mappings.return_value.first.return_value = {"feedback_frame_json": payload}
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    store.save_feedback_frame(_frame())
    loaded = store.load_latest_feedback_frame()
    assert loaded is not None
    assert loaded.frame_id == _frame().frame_id


def test_load_by_dispatch_frame_id(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = FeedbackRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        if "source_execution_dispatch_frame_id" in str(stmt):
            result.mappings.return_value.first.return_value = {"feedback_frame_json": payload}
        else:
            result.mappings.return_value.first.return_value = None
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    loaded = store.load_feedback_frame_for_dispatch(
        "execution.dispatch.frame:pf1:execution_dispatch_policy.v1"
    )
    assert loaded is not None
    assert loaded.source_execution_dispatch_frame_id == (
        "execution.dispatch.frame:pf1:execution_dispatch_policy.v1"
    )


def test_save_idempotent_by_frame_id(monkeypatch) -> None:
    store = FeedbackRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    calls: list[str] = []

    def execute_side_effect(stmt, params=None):
        calls.append(str(stmt))
        return MagicMock()

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    store.save_feedback_frame(_frame())
    assert any("ON CONFLICT (frame_id)" in sql for sql in calls)
