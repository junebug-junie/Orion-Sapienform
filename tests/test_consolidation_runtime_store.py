from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-consolidation-runtime"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_store_class():
    spec = importlib.util.spec_from_file_location(
        "consolidation_runtime_store",
        SVC / "app" / "store.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ConsolidationRuntimeStore


ConsolidationRuntimeStore = _load_store_class()

from orion.schemas.consolidation_frame import ConsolidationFrameV1, ExpectationV1, SparseTensorSliceV1  # noqa: E402

NOW = datetime(2026, 5, 25, 15, 30, tzinfo=timezone.utc)
START = datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc)


def _frame() -> ConsolidationFrameV1:
    return ConsolidationFrameV1(
        frame_id=(
            "consolidation.frame:2026-05-25T14:30:00+00:00:"
            "2026-05-25T15:30:00+00:00:consolidation_policy.v1"
        ),
        generated_at=NOW,
        window_start=START,
        window_end=NOW,
        source_counts={"self_state": 1, "feedback": 2},
    )


def test_save_and_load_for_window(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = ConsolidationRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "INSERT INTO substrate_consolidation_frames" in sql:
            result.rowcount = 1
        elif "WHERE frame_id = :frame_id" in sql:
            result.mappings.return_value.first.return_value = {
                "consolidation_frame_json": payload
            }
        else:
            result.mappings.return_value.first.return_value = {
                "consolidation_frame_json": payload
            }
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    store.save_consolidation_frame(_frame())
    loaded = store.load_consolidation_frame_for_window(_frame().frame_id)
    assert loaded is not None
    assert loaded.frame_id == _frame().frame_id


def test_load_latest(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = ConsolidationRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        result.mappings.return_value.first.return_value = {
            "consolidation_frame_json": payload
        }
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    loaded = store.load_latest_consolidation_frame()
    assert loaded is not None
    assert loaded.source_counts["feedback"] == 2


def test_save_idempotent_by_frame_id(monkeypatch) -> None:
    store = ConsolidationRuntimeStore("postgresql://test:test@localhost/test")
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

    store.save_consolidation_frame(_frame())
    assert any("ON CONFLICT (frame_id) DO NOTHING" in sql for sql in calls)


def test_upsert_expectations_idempotent(monkeypatch) -> None:
    store = ConsolidationRuntimeStore("postgresql://test:test@localhost/test")
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

    expectation = ExpectationV1(
        expectation_id="expectation:motif:loaded_but_reliable:consolidation_policy.v1:reliability_clear",
        trigger_motif_id="motif:loaded_but_reliable:consolidation_policy.v1",
        expected_outcome_kind="reliability_clear",
        confidence_score=0.7,
        support_count=3,
    )
    store.upsert_expectations([expectation])
    store.upsert_expectations([expectation])
    assert sum("INSERT INTO substrate_expectations" in sql for sql in calls) == 2
    assert all("ON CONFLICT (expectation_id) DO UPDATE" in sql for sql in calls if "substrate_expectations" in sql)


def test_save_tensor_slices_idempotent(monkeypatch) -> None:
    store = ConsolidationRuntimeStore("postgresql://test:test@localhost/test")
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

    tensor_slice = SparseTensorSliceV1(
        tensor_id="tensor:field_attention_self:2026-05-25T14:30:00+00:00:2026-05-25T15:30:00+00:00",
        tensor_kind="field_attention_self",
        axes=["time_bucket", "self_condition", "attention_target", "dimension"],
        coordinates=[{"time_bucket": "2026-05-25T15:00:00+00:00", "self_condition": "loaded", "attention_target": "node:athena", "dimension": "execution_pressure"}],
        values=[0.8],
    )
    store.save_tensor_slices([tensor_slice], START, NOW)
    store.save_tensor_slices([tensor_slice], START, NOW)
    assert sum("INSERT INTO substrate_tensor_slices" in sql for sql in calls) == 2
    assert all("ON CONFLICT (tensor_id) DO NOTHING" in sql for sql in calls if "substrate_tensor_slices" in sql)
