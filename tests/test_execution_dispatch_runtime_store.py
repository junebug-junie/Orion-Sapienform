from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-execution-dispatch-runtime"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_store_class():
    spec = importlib.util.spec_from_file_location(
        "execution_dispatch_runtime_store",
        SVC / "app" / "store.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ExecutionDispatchRuntimeStore


ExecutionDispatchRuntimeStore = _load_store_class()

from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _frame() -> ExecutionDispatchFrameV1:
    return ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:policy.frame:pf1:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:pf1:substrate_policy.v1",
        source_proposal_frame_id="proposal.frame:pf1:proposal_policy.v1",
        source_self_state_id="self.state:pf1",
        dispatch_mode="dry_run",
    )


def test_save_and_load_latest(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "INSERT INTO substrate_execution_dispatch_frames" in sql:
            result.rowcount = 1
        elif "source_policy_frame_id" in sql and "ORDER BY" in sql:
            result.mappings.return_value.first.return_value = None
        else:
            result.mappings.return_value.first.return_value = {"dispatch_frame_json": payload}
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    store.save_dispatch_frame(_frame())
    loaded = store.load_latest_dispatch_frame()
    assert loaded is not None
    assert loaded.frame_id == _frame().frame_id


def test_load_by_policy_frame_id(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        if "source_policy_frame_id" in str(stmt):
            result.mappings.return_value.first.return_value = {"dispatch_frame_json": payload}
        else:
            result.mappings.return_value.first.return_value = None
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    loaded = store.load_dispatch_frame_for_policy_frame("policy.frame:pf1:substrate_policy.v1")
    assert loaded is not None
    assert loaded.source_policy_frame_id == "policy.frame:pf1:substrate_policy.v1"


def test_save_idempotent_by_frame_id(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
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

    store.save_dispatch_frame(_frame())
    assert any("ON CONFLICT (frame_id)" in sql for sql in calls)


def _incompatible_dispatch_frame_payload() -> dict:
    # dispatch_status="dispatched" without dispatched_at/result_ref/dispatch_error
    # is now rejected by ExecutionDispatchCandidateV1's evidence validator
    # (2026-07-13 status-honesty patch). A historical row shaped like this
    # would previously have loaded fine; it must now degrade to None instead
    # of raising, the same way a legacy self_state row does.
    return {
        "schema_version": "execution.dispatch.frame.v1",
        "frame_id": "execution.dispatch.frame:policy.frame:legacy:execution_dispatch_policy.v1",
        "generated_at": NOW.isoformat(),
        "source_policy_frame_id": "policy.frame:legacy:substrate_policy.v1",
        "source_proposal_frame_id": "proposal.frame:legacy:proposal_policy.v1",
        "source_self_state_id": "self.state:legacy",
        "dispatch_mode": "dispatch_read_only",
        "dispatched_candidates": [
            {
                "dispatch_id": "dispatch:proposal:inspect:execution_dispatch_policy.v1",
                "source_decision_id": "pd1",
                "source_proposal_id": "proposal:inspect:state",
                "dispatch_status": "dispatched",
                "dispatch_mode": "dispatch_read_only",
                "dispatch_kind": "inspect",
                "target_id": "t1",
                "target_kind": "capability",
                "risk_score": 0.05,
                "confidence_score": 0.9,
            }
        ],
    }


def test_load_latest_dispatch_frame_degrades_to_none_on_legacy_incompatible_row(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "dispatch_frame_json": _incompatible_dispatch_frame_payload(),
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.load_latest_dispatch_frame() is None


def test_load_dispatch_frame_for_policy_frame_degrades_to_none_on_legacy_incompatible_row(
    monkeypatch,
) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "dispatch_frame_json": _incompatible_dispatch_frame_payload(),
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.load_dispatch_frame_for_policy_frame("policy.frame:legacy:substrate_policy.v1") is None


def test_load_self_state_degrades_to_none_on_legacy_incompatible_row(monkeypatch) -> None:
    # Live incident (2026-07-12): see the identical test in
    # tests/test_policy_runtime_store.py for the full explanation. Looked up
    # by a fixed self_state_id (a policy frame's source), so a naive raise
    # would block this (and every queued) dispatch frame forever.
    legacy_payload = {
        "schema_version": "self.state.v1",
        "self_state_id": "self.state:legacy",
        "generated_at": NOW.isoformat(),
        "source_field_tick_id": "tick",
        "source_field_generated_at": NOW.isoformat(),
        "source_attention_frame_id": "frame",
        "source_attention_generated_at": NOW.isoformat(),
        "self_state_policy_id": "self_state_policy.v1",
        "overall_condition": "steady",
        "overall_intensity": 0.4,
        "overall_confidence": 0.8,
        "dimensions": {
            "policy_pressure": {
                "dimension_id": "policy_pressure",
                "score": 0.0,
                "confidence": 0.5,
                "dominant_evidence": [],
                "reasons": ["policy_pressure from field+attention channel synthesis"],
            }
        },
    }
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "self_state_json": legacy_payload,
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.load_self_state("self.state:legacy") is None


def test_save_dispatch_result_inserts_expected_row(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    calls: list[tuple[str, dict]] = []

    def execute_side_effect(stmt, params=None):
        calls.append((str(stmt), params or {}))
        return MagicMock()

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    store.save_dispatch_result(
        result_id="result:dispatch:1",
        dispatch_id="dispatch:1",
        frame_id="execution.dispatch.frame:1",
        status="success",
        result_json={"observation": "steady", "salient_facts": [], "confidence": 0.7},
        raw_len=6,
    )

    assert len(calls) == 1
    sql, params = calls[0]
    assert "INSERT INTO substrate_dispatch_results" in sql
    assert "ON CONFLICT (result_id) DO UPDATE" in sql
    assert params["result_id"] == "result:dispatch:1"
    assert params["dispatch_id"] == "dispatch:1"
    assert params["status"] == "success"
    assert params["raw_len"] == 6


def test_count_dispatches_today_returns_row_count(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {"n": 3}
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.count_dispatches_today() == 3


def test_count_dispatches_today_zero_when_no_row(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = None
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.count_dispatches_today() == 0


def test_recent_dispatch_result_statuses_returns_ordered_list(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.all.return_value = [
        {"status": "success"},
        {"status": "empty"},
        {"status": "failed"},
    ]
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.recent_dispatch_result_statuses(10) == ["success", "empty", "failed"]


def test_recent_dispatch_result_statuses_empty_when_no_rows(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.all.return_value = []
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.recent_dispatch_result_statuses(10) == []
