from __future__ import annotations

import importlib.util
import sys
from datetime import date, datetime, timedelta, timezone
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
        source_field_tick_id="field.tick:pf1",
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


def _legacy_self_state_policy_payload() -> dict:
    # Shaped like a pre-2026-07-22 (SelfStateV1 burn) policy decision frame
    # row -- source_self_state_id no longer exists on PolicyDecisionFrameV1.
    return {
        "schema_version": "policy.decision.frame.v1",
        "frame_id": "policy.frame:legacy:substrate_policy.v1",
        "generated_at": NOW.isoformat(),
        "source_proposal_frame_id": "proposal.frame:legacy:proposal_policy.v1",
        "source_self_state_id": "self.state:legacy",
        "decisions": [],
        "overall_risk": 0.0,
    }


def test_load_oldest_policy_frames_without_dispatch_retires_incompatible_row(monkeypatch) -> None:
    # Live incident (2026-07-22): the SelfStateV1 burn removed
    # source_self_state_id from PolicyDecisionFrameV1. This is the FIFO
    # "oldest undispatched policy frames" batch lookup (2026-07-30 perf fix:
    # renamed/batched from the old single-row load_latest_policy_frame_
    # without_dispatch -- see that method's replacement docstring) -- a
    # naive raise here crash-loops the whole worker forever (confirmed
    # live). An incompatible row must be excluded from the returned batch
    # AND write a stub, unattempted dispatch frame so the FIFO advances past
    # the bad row instead of re-selecting it forever.
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    insert_calls: list[dict] = []

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "INSERT INTO substrate_execution_dispatch_frames" in sql:
            insert_calls.append(params or {})
            result.rowcount = 1
        else:
            result.mappings.return_value.all.return_value = [
                {"policy_decision_frame_json": _legacy_self_state_policy_payload()}
            ]
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    result = store.load_oldest_policy_frames_without_dispatch(limit=200)

    assert result == []
    assert len(insert_calls) == 1
    assert insert_calls[0]["source_policy_frame_id"] == "policy.frame:legacy:substrate_policy.v1"
    assert insert_calls[0]["source_proposal_frame_id"] == "proposal.frame:legacy:proposal_policy.v1"


def _valid_policy_payload(frame_id: str) -> dict:
    return {
        "schema_version": "policy.decision.frame.v1",
        "frame_id": frame_id,
        "generated_at": NOW.isoformat(),
        "source_proposal_frame_id": f"proposal.frame:{frame_id}",
        "decisions": [],
        "overall_risk": 0.0,
    }


def test_load_oldest_policy_frames_without_dispatch_returns_valid_batch(monkeypatch) -> None:
    """2026-07-30 perf fix (docs/superpowers/specs/2026-07-30-execution-
    dispatch-staleness-discard-design.md's "Part 1c"): this now fetches a
    whole batch in one query (.mappings().all(), LIMIT :limit) instead of a
    single row (.mappings().first(), LIMIT 1) called in a loop -- confirms
    the happy path actually builds a real list of validated frames, not just
    the incompatible-row edge case above."""
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    captured_params: dict = {}

    def execute_side_effect(stmt, params=None):
        captured_params.update(params or {})
        result = MagicMock()
        result.mappings.return_value.all.return_value = [
            {"policy_decision_frame_json": _valid_policy_payload("policy.frame:a")},
            {"policy_decision_frame_json": _valid_policy_payload("policy.frame:b")},
        ]
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    result = store.load_oldest_policy_frames_without_dispatch(limit=200)

    assert [f.frame_id for f in result] == ["policy.frame:a", "policy.frame:b"]
    assert captured_params["limit"] == 200


def test_load_oldest_policy_frames_without_dispatch_skips_only_the_bad_row_in_a_mixed_batch(
    monkeypatch,
) -> None:
    """Regression guard (code review, 2026-07-30): a schema-incompatible row
    anywhere in the batch must be retired and excluded WITHOUT truncating or
    dropping the other, valid rows around it -- a real behavior improvement
    over the old single-row lookup, which returned None (and so stopped the
    whole tick's drain) the instant it hit ANY incompatible row, even with
    valid rows queued right behind it. Batch position of the bad row (here:
    the middle one) is deliberate -- proves the loop doesn't just handle a
    bad row at the start or end."""
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    insert_calls: list[dict] = []

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "INSERT INTO substrate_execution_dispatch_frames" in sql:
            insert_calls.append(params or {})
            result.rowcount = 1
        else:
            result.mappings.return_value.all.return_value = [
                {"policy_decision_frame_json": _valid_policy_payload("policy.frame:a")},
                {"policy_decision_frame_json": _legacy_self_state_policy_payload()},
                {"policy_decision_frame_json": _valid_policy_payload("policy.frame:c")},
            ]
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    result = store.load_oldest_policy_frames_without_dispatch(limit=200)

    assert [f.frame_id for f in result] == ["policy.frame:a", "policy.frame:c"]
    assert len(insert_calls) == 1
    assert insert_calls[0]["source_policy_frame_id"] == "policy.frame:legacy:substrate_policy.v1"


def test_load_freshest_policy_frame_without_dispatch_uses_not_exists(monkeypatch) -> None:
    """2026-07-30 perf fix: this direction uses WHERE NOT EXISTS, not
    LEFT JOIN ... WHERE d.frame_id IS NULL -- EXPLAIN ANALYZE confirmed live
    this is ~1500x cheaper for the DESC/newest-first direction specifically
    (0.19ms vs ~294ms), because almost nothing near "now" has been processed
    yet, so a nested-loop anti-join terminates on the very first probe."""
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    captured_sql: list[str] = []

    def execute_side_effect(stmt, params=None):
        captured_sql.append(str(stmt))
        result = MagicMock()
        result.mappings.return_value.first.return_value = {
            "policy_decision_frame_json": _valid_policy_payload("policy.frame:freshest")
        }
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    result = store.load_freshest_policy_frame_without_dispatch()

    assert result is not None
    assert result.frame_id == "policy.frame:freshest"
    assert "NOT EXISTS" in captured_sql[0]
    assert "LEFT JOIN" not in captured_sql[0]
    assert "DESC" in captured_sql[0]


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
        "source_field_tick_id": "field.tick:legacy",
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


def test_sum_risk_dispatched_today_returns_real_cumulative_risk(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {"total_risk": 4.4}
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.sum_risk_dispatched_today() == 4.4


def test_sum_risk_dispatched_today_zero_when_no_row(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = None
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.sum_risk_dispatched_today() == 0.0


def test_latest_bus_synaptic_prediction_error_returns_real_value(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "value": "0.1675",
        "generated_at": datetime.now(timezone.utc),
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.latest_bus_synaptic_prediction_error() == 0.1675


def test_latest_bus_synaptic_prediction_error_none_when_no_row(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = None
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.latest_bus_synaptic_prediction_error() is None


def test_latest_bus_synaptic_prediction_error_none_when_node_absent(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "value": None,
        "generated_at": datetime.now(timezone.utc),
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.latest_bus_synaptic_prediction_error() is None


def test_latest_bus_synaptic_prediction_error_none_when_stale(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    stale_generated_at = datetime.now(timezone.utc) - timedelta(hours=2)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "value": "0.1675",
        "generated_at": stale_generated_at,
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.latest_bus_synaptic_prediction_error() is None


def test_latest_bus_synaptic_prediction_error_none_when_unparseable(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "value": "not-a-float",
        "generated_at": datetime.now(timezone.utc),
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.latest_bus_synaptic_prediction_error() is None


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


def test_load_dispatch_result_by_dispatch_id_found(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "result_id": "result:dispatch:1",
        "status": "success",
        "result_json": {"observation": "steady"},
        "raw_len": 6,
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    result = store.load_dispatch_result_by_dispatch_id("dispatch:1")

    assert result == {
        "result_id": "result:dispatch:1",
        "status": "success",
        "result_json": {"observation": "steady"},
        "raw_len": 6,
    }


def test_load_dispatch_result_by_dispatch_id_parses_json_string(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "result_id": "result:dispatch:1",
        "status": "success",
        "result_json": '{"observation": "steady"}',
        "raw_len": 6,
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    result = store.load_dispatch_result_by_dispatch_id("dispatch:1")

    assert result["result_json"] == {"observation": "steady"}


def test_load_dispatch_result_by_dispatch_id_none_when_no_row(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = None
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.load_dispatch_result_by_dispatch_id("dispatch:missing") is None


# ---------------------------------------------------------------------------
# Self-calibrating daily risk ceiling (2026-07-29)
# ---------------------------------------------------------------------------


def test_sum_uncapped_risk_for_day_reads_prepared_and_dispatched_arrays(monkeypatch) -> None:
    """Real feedstock for the EWMA baseline: prepared_for_dispatch candidates
    left unsent (from `candidates`) PLUS everything already spent (from
    `dispatched_candidates`) -- both arrays, not just dispatched_candidates
    (that would be the exact right-censored-spend bug this method exists to
    avoid)."""
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {"total_risk": 817.65}
    monkeypatch.setattr(store, "_engine", fake_engine)

    total = store.sum_uncapped_risk_for_day(
        datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc),
    )

    assert total == 817.65
    sql = str(conn.execute.call_args.args[0])
    assert "prepared_for_dispatch" in sql
    assert "'dispatched_candidates'" in sql
    assert "UNION ALL" in sql
    params = conn.execute.call_args.args[1]
    assert params["day_start"] == datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc)
    assert params["day_end"] == datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc)


def test_sum_uncapped_risk_for_day_zero_when_no_data(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {"total_risk": None}
    monkeypatch.setattr(store, "_engine", fake_engine)

    total = store.sum_uncapped_risk_for_day(
        datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc),
    )

    assert total == 0.0


def test_load_latest_daily_risk_baseline_returns_latest_row(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "ewma": "140.0",
        "var": "4000.0",
        "n": "2",
        "last_day": "2026-07-29",
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    result = store.load_latest_daily_risk_baseline()

    assert result == {
        "daily_risk_baseline_ewma": 140.0,
        "daily_risk_baseline_ewma_var": 4000.0,
        "daily_risk_baseline_ewma_n": 2,
        "daily_risk_baseline_last_day": "2026-07-29",
    }


def test_load_latest_daily_risk_baseline_none_when_no_rows(monkeypatch) -> None:
    """Truly no dispatch frame rows at all -- distinct from a real row that
    predates these fields (see the pre-migration test below), which must
    NOT return None."""
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = None
    monkeypatch.setattr(store, "_engine", fake_engine)

    assert store.load_latest_daily_risk_baseline() is None


def test_load_latest_daily_risk_baseline_defaults_missing_fields_for_pre_migration_row(
    monkeypatch,
) -> None:
    """A real row exists but predates these fields (pre-2026-07-29 dispatch
    frame) -- all four JSON keys come back None. Must degrade to cold-start
    defaults so the caller's own cold-start-seed path kicks in, not raise or
    silently report None (a real row is not "no history")."""
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "ewma": None,
        "var": None,
        "n": None,
        "last_day": None,
    }
    monkeypatch.setattr(store, "_engine", fake_engine)

    result = store.load_latest_daily_risk_baseline()

    assert result == {
        "daily_risk_baseline_ewma": 0.0,
        "daily_risk_baseline_ewma_var": 0.0,
        "daily_risk_baseline_ewma_n": 0,
        "daily_risk_baseline_last_day": None,
    }


def test_most_recent_closed_day_with_data_returns_day_and_real_uncapped_total(
    monkeypatch,
) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "AS day" in sql:
            result.mappings.return_value.first.return_value = {"day": date(2026, 7, 28)}
        elif "total_risk" in sql:
            result.mappings.return_value.first.return_value = {"total_risk": 817.65}
        else:
            raise AssertionError(f"unexpected SQL: {sql}")
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    result = store.most_recent_closed_day_with_data(
        datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc)
    )

    assert result == ("2026-07-28", 817.65)


def test_most_recent_closed_day_with_data_none_when_no_history(monkeypatch) -> None:
    store = ExecutionDispatchRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.mappings.return_value.first.return_value = None
    monkeypatch.setattr(store, "_engine", fake_engine)

    result = store.most_recent_closed_day_with_data(
        datetime(2026, 7, 29, 0, 0, tzinfo=timezone.utc)
    )

    assert result is None
