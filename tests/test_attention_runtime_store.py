from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-attention-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

from app.store import AttentionRuntimeStore  # noqa: E402
from orion.schemas.field_attention_frame import FieldAttentionFrameV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _frame() -> FieldAttentionFrameV1:
    return FieldAttentionFrameV1(
        frame_id="attention.frame:tick_a:field_attention_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_a",
        source_field_generated_at=NOW,
        overall_salience=0.5,
    )


def _mock_engine_for_frames(*, latest: dict | None, by_tick: dict | None = None):
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "substrate_field_state" in sql:
            result.mappings.return_value.first.return_value = None
        elif "source_field_tick_id" in sql:
            row = {"frame_json": by_tick} if by_tick else None
            result.mappings.return_value.first.return_value = row
        else:
            row = {"frame_json": latest} if latest else None
            result.mappings.return_value.first.return_value = row
        return result

    conn.execute.side_effect = execute_side_effect
    return fake_engine


def test_load_latest_field(monkeypatch) -> None:
    from orion.schemas.field_state import FieldStateV1

    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_field",
        node_vectors={"node:athena": {"cortex_exec_step_load": 0.5}},
    )
    payload = field.model_dump(mode="json")
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        result.mappings.return_value.first.return_value = {"field_json": payload}
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    loaded = store.load_latest_field()
    assert loaded is not None
    assert loaded.tick_id == "tick_field"


def test_load_latest_attention_frame(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    monkeypatch.setattr(store, "_engine", _mock_engine_for_frames(latest=payload))
    loaded = store.load_latest_attention_frame()
    assert loaded is not None
    assert loaded.frame_id == _frame().frame_id


def test_load_attention_frame_for_field_tick(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    monkeypatch.setattr(store, "_engine", _mock_engine_for_frames(latest=None, by_tick=payload))
    loaded = store.load_attention_frame_for_field_tick("tick_a")
    assert loaded is not None
    assert loaded.source_field_tick_id == "tick_a"


def test_save_attention_frame_idempotent(monkeypatch) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = _mock_engine_for_frames(latest=None)
    store._engine = fake_engine
    store.save_attention_frame(_frame())
    conn = fake_engine.begin.return_value.__enter__.return_value
    assert conn.execute.called
    sql = str(conn.execute.call_args[0][0])
    assert "ON CONFLICT (frame_id)" in sql


# -- advance_node_prediction_error_baseline (2026-07-30 fix) -------------------
# See orion/sentience_striving_program/README.md §12 for the live incident this
# method exists to fix: the old per-tick rolling-window recompute
# (load_prediction_error_history) let a target's real n_samples reset to
# whatever survived a ~30-minute retention window. This method persists a
# cumulative baseline instead.


def _mock_engine_for_baseline(
    *,
    existing_row: dict | None,
    new_rows: list[dict],
    version_column: bool = False,
    persisted: list[dict] | None = None,
):
    """``version_column`` answers the definition_version column probe; False keeps
    the pre-2026-09-25 behaviour the older tests below were written against.
    ``persisted`` collects the params of every baseline upsert."""
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    calls: list[str] = []

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "information_schema.columns" in sql:
            calls.append("probe_version_column")
            result.scalar.return_value = version_column
        elif "substrate_node_prediction_error_baseline" in sql and "SELECT" in sql:
            calls.append("read_baseline")
            result.mappings.return_value.first.return_value = existing_row
        elif "substrate_reduction_receipts" in sql:
            calls.append("fetch_new_rows")
            result.mappings.return_value.all.return_value = new_rows
        elif "INSERT INTO substrate_node_prediction_error_baseline" in sql:
            calls.append("persist_baseline")
            if persisted is not None:
                persisted.append({**(params or {}), "_versioned_sql": "definition_version" in sql})
        else:
            raise AssertionError(f"unexpected SQL in mock: {sql}")
        return result

    conn.execute.side_effect = execute_side_effect
    return fake_engine, conn, calls


def test_advance_node_prediction_error_baseline_cold_start_no_prior_row(monkeypatch) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    fake_engine, conn, calls = _mock_engine_for_baseline(
        existing_row=None,
        new_rows=[
            {"error": "0.05", "created_at": now},
            {"error": "0.9", "created_at": now},
        ],
    )
    store._engine = fake_engine

    baseline = store.advance_node_prediction_error_baseline(
        target_id="node:substrate.execution",
        reducer_key="execution_trajectory",
        alpha=0.2,
        min_variance=1e-5,
        fetch_limit=200,
    )

    assert baseline.observation_count == 2
    assert baseline.last_value == pytest.approx(0.9)
    assert calls == ["probe_version_column", "read_baseline", "fetch_new_rows", "persist_baseline"]


def test_advance_node_prediction_error_baseline_no_new_rows_skips_write(monkeypatch) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    existing = {
        "ewma": 0.1,
        "variance": 0.02,
        "observation_count": 10,
        "last_value": 0.15,
        "last_receipt_created_at": datetime(2026, 7, 30, 11, 0, tzinfo=timezone.utc),
    }
    fake_engine, conn, calls = _mock_engine_for_baseline(existing_row=existing, new_rows=[])
    store._engine = fake_engine

    baseline = store.advance_node_prediction_error_baseline(
        target_id="node:substrate.chat",
        reducer_key="chat_session",
        alpha=0.2,
        min_variance=1e-5,
        fetch_limit=200,
    )

    # A true no-op tick: the persisted state is returned unchanged and no
    # write is issued -- "nothing new landed this tick" must not be
    # misrepresented as "no real history."
    assert baseline.observation_count == 10
    assert baseline.last_value == pytest.approx(0.15)
    assert calls == ["probe_version_column", "read_baseline", "fetch_new_rows"]  # no persist


def test_advance_node_prediction_error_baseline_accumulates_on_existing_row(monkeypatch) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    existing = {
        "ewma": 0.05,
        "variance": 0.0,
        "observation_count": 2,
        "last_value": 0.0053,
        "last_receipt_created_at": datetime(2026, 7, 30, 11, 0, tzinfo=timezone.utc),
    }
    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    fake_engine, conn, calls = _mock_engine_for_baseline(
        existing_row=existing, new_rows=[{"error": "0.006", "created_at": now}]
    )
    store._engine = fake_engine

    baseline = store.advance_node_prediction_error_baseline(
        target_id="node:substrate.chat",
        reducer_key="chat_session",
        alpha=0.2,
        min_variance=1e-5,
        fetch_limit=200,
    )

    # Real cumulative count: 2 (persisted) + 1 (new this tick) = 3, not reset.
    assert baseline.observation_count == 3
    assert baseline.last_value == pytest.approx(0.006)
    assert "persist_baseline" in calls


def test_advance_node_prediction_error_baseline_skips_malformed_rows_but_advances_cursor(
    monkeypatch,
) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    fake_engine, conn, calls = _mock_engine_for_baseline(
        existing_row=None,
        new_rows=[
            {"error": None, "created_at": now},
            {"error": "not-a-number", "created_at": now},
            {"error": "0.3", "created_at": now},
        ],
    )
    store._engine = fake_engine

    baseline = store.advance_node_prediction_error_baseline(
        target_id="node:substrate.route",
        reducer_key="route_arbitration",
        alpha=0.2,
        min_variance=1e-5,
        fetch_limit=200,
    )

    # Only the one real parseable value is folded -- malformed rows are
    # skipped for the fold, but the batch (and thus the cursor) is still
    # consumed and persisted, per the method's own no-permanent-wedge contract.
    assert baseline.observation_count == 1
    assert baseline.last_value == pytest.approx(0.3)
    assert "persist_baseline" in calls


def test_advance_node_prediction_error_baseline_degrades_to_cold_start_on_error(
    monkeypatch,
) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    fake_engine.begin.side_effect = RuntimeError("boom")
    store._engine = fake_engine

    baseline = store.advance_node_prediction_error_baseline(
        target_id="node:substrate.bus_synaptic",
        reducer_key="bus_synaptic",
        alpha=0.2,
        min_variance=1e-5,
        fetch_limit=200,
    )

    assert baseline.observation_count == 0
    assert baseline.last_value is None


# -- definition-version reset (2026-09-25) --------------------------------------
# route_arbitration and chat_session moved to prediction-error definition v2
# (orion/schemas/prediction_error_definitions.py). A baseline built on v1 numbers
# must restart, and receipts the old producer wrote must not seed the new one.

_CURSOR = datetime(2026, 9, 25, 5, 0, tzinfo=timezone.utc)
_V1_ROUTE_ROW = {
    "ewma": 1.7e-17,
    "variance": 1.9e-20,
    "observation_count": 10254,
    "last_value": 0.0,
    "last_receipt_created_at": _CURSOR,
    "definition_version": None,  # column default 1 -> read back as "1"; None = unstamped
}


def _advance(store, target_id: str, reducer_key: str):
    return store.advance_node_prediction_error_baseline(
        target_id=target_id,
        reducer_key=reducer_key,
        alpha=0.2,
        min_variance=1e-5,
        fetch_limit=200,
    )


def test_baseline_resets_when_definition_version_moves() -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    later = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)
    persisted: list[dict] = []
    store._engine, _conn, calls = _mock_engine_for_baseline(
        existing_row={**_V1_ROUTE_ROW, "definition_version": "1"},
        new_rows=[
            {"error": "0.0", "definition_version": None, "created_at": later},  # old producer
            {"error": "0.5", "definition_version": "2", "created_at": later},
        ],
        version_column=True,
        persisted=persisted,
    )

    baseline = _advance(store, "node:substrate.route", "route_arbitration")

    # 10,254 v1 observations discarded; only the one v2 receipt folded.
    assert baseline.observation_count == 1
    assert baseline.last_value == pytest.approx(0.5)
    assert persisted[-1]["definition_version"] == 2
    assert persisted[-1]["_versioned_sql"] is True
    assert persisted[-1]["last_receipt_created_at"] == later  # cursor passes both rows


def test_baseline_reset_is_persisted_even_with_no_new_receipts() -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    persisted: list[dict] = []
    store._engine, _conn, calls = _mock_engine_for_baseline(
        existing_row={**_V1_ROUTE_ROW, "definition_version": "1"},
        new_rows=[],
        version_column=True,
        persisted=persisted,
    )

    baseline = _advance(store, "node:substrate.route", "route_arbitration")

    assert baseline.observation_count == 0
    assert baseline.last_value is None
    assert persisted and persisted[-1]["definition_version"] == 2
    assert persisted[-1]["last_receipt_created_at"] == _CURSOR  # cursor kept


def test_old_version_receipts_never_seed_a_fresh_baseline() -> None:
    """Attention runtime deployed before the substrate runtime: every receipt is still
    unstamped (v1). They advance the cursor but are not folded, so the target stays
    honestly cold instead of rebuilding a v2 baseline out of v1 numbers."""
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    later = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)
    persisted: list[dict] = []
    store._engine, _conn, _calls = _mock_engine_for_baseline(
        existing_row={**_V1_ROUTE_ROW, "definition_version": "2", "observation_count": 0,
                      "ewma": 0.0, "variance": 0.0, "last_value": None},
        new_rows=[{"error": "0.0003", "definition_version": None, "created_at": later}],
        version_column=True,
        persisted=persisted,
    )

    baseline = _advance(store, "node:substrate.route", "route_arbitration")

    assert baseline.observation_count == 0
    assert persisted[-1]["last_receipt_created_at"] == later


def test_unchanged_domain_is_not_reset() -> None:
    """Execution's definition did not change (v1): its unstamped row and receipts are
    both v1, so it accumulates exactly as before."""
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    later = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)
    store._engine, _conn, _calls = _mock_engine_for_baseline(
        existing_row={
            "ewma": 0.2, "variance": 0.1, "observation_count": 131654, "last_value": 0.8,
            "last_receipt_created_at": _CURSOR, "definition_version": "1",
        },
        new_rows=[{"error": "0.3", "definition_version": None, "created_at": later}],
        version_column=True,
    )

    baseline = _advance(store, "node:substrate.execution", "execution_trajectory")

    assert baseline.observation_count == 131655


def test_missing_version_column_keeps_legacy_behaviour() -> None:
    """Before the v2 migration is applied: no reset, no version filtering, and the
    upsert does not name the missing column."""
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    later = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)
    persisted: list[dict] = []
    store._engine, _conn, _calls = _mock_engine_for_baseline(
        existing_row={k: v for k, v in _V1_ROUTE_ROW.items() if k != "definition_version"},
        new_rows=[{"error": "0.5", "definition_version": "2", "created_at": later}],
        version_column=False,
        persisted=persisted,
    )

    baseline = _advance(store, "node:substrate.route", "route_arbitration")

    assert baseline.observation_count == 10255
    assert persisted[-1]["_versioned_sql"] is False


def test_version_column_probe_is_cached_once_present() -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    store._engine, _conn, calls = _mock_engine_for_baseline(
        existing_row=None, new_rows=[], version_column=True
    )
    _advance(store, "node:substrate.route", "route_arbitration")
    _advance(store, "node:substrate.chat", "chat_session")
    assert calls.count("probe_version_column") == 1


class TestReceiptLookupsStayIndexEligible:
    """The two reduction-receipt reads must filter the real `reducer_name` COLUMN.

    Found by code review of the 2026-08-19 field-tick index patch, not by the metrics sweep that
    found its siblings -- because this one never showed up as expensive. It is the same defect:
    filtering `receipt_json -> 'state_deltas' -> 0 ->> 'reducer_id'` returns identical rows to
    filtering the column (live: 0 disagreeing across 9,345 rows, the 4,138 NULLs coincide, and
    max(jsonb_array_length(state_deltas)) = 1 so the [0] subscript hides nothing), while making
    idx_substrate_reduction_receipts_reducer_name unusable.

    Measured at 3,248 of 3,254 buffers served from cache, so this costs CPU rather than disk and
    is NOT part of the I/O ceiling the rest of that patch addressed. The reason to gate it is
    growth: substrate_reduction_receipts has no pruner, and a seq scan that is cache-resident at
    9k rows is not cache-resident at 500k.
    """

    @staticmethod
    def _captured_sql(method: str, **kwargs) -> list[str]:
        seen: list[str] = []
        store = AttentionRuntimeStore.__new__(AttentionRuntimeStore)
        fake_engine = MagicMock()
        conn = MagicMock()
        fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
        fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

        def execute_side_effect(stmt, params=None):
            seen.append(" ".join(str(stmt).split()))
            result = MagicMock()
            result.mappings.return_value.all.return_value = []
            result.mappings.return_value.first.return_value = None
            return result

        conn.execute.side_effect = execute_side_effect
        store._engine = fake_engine
        getattr(store, method)(**kwargs)
        return seen

    def test_history_read_filters_the_column(self) -> None:
        sql = self._captured_sql("load_prediction_error_history", reducer_key="route", limit=5)
        receipts = [q for q in sql if "substrate_reduction_receipts" in q]
        assert receipts, "no receipts query issued"
        for q in receipts:
            assert "WHERE reducer_name = :reducer_id" in q
            assert "'reducer_id'" not in q, "JSON extraction cannot use the reducer_name index"

    def test_baseline_advance_filters_the_column(self) -> None:
        sql = self._captured_sql(
            "advance_node_prediction_error_baseline",
            target_id="node:substrate.route",
            reducer_key="route",
            alpha=0.1,
            min_variance=1e-6,
            fetch_limit=5,
        )
        receipts = [q for q in sql if "substrate_reduction_receipts" in q]
        assert receipts, "no receipts query issued"
        for q in receipts:
            assert "WHERE reducer_name = :reducer_id" in q
            assert "'reducer_id'" not in q


def test_version_skipped_receipts_are_logged(caplog) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    later = datetime(2026, 9, 25, 6, 0, tzinfo=timezone.utc)
    store._engine, _conn, _calls = _mock_engine_for_baseline(
        existing_row={**_V1_ROUTE_ROW, "definition_version": "2"},
        new_rows=[{"error": "0.0003", "definition_version": None, "created_at": later}],
        version_column=True,
    )
    with caplog.at_level("WARNING"):
        _advance(store, "node:substrate.route", "route_arbitration")
    assert "node_prediction_error_baseline_version_skipped" in caplog.text
    assert "skipped=1" in caplog.text


def test_column_probe_is_scoped_to_the_current_schema() -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    seen: list[str] = []
    conn = MagicMock()
    conn.execute.side_effect = lambda stmt, params=None: seen.append(str(stmt)) or MagicMock()
    store._has_definition_version_column(conn)
    assert "table_schema = current_schema()" in seen[0]


def test_advance_failure_forces_a_column_re_probe() -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    store._definition_version_column = True
    fake_engine = MagicMock()
    fake_engine.begin.side_effect = RuntimeError("column does not exist")
    store._engine = fake_engine
    _advance(store, "node:substrate.route", "route_arbitration")
    assert store._definition_version_column is None
