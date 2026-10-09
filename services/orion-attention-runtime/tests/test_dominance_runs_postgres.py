"""Real transaction tests, against an explicitly supplied disposable Postgres only."""
import os
from pathlib import Path
from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text

from app.store import AttentionRuntimeStore
from app.worker import AttentionRuntimeWorker
from orion.attention.field_attention.goal_provenance import DominanceStreak
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from test_dominance_runs import START

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def store():
    uri = os.environ.get("FOCUS_RUN_TEST_POSTGRES_URI")
    if not uri:
        pytest.skip("FOCUS_RUN_TEST_POSTGRES_URI must name a disposable test DB")
    engine = create_engine(uri)
    schema = "focus_test_" + uuid4().hex
    with engine.begin() as c:
        c.execute(text(f'CREATE SCHEMA "{schema}"'))
    scoped = create_engine(uri, connect_args={"options": f"-c search_path={schema}"})
    with scoped.begin() as c:
        for name in ("attention_frame_v1", "goal_provenance_streak_v1", "field_dominance_run_v1"):
            ddl = (REPO / f"services/orion-sql-db/manual_migration_{name}.sql").read_text()
            c.exec_driver_sql(ddl.replace("BEGIN;", "").replace("COMMIT;", ""))
    instance = AttentionRuntimeStore.__new__(AttentionRuntimeStore)
    instance._engine = scoped
    yield instance
    scoped.dispose()
    # Deliberately retain isolated schemas in the disposable DB for inspection.
    engine.dispose()


def frame(i):
    return FieldAttentionFrameV1(frame_id=f"frame-{i}", source_field_tick_id=f"tick-{i}",
        source_field_generated_at=START + timedelta(seconds=i * 2),
        generated_at=START + timedelta(seconds=i * 2), overall_salience=0.5)


def save(store, i, target, count):
    return store.save_attention_frame(frame(i), streak=DominanceStreak(target, count),
                                      target_kind="node" if target else None, min_streak=3)


def rows(store):
    with store._engine.connect() as c:
        return [dict(r) for r in c.execute(text("SELECT * FROM field_dominance_run ORDER BY started_at")).mappings()]


def test_restart_duplicate_and_no_winner_land_exact_rows(store):
    assert save(store, 0, "A", 1)
    assert save(store, 1, "A", 2)
    assert rows(store) == []
    assert not save(store, 1, "A", 2)
    restarted = AttentionRuntimeStore.__new__(AttentionRuntimeStore)
    restarted._engine = store._engine
    assert restarted.load_node_dominance_streak() == DominanceStreak("A", 2)
    assert save(restarted, 2, "B", 1)
    assert save(restarted, 3, None, 0)
    landed = rows(store)
    assert [(r["target_id"], r["tick_count"]) for r in landed] == [("A", 2), ("B", 1)]
    assert landed[0]["ended_at"] == frame(2).generated_at
    assert landed[0]["last_source_attention_frame_id"] == "frame-1"
    assert not landed[0]["left_censored"]


def test_failed_recording_preserves_frames_goals_and_drops_unknown_gap(store, caplog):
    save(store, 0, "A", 1)
    with store._engine.begin() as c:
        c.exec_driver_sql("ALTER TABLE field_dominance_run ADD CONSTRAINT inject_failure CHECK (tick_count < 0)")
    # A transition-record failure cannot stop the frame or decision debounce.
    assert save(store, 1, "B", 1)
    assert store.load_attention_frame_for_field_tick("tick-1") is not None
    assert store.load_node_dominance_streak() == DominanceStreak("B", 1)
    assert rows(store) == []
    assert "attention_focus_record_failed" in caplog.text
    with store._engine.begin() as c:
        c.exec_driver_sql("ALTER TABLE field_dominance_run DROP CONSTRAINT inject_failure")
    # Recovery across a process restart detects the missing recorder tick from
    # saved frames, without an in-memory flag. No fabricated A->B history.
    restarted = AttentionRuntimeStore.__new__(AttentionRuntimeStore)
    restarted._engine = store._engine
    save(restarted, 2, "B", 2)
    save(restarted, 3, "C", 1)
    assert "attention_focus_observation_gap" in caplog.text
    row = rows(store)[0]
    assert (row["target_id"], row["tick_count"], row["left_censored"]) == ("B", 1, True)


def test_missing_recording_migration_keeps_worker_goal_emission(store, monkeypatch):
    from test_goal_provenance_producer import _make_worker, _target, _frame
    worker = _make_worker(monkeypatch)
    worker._store = store
    with store._engine.begin() as c:
        c.exec_driver_sql("ALTER TABLE substrate_goal_provenance_streak RENAME COLUMN run_state TO unavailable_run_state")
    for i in range(3):
        current = _frame([_target("node:substrate.biometrics", 0.8)]).model_copy(update={
            "frame_id": f"worker-{i}", "source_field_tick_id": f"worker-tick-{i}",
            "generated_at": START + timedelta(seconds=i * 2)})
        goal = worker._maybe_build_goal(current)
    assert goal is not None and goal.field_target_id == "node:substrate.biometrics"
    assert store.load_attention_frame_for_field_tick("worker-tick-2") is not None
    assert store.load_node_dominance_streak().count == 3


def test_install_mid_streak_does_not_invent_start_or_count(store):
    store.save_node_dominance_streak(DominanceStreak("A", 200))
    save(store, 0, "A", 201)
    save(store, 1, "A", 202)
    save(store, 2, "B", 1)
    row = rows(store)[0]
    assert row["left_censored"]
    assert row["tick_count"] == 2
    assert row["started_at"] == START


def test_retirement_migration_removes_table_without_touching_runs(store):
    save(store, 0, "A", 1)
    save(store, 1, "B", 1)
    with store._engine.begin() as c:
        c.exec_driver_sql("CREATE TABLE goal_provenance_streak_ticks (id INTEGER)")
        ddl = (REPO / "services/orion-sql-db/manual_migration_retire_streak_tick_v1.sql").read_text()
        c.exec_driver_sql(ddl.replace("BEGIN;", "").replace("COMMIT;", ""))
        assert c.execute(text("SELECT to_regclass('goal_provenance_streak_ticks')")).scalar() is None
    assert len(rows(store)) == 1
