"""Goal episodes against real Postgres (issue #3924).

Opt-in like the other DB tests: set ORION_SQL_WRITER_TEST_DATABASE_URI to a scratch database.
Tables come from the real migration file inside a throwaway schema (search_path), so a
misconfigured URI never touches the real field_goal_episode tables.

Proves what SQLite cannot: the migration runs on Postgres, and concurrent handler calls
(separate connections) neither lose a count nor open two episodes nor double count a
re-delivered artifact, because fold_goal serializes on one advisory lock.
"""
from __future__ import annotations

import os
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.field_goal import FieldGoalProvenanceV1  # noqa: E402

from app.field_goal_episode import fold_goal  # noqa: E402
from app.models.field_goal_episode import FieldGoalEpisodeArtifactSQL, FieldGoalEpisodeSQL  # noqa: E402

pytestmark = pytest.mark.integration
MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_field_goal_episode_v1.sql"
T0 = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _goal(target: str, ts: datetime) -> FieldGoalProvenanceV1:
    return FieldGoalProvenanceV1(
        subject="attention", model_layer="field_attention", entity_id=target,
        kind="memory.field_goals.proposed.v1", field_target_id=target, target_kind="node",
        salience_score=0.9, source_field_tick_id="t", source_attention_frame_id="f", priority=0.9,
        ts=ts, provenance={"intake_channel": "internal.attention_runtime"},
    )


@pytest.fixture
def factory():
    uri = os.environ.get("ORION_SQL_WRITER_TEST_DATABASE_URI")
    if not uri:
        pytest.skip("ORION_SQL_WRITER_TEST_DATABASE_URI not set")
    schema = f"goal_ep_{uuid.uuid4().hex[:10]}"
    admin = create_engine(uri)
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(uri, pool_size=20, connect_args={"options": f"-csearch_path={schema}"})
    with engine.begin() as conn:
        conn.exec_driver_sql(MIGRATION.read_text())
        conn.exec_driver_sql(MIGRATION.read_text())  # re-runnable (IF NOT EXISTS)
    try:
        yield sessionmaker(bind=engine)
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def _fold_all(factory, goals, threads: int = 16) -> list[str]:
    outcomes: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(threads)
    chunks = [goals[i::threads] for i in range(threads)]

    def run(chunk):
        barrier.wait()
        for g in chunk:
            sess = factory()
            try:
                out = fold_goal(sess, g, envelope_id=None)
                sess.commit()
            finally:
                sess.close()
            with lock:
                outcomes.append(out)

    ts = [threading.Thread(target=run, args=(c,)) for c in chunks]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return outcomes


def test_concurrent_repeats_fold_into_one_episode_with_exact_count(factory) -> None:
    goals = [_goal("world:chat", T0 + timedelta(seconds=i)) for i in range(160)]
    dupes = goals[:40]  # re-delivered concurrently with the originals
    outcomes = _fold_all(factory, goals + dupes)
    sess = factory()
    try:
        (ep,) = sess.query(FieldGoalEpisodeSQL).all()
        assert ep.repeat_count == 160
        assert outcomes.count("duplicate") == 40 and outcomes.count("opened") == 1
        assert sess.query(FieldGoalEpisodeArtifactSQL).count() == 160
    finally:
        sess.close()


def test_postgres_a_b_a_and_gap(factory) -> None:
    sess = factory()
    try:
        for target, sec in (("a", 0), ("a", 2), ("b", 4), ("a", 6), ("a", 6 + 601)):
            fold_goal(sess, _goal(target, T0 + timedelta(seconds=sec)), envelope_id="e")
            sess.commit()
        eps = sess.query(FieldGoalEpisodeSQL).order_by(FieldGoalEpisodeSQL.first_ts).all()
        assert [(e.field_target_id, e.repeat_count) for e in eps] == [("a", 2), ("b", 1), ("a", 1), ("a", 1)]
        assert eps[0].first_ts.tzinfo is not None
    finally:
        sess.close()
