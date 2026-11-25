"""Replay of goal proposals lost to bus_fallback_log (stale route map, 2026-11-25).

The script must store them the way the live handler does (same coercion, same fold), fold an old
run into its own episodes even when a newer episode already exists, and do nothing on a second run.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.field_goal import FieldGoalProvenanceV1  # noqa: E402

from app.field_goal_episode import ARTIFACT_DEDUP_WINDOW, fold_goal  # noqa: E402
from app.models.field_goal_episode import FieldGoalEpisodeArtifactSQL, FieldGoalEpisodeSQL  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "replay_field_goals", SERVICE_ROOT / "scripts/replay_field_goals_from_bus_fallback.py")
replay_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_mod)

KIND = "memory.field_goals.proposed.v1"
T0 = datetime(2026, 11, 25, 8, 29, tzinfo=timezone.utc)


def _payload(target: str, ts: datetime, **kw) -> dict:
    goal = FieldGoalProvenanceV1(
        subject="attention", model_layer="field_attention", entity_id=target, kind=KIND,
        field_target_id=target, target_kind="node", salience_score=0.9,
        source_field_tick_id="tick-1", source_attention_frame_id="frame-1", priority=0.9,
        ts=ts, provenance={"intake_channel": "internal.attention_runtime"}, **kw)
    return goal.model_dump(mode="json")


def _session():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    FieldGoalEpisodeSQL.__table__.create(bind=engine)
    FieldGoalEpisodeArtifactSQL.__table__.create(bind=engine)
    return sessionmaker(bind=engine)()


def _rows(payloads: list) -> list[dict]:
    return [{"id": i + 1, "payload": json.dumps(p) if i % 2 else p} for i, p in enumerate(payloads)]


def test_coerce_matches_the_live_handler() -> None:
    p = _payload("world:chat", T0)
    assert replay_mod.coerce(p).artifact_id == p["artifact_id"]
    assert replay_mod.coerce(json.dumps(p)).field_target_id == "world:chat"
    with pytest.raises(ValueError):
        replay_mod.coerce({k: v for k, v in p.items() if k != "artifact_id"})
    with pytest.raises(Exception):
        replay_mod.coerce({"artifact_id": "a-1", "salience_score": "high"})


def test_plan_orders_by_ts_and_refuses_bad_and_repeated_rows() -> None:
    late, early = _payload("world:chat", T0 + timedelta(minutes=5)), _payload("world:chat", T0)
    goals, refused = replay_mod.plan(_rows([late, early, {"no": "artifact"}, early]))
    assert [g["goal"].artifact_id for g in goals] == [early["artifact_id"], late["artifact_id"]]
    assert len(refused) == 2


def test_replay_folds_an_old_run_into_its_own_episodes_next_to_a_newer_one() -> None:
    sess = _session()
    # A live episode opened later today, after the lost window.
    fold_goal(sess, replay_mod.coerce(_payload("world:chat", T0 + timedelta(hours=9))), envelope_id="e")
    sess.commit()
    lost = ([_payload("node:a", T0 + timedelta(seconds=2 * i)) for i in range(30)]
            + [_payload("world:chat", T0 + timedelta(minutes=20, seconds=2 * i)) for i in range(20)]
            + [_payload("node:a", T0 + timedelta(minutes=50, seconds=2 * i)) for i in range(10)])
    goals, refused = replay_mod.plan(_rows(lost))
    assert not refused
    counts = replay_mod.replay(sess, goals, fold=fold_goal)
    sess.commit()
    assert counts == {"opened": 3, "folded": 57}
    eps = sess.query(FieldGoalEpisodeSQL).order_by(FieldGoalEpisodeSQL.first_ts).all()
    assert [(e.field_target_id, e.repeat_count) for e in eps] == [
        ("node:a", 30), ("world:chat", 20), ("node:a", 10), ("world:chat", 1)]
    # The live fold (replay=False) would have opened one episode per goal next to the newest row.

    # A second run stores nothing.
    again = replay_mod.replay(sess, replay_mod.plan(_rows(lost))[0], fold=fold_goal)
    sess.commit()
    assert again == {"duplicate": 60}
    assert sum(e.repeat_count for e in sess.query(FieldGoalEpisodeSQL)) == 61


def test_live_fold_without_replay_would_split_the_old_run() -> None:
    """Why replay=True exists: the live rule compares against the newest episode of any time."""
    sess = _session()
    fold_goal(sess, replay_mod.coerce(_payload("world:chat", T0 + timedelta(hours=9))), envelope_id="e")
    for i in range(3):
        fold_goal(sess, replay_mod.coerce(_payload("node:a", T0 + timedelta(seconds=2 * i))), envelope_id=None)
    assert sess.query(FieldGoalEpisodeSQL).count() == 4


def test_refuses_goals_past_the_dedup_window() -> None:
    goals, _ = replay_mod.plan(_rows([_payload("node:a", T0)]))
    assert replay_mod.too_old(goals, T0 + ARTIFACT_DEDUP_WINDOW, ARTIFACT_DEDUP_WINDOW)
    assert not replay_mod.too_old(goals, T0 + timedelta(hours=10), ARTIFACT_DEDUP_WINDOW)


def test_coerce_unwraps_a_nested_envelope_like_the_live_handler() -> None:
    inner = _payload("world:chat", T0)
    wrapped = {"artifact_id": inner["artifact_id"], "kind": KIND, "source": {"name": "x"},
               "correlation_id": "c", "payload": inner}
    assert replay_mod.coerce(wrapped).field_target_id == "world:chat"


def test_refuses_when_a_replayed_goal_is_newer_than_the_newest_live_episode() -> None:
    goals, _ = replay_mod.plan(_rows([_payload("node:a", T0 + timedelta(hours=2))]))
    assert replay_mod.overtakes_live(goals, [{"first_ts": T0 + timedelta(hours=1)}])
    assert not replay_mod.overtakes_live(goals, [{"first_ts": T0 + timedelta(hours=9)}])
    assert not replay_mod.overtakes_live(goals, [])


def test_main_dry_run_rolls_back_apply_writes_and_a_second_apply_is_a_no_op(tmp_path) -> None:
    from sqlalchemy import text

    db = tmp_path / "replay.db"
    dsn = f"sqlite:///{db}"
    engine = create_engine(dsn)
    FieldGoalEpisodeSQL.__table__.create(bind=engine)
    FieldGoalEpisodeArtifactSQL.__table__.create(bind=engine)
    start = datetime.now(timezone.utc) - timedelta(hours=10)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE bus_fallback_log (id INTEGER PRIMARY KEY, kind TEXT, correlation_id TEXT, "
                          "payload TEXT, created_at_ts TIMESTAMP, error TEXT)"))
        lost = ([_payload("node:a", start + timedelta(seconds=2 * i)) for i in range(5)]
                + [_payload("world:chat", start + timedelta(minutes=30, seconds=2 * i)) for i in range(4)])
        for i, p in enumerate(lost, 1):
            conn.execute(text("INSERT INTO bus_fallback_log VALUES (:i, :k, '', :p, :t, :e)"),
                         {"i": i, "k": KIND, "p": json.dumps(p), "t": start, "e": "Unknown kind"})
        # Another kind and another error are never touched.
        conn.execute(text("INSERT INTO bus_fallback_log VALUES (99, 'other.kind', '', '{}', :t, 'Unknown kind')"),
                     {"t": start})
    # A live episode opened after the lost window.
    sess = sessionmaker(bind=engine)()
    fold_goal(sess, replay_mod.coerce(_payload("world:chat", datetime.now(timezone.utc))), envelope_id="e")
    sess.commit()
    sess.close()

    def count() -> int:
        with engine.connect() as conn:
            return conn.execute(text("SELECT count(*) FROM field_goal_episode")).scalar_one()

    job = tmp_path / "job"
    assert replay_mod.main(["--dsn", dsn, "--job-dir", str(job)]) == 0
    assert count() == 1
    assert replay_mod.main(["--dsn", dsn, "--job-dir", str(job), "--apply"]) == 0
    assert count() == 3
    reports = sorted(job.glob("report_apply_*.md"))
    assert reports and "opened 2, folded 7, duplicate (already stored) 0" in reports[-1].read_text()
    assert list(job.glob("before_after_apply_*.csv")) and list(job.glob("report_dryrun_*.md"))
    assert replay_mod.main(["--dsn", dsn, "--job-dir", str(job), "--apply"]) == 0
    assert count() == 3
    assert "duplicate (already stored) 9" in sorted(job.glob("report_apply_*.md"))[-1].read_text()
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM bus_fallback_log")).scalar_one() == 10


def test_a_late_older_goal_leaves_last_salience_to_the_newest_goal() -> None:
    sess = _session()
    newer = _payload("node:a", T0 + timedelta(seconds=10)) | {"salience_score": 0.7}
    older = _payload("node:a", T0) | {"salience_score": 0.95}
    fold_goal(sess, replay_mod.coerce(newer), envelope_id=None)
    fold_goal(sess, replay_mod.coerce(older), envelope_id=None)
    ep = sess.query(FieldGoalEpisodeSQL).one()
    assert (ep.repeat_count, ep.last_salience, ep.max_salience) == (2, 0.7, 0.95)
