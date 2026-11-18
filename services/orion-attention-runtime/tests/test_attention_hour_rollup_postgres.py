"""The hourly rollup, end to end on a disposable Postgres: the store writes it with each frame, and
`dig attention`'s own SQL reads it back. Skips without FOCUS_RUN_TEST_POSTGRES_URI (CI provides one)."""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text

from app.store import AttentionRuntimeStore
from orion.dig import attention as A
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1

REPO = Path(__file__).resolve().parents[3]
T0 = datetime(2026, 11, 17, 7, 0, tzinfo=timezone.utc)


@pytest.fixture
def store():
    uri = os.environ.get("FOCUS_RUN_TEST_POSTGRES_URI")
    if not uri:
        pytest.skip("FOCUS_RUN_TEST_POSTGRES_URI must name a disposable test DB")
    engine = create_engine(uri)
    schema = "rollup_test_" + uuid4().hex
    with engine.begin() as c:
        c.execute(text(f'CREATE SCHEMA "{schema}"'))
    scoped = create_engine(uri, connect_args={"options": f"-c search_path={schema}"})
    with scoped.begin() as c:
        for name in ("attention_frame_v1", "attention_hour_rollup_v1"):
            ddl = (REPO / f"services/orion-sql-db/manual_migration_{name}.sql").read_text()
            c.exec_driver_sql(ddl.replace("BEGIN;", "").replace("COMMIT;", ""))
    instance = AttentionRuntimeStore.__new__(AttentionRuntimeStore)
    instance._engine = scoped
    yield instance
    scoped.dispose()
    engine.dispose()


def target(tid, reason, salience=0.5):
    return FieldAttentionTargetV1(target_id=tid, target_kind="node", salience_score=salience, pressure_score=0.0,
                                  novelty_score=0.0, urgency_score=0.0, confidence_score=1.0, reasons=[reason])


def frame(i, dominant, suppressed):
    at = T0 + timedelta(seconds=i * 1800)  # two frames an hour: hours 07 and 08
    return FieldAttentionFrameV1(frame_id=f"f-{i}", generated_at=at, source_field_tick_id=f"t-{i}",
                                 source_field_generated_at=at, overall_salience=0.5 if dominant else 0.0,
                                 dominant_targets=dominant, suppressed_targets=suppressed)


ATT = "world-first: attended -- world busy for itself (high); x"
QUIET = "world-first: not attended -- world quiet for itself: not busy; x"


def read_back(store, end):
    w = A.parse_window("3h", end)
    with store._engine.connect() as c:
        def q(sql):
            body = sql.split(";\n", 1)[1]  # drop SET LOCAL (outside a transaction block it warns)
            return [{k: str(v) for k, v in r.items()} for r in c.execute(text(body)).mappings()]
        first = A.parse_first(q(A.coverage_sql(w)))
        w = w.covered(first)
        return A.parse(w, q(A.frames_sql(w)), [])


def test_frames_land_in_the_rollup_once_and_dig_reads_the_rates(store):
    frames = [
        frame(0, [target("world:chat", ATT), target("node:p", ATT)], [target("world:reading", QUIET)]),
        frame(1, [], [target("world:chat", QUIET), target("world:reading", QUIET)]),
        frame(2, [target("node:p", ATT)], [target("world:chat", QUIET)]),
        frame(3, [], [target("world:chat", QUIET)]),
    ]
    # the deploy hour (06): the rollup's first hour, only partly counted, so dig skips it
    assert store.save_attention_frame(frame(-1, [target("node:p", ATT)], []))
    for f in frames:
        assert store.save_attention_frame(f)
    assert not store.save_attention_frame(frames[0])  # duplicate frame: no second count
    r = read_back(store, "2026-11-17T09:00Z")
    c = r.cur
    assert (c.frames, c.rest, c.contested) == (4, 2, 1)
    assert (c.present["world:chat"], c.won["world:chat"], c.present_contested["world:chat"]) == (1, 1, 1)
    assert (c.present["node:p"], c.won["node:p"], c.won_contested["node:p"]) == (2, 1, 0)
    assert c.causes["world:chat"] == {"quiet for itself": 3}
    with store._engine.connect() as conn:
        hours = conn.execute(text("SELECT DISTINCT hour FROM substrate_attention_hour_rollup ORDER BY 1")).scalars().all()
    assert [h.astimezone(timezone.utc).hour for h in hours] == [6, 7, 8]
    assert r.start.startswith("2026-11-17T07:00")  # read from the first whole hour


def test_a_rollup_failure_never_stops_the_frame(store, caplog):
    with store._engine.begin() as c:
        c.exec_driver_sql("ALTER TABLE substrate_attention_hour_rollup ADD CONSTRAINT inject_failure CHECK (a < 0)")
    assert store.save_attention_frame(frame(0, [], []))
    with store._engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM substrate_attention_frames")).scalar() == 1
        assert c.execute(text("SELECT count(*) FROM substrate_attention_hour_rollup")).scalar() == 0
    assert "attention_hour_rollup_failed" in caplog.text
