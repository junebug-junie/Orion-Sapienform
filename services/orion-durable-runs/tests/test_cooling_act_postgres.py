"""The cooling act's store on REAL Postgres (ORION_ADMISSION_TEST_DSN; skipped without it).

Proves the SQL the loop runs: the lazy ledger DDL, the precommit insert (idempotent), admission inputs, the
pending-settlement join to the pool's gpu_pool_orion_shed row (and its absence), the settlement merge, and
that the sql-writer scorer's own reader (WorldSettlementStore.load_world_episodes_due) finds the episode.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

DSN = os.environ.get("ORION_ADMISSION_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="ORION_ADMISSION_TEST_DSN not set")


@pytest.fixture()
def store():
    from sqlalchemy import create_engine, text

    from app.cooling_act import CoolingActStore, sqlalchemy_url

    schema = f"cooling_{uuid.uuid4().hex[:8]}"
    admin = create_engine(sqlalchemy_url(DSN))
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(sqlalchemy_url(DSN), connect_args={"options": f"-csearch_path={schema}"})
    try:
        yield CoolingActStore(engine), engine, schema
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


def _row(eid, arm, now, settlement=None):
    from orion.autonomy import shed_act

    return shed_act.episode_row(episode_id=eid, arm=arm, now=now, eligibility={"eligible": True, "want": {"want_id": "w"}},
                                effect=shed_act.expected_effect(None), decision_id="dec-1", ttl_sec=900.0,
                                settlement=settlement)


def test_store_round_trip_on_real_postgres(store):
    from sqlalchemy import text

    from orion.autonomy.shed_settlement import SHED_PENDING
    from orion.autonomy.world_settlement_store import WorldSettlementStore

    st, engine, schema = store
    now = datetime.now(timezone.utc)
    assert st.admission_inputs(now=now) == ([], [])            # no ledger yet: nothing in flight
    assert st.load_pending_settlements() == []
    pending = {"state": SHED_PENDING, "decided_at": now.isoformat(), "ttl_sec": 900.0}
    assert st.insert_episode(_row("cooling_act:a", "treated", now, pending)) is True
    assert st.insert_episode(_row("cooling_act:a", "treated", now, pending)) is False   # idempotent
    assert st.episode_arm("cooling_act:a") == "treated"
    in_flight, treated = st.admission_inputs(now=now + timedelta(seconds=1))
    assert in_flight == ["cooling_act:a"] and treated[0]["episode_id"] == "cooling_act:a"
    # no pool ledger: the episode still comes back to be orphaned later
    (p,) = st.load_pending_settlements()
    assert p["pool_row"] is None and p["settlement"]["state"] == SHED_PENDING
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE gpu_pool_orion_shed (shed_id text, dispatch_id text, state text, refusal text, "
                          "started_at timestamptz, ended_at timestamptz, drained_at timestamptz, grants_withheld int, "
                          "delayed_grant_sec float, background_live_at_start int, valid_until timestamptz, ttl_sec float)"))
        conn.execute(text("INSERT INTO gpu_pool_orion_shed VALUES ('oshed_1', 'cooling_act:a', 'expired', NULL, :t, :e, "
                          "NULL, 3, 60.0, 5, :e, 900)"), {"t": now, "e": now + timedelta(seconds=900)})
    (p,) = st.load_pending_settlements()
    assert p["pool_row"]["state"] == "expired" and p["pool_row"]["shed_id"] == "oshed_1"
    st.record_settlement(episode_id="cooling_act:a", shed_id="oshed_1", state="expired",
                         settlement={"state": "settled", "terminal": "expired"})
    assert st.load_pending_settlements() == []                 # settled: no longer pending
    due = WorldSettlementStore(engine).load_world_episodes_due(now=now + timedelta(seconds=1300))
    assert [d["episode_id"] for d in due] == ["cooling_act:a"]
    assert due[0]["settlement_state"] == "expired" and due[0]["settlement"]["ttl_sec"] == 900.0
