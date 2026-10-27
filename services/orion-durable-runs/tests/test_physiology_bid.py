"""Body economy Patch C1: POST /physiology/bid (SHADOW) -- the market's reads, the decision, the
publish, and the never-fail-open rule. Fakes for Redis / Postgres / the bus; the history SQL also
runs against an isolated Postgres schema when ORION_ADMISSION_TEST_DSN is set (CI provides one)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from app.physiology_market import ACTS_SQL, FUNDED_SQL, USES_SQL, Market, read_funded, read_history  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402
from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R, PhysiologyStateV1  # noqa: E402
from orion.schemas.physiology_want import WantDecisionV1, WantV1  # noqa: E402

NOW = datetime.now(timezone.utc)
FAMS, _ = load_families(REPO_ROOT / "config/physiology/families.yaml")
DSN = os.environ.get("ORION_ADMISSION_TEST_DSN")


def _state(at=None) -> PhysiologyStateV1:
    at = at or NOW
    return PhysiologyStateV1(tick_id=f"physiology:{at.isoformat()}", generated_at=at, constraints=[
        R(constraint="lane:agent", unit="slot_s", level=600.0, budget=3600.0, budget_kind="capacity",
          detail={"slots": 1, "window_sec": 3600.0, "deferrable_queued": 2, "deferrable_waiting_sec": 1800.0},
          price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""}),
        R(constraint="attention:juniper", unit="messages", level=1.0, budget=4.0, budget_kind="daily_cap",
          price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 86400.0, "why": ""}),
    ])


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    async def fetchall(self):
        return self.rows


class _Conn:
    def __init__(self, pool):
        self.pool = pool

    async def execute(self, sql, params=None):
        self.pool.calls += 1
        if self.pool.fail:
            raise RuntimeError("db down")
        if "want_ledger" in sql:
            if self.pool.ledger_fail:
                raise RuntimeError("want_ledger missing")
            return _Cursor(self.pool.funded)
        return _Cursor(self.pool.acts if "grammar_acts" in sql else self.pool.uses)


class _Pool:
    def __init__(self, acts=(), uses=(), fail=False, funded=(), ledger_fail=False):
        self.acts, self.uses, self.fail, self.calls = list(acts), list(uses), fail, 0
        self.funded, self.ledger_fail = list(funded), ledger_fail

    def connection(self):
        pool = self

        class _CM:
            async def __aenter__(self):
                return _Conn(pool)

            async def __aexit__(self, *a):
                return False

        return _CM()


class _Redis:
    def __init__(self, raw):
        self.raw = raw

    async def get(self, key):
        assert key == "orion:physiology:latest"
        return self.raw


def _acts():
    out = []
    for i in range(8):
        closed = NOW - timedelta(hours=2 + 3 * i)
        out.append({"act_id": f"a{i}", "actor": "durable.self_inquiry", "opened_at": closed - timedelta(hours=1),
                    "closed_at": closed, "outcome_value": 1.0, "outcome_source": "producer", "signal_id": None,
                    "cost_json": json.dumps({"seconds": 3600.0, "gpu_slot_s_by_lane": {"agent": 1200.0}})})
    return out


def _market(raw, pool, published):
    async def publish(decision):
        published.append(decision)

    return Market(families=FAMS, families_error=None, pool_getter=lambda: pool, redis_getter=lambda: _Redis(raw),
                  publish=publish)


def _want(i=0, family="durable.self_inquiry"):
    return WantV1(want_id=f"w{i}", family=family, producer="unit-producer", verb="ask", tier=2)


def test_bid_prices_against_the_latest_state_publishes_and_caches_history():
    published: list = []
    pool = _Pool(acts=_acts())
    m = _market(_state().model_dump_json(), pool, published)
    d = asyncio.run(m.bid(_want(), NOW))
    assert d.shadow is True and d.outcome == "funded" and d.lane == "agent"
    assert d.state_tick_id == _state().tick_id and d.prices_seen["lane:agent"] == 0.0
    assert published == [d]
    calls = pool.calls
    asyncio.run(m.bid(_want(1), NOW + timedelta(seconds=10)))
    assert pool.calls == calls                  # history cached inside the TTL
    assert m.health(NOW)["reservations"] == 2 and m.release("w0") == 1


def test_the_weekly_floor_is_seeded_from_want_ledger_and_a_ledger_failure_does_not_fail_the_bid():
    """Patch C3a: the floor counts funded wants from want_ledger (rolling 7 days), re-read with the
    history; the context keeps tier1_recent_use (C1 dropped it rebuilding the context by hand)."""
    funded = [{"decision_id": f"x{i}@t", "family": "durable.curiosity_investigate", "tier": 2,
               "decided_at": NOW - timedelta(days=1)} for i in range(99)]
    funded.append({"decision_id": "y@t", "family": "durable.self_inquiry", "tier": 2, "decided_at": NOW - timedelta(days=2)})
    m = _market(_state().model_dump_json(), _Pool(acts=_acts(), funded=funded), [])
    d = asyncio.run(m.bid(_want(), NOW))
    assert m.book.funded_seeded is True and m.health(NOW)["funded_seeded"] is True
    # 1 of 100 before this bid; the bid itself was funded and counted once.
    assert m.book.funded_share("durable.self_inquiry", 2, NOW) == pytest.approx(2 / 101 if d.outcome == "funded" else 1 / 100)
    ctx = asyncio.run(m.context(_state(), NOW))
    assert ctx.stale_after_sec == m._stale and isinstance(ctx.tier1_recent_use, dict)
    bad = _market(_state().model_dump_json(), _Pool(acts=_acts(), ledger_fail=True), [])
    d2 = asyncio.run(bad.bid(_want(), NOW))
    assert d2.reason != "physiology_unavailable" and bad.book.funded_seeded is False


def test_mark_opened_and_release_follow_the_act():
    m = _market(_state().model_dump_json(), _Pool(acts=_acts()), [])
    d = asyncio.run(m.bid(_want(), NOW))
    assert d.outcome == "funded"
    assert m.mark_opened("w0", NOW + timedelta(seconds=30)) == 1
    assert m.book.snapshot(NOW + timedelta(seconds=31))[0]["opened"] is True
    assert m.release("w0") == 1 and m.book.snapshot(NOW + timedelta(seconds=31)) == []


def test_never_funded_on_a_missing_market():
    for raw, pool in ((None, _Pool(acts=_acts())), (b"{bad", _Pool(acts=_acts())),
                      (_state().model_dump_json(), _Pool(fail=True)),
                      (_state().model_dump_json(), _Pool(acts=[])),   # an empty act read is unknown
                      (_state(NOW - timedelta(seconds=900)).model_dump_json(), _Pool(acts=_acts()))):
        d = asyncio.run(_market(raw, pool, []).bid(_want(), NOW))
        assert d.outcome == "deferred" and d.reason == "physiology_unavailable", (raw, pool.fail)
    m = Market(families=None, families_error="missing:families.yaml", pool_getter=lambda: _Pool(),
               redis_getter=lambda: _Redis(_state().model_dump_json()), publish=lambda d: asyncio.sleep(0))
    d = asyncio.run(m.bid(_want(), NOW))
    assert d.reason == "physiology_unavailable" and "families.yaml" in d.why


def test_a_publish_failure_still_answers_the_caller():
    async def boom(decision):
        raise RuntimeError("bus down")

    m = Market(families=FAMS, families_error=None, pool_getter=lambda: _Pool(acts=_acts()),
               redis_getter=lambda: _Redis(_state().model_dump_json()), publish=boom)
    d = asyncio.run(m.bid(_want(), NOW))
    assert d.outcome == "funded" and m.counts["publish_failed"] == 1


def test_route_validates_answers_and_503s_when_disabled(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from fastapi import HTTPException

    from app import main as app_main

    published: list = []
    old = app_main.market
    try:
        app_main.market = None
        with pytest.raises(HTTPException) as e:
            asyncio.run(app_main.physiology_bid(_want().model_dump(mode="json")))
        assert e.value.status_code == 503
        app_main.market = _market(_state().model_dump_json(), _Pool(acts=_acts()), published)
        out = asyncio.run(app_main.physiology_bid(_want().model_dump(mode="json")))
        assert WantDecisionV1.model_validate(out).shadow is True and len(published) == 1
        with pytest.raises(HTTPException) as e:
            asyncio.run(app_main.physiology_bid({**_want().model_dump(mode="json"), "tier": 0}))
        assert e.value.status_code == 422
        rel = asyncio.run(app_main.physiology_bid_release("w0"))
        assert rel == {"want_id": "w0", "released": 1 if out["outcome"] == "funded" else 0}
        assert asyncio.run(app_main.physiology_reservations())["shadow"] is True
    finally:
        app_main.market = old


@pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")
def test_history_sql_on_postgres():
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    schema = f"t_{uuid4().hex[:10]}"

    async def run():
        async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            async with AsyncConnectionPool(DSN, min_size=1, max_size=2, open=False,
                                           kwargs={"autocommit": True, "row_factory": dict_row,
                                                   "options": f"-c search_path={schema}"}) as pool:
                async with pool.connection() as conn:
                    await conn.execute("""CREATE TABLE grammar_acts (act_id varchar PRIMARY KEY, actor varchar,
                        opened_at timestamptz, closed_at timestamptz, outcome_value double precision,
                        outcome_source varchar, signal_id varchar, cost_json jsonb, created_at timestamptz DEFAULT now())""")
                    await conn.execute("""CREATE TABLE act_downstream_use (act_id varchar PRIMARY KEY, actor varchar,
                        started_at timestamptz, link_status varchar, used boolean, window_closed boolean)""")
                    rows = [("new", "x", NOW - timedelta(days=1), NOW - timedelta(hours=20)),
                            ("old1", "y", NOW - timedelta(days=50), NOW - timedelta(days=49)),
                            ("old2", "y", NOW - timedelta(days=40), NOW - timedelta(days=39))]
                    for aid, actor, o, c in rows:
                        await conn.execute("INSERT INTO grammar_acts (act_id, actor, opened_at, closed_at, cost_json) "
                                           "VALUES (%s, %s, %s, %s, %s)", (aid, actor, o, c, json.dumps({"seconds": 1.0})))
                    await conn.execute("INSERT INTO act_downstream_use VALUES ('new', 'x', %s, 'linked', true, true)",
                                       (NOW - timedelta(days=1),))
                    # want_ledger's real columns (services/orion-sql-db/manual_migration_want_ledger_v1.sql).
                    await conn.execute("""CREATE TABLE want_ledger (decision_id text PRIMARY KEY, want_id text NOT NULL,
                        family text NOT NULL, producer text NOT NULL, verb text NOT NULL, tier integer NOT NULL,
                        decided_at timestamptz NOT NULL, outcome text NOT NULL, reason text NOT NULL, lane text,
                        net double precision, binding_constraint text, shadow boolean NOT NULL, correlation_id text,
                        decision jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now())""")
                    for did, outcome, at, producer in (("f1", "funded", NOW - timedelta(days=1), "p"),
                                                       ("d1", "deferred", NOW, "p"),
                                                       ("old", "funded", NOW - timedelta(days=9), "p"),
                                                       ("probe1", "funded", NOW, "deploy-verification")):
                        await conn.execute("INSERT INTO want_ledger (decision_id, want_id, family, producer, verb, tier, "
                                           "decided_at, outcome, reason, shadow, decision) VALUES "
                                           "(%s, 'w', 'durable.self_inquiry', %s, 'ask', 2, %s, %s, 'r', true, '{}')",
                                           (did, producer, at, outcome))
                funded = await read_funded(pool, NOW)
                assert [f.decision_id for f in funded] == ["f1"] and funded[0].tier == 2
                acts, uses = await read_history(pool, NOW)
                assert sorted(a.act_id for a in acts) == ["new", "old2"]   # y's newest close survives the window
                assert acts[0].cost == {"seconds": 1.0} or acts[1].cost == {"seconds": 1.0}
                assert [u.used for u in uses] == [True]
        finally:
            async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
                await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')

    assert "UNION" in ACTS_SQL and "act_downstream_use" in USES_SQL and "want_ledger" in FUNDED_SQL
    asyncio.run(run())
