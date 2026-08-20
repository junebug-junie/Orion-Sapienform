"""The grammar-event source resolver, which was dead from the day it was written.

`resolve_grammar_event_ref` queried `grammar_traces WHERE trace_id = $1 OR event_id = $1`.
`grammar_traces` has no `event_id` column, so that statement raised on EVERY call and was
swallowed by a bare `except Exception: pass`. It then fell through to
`substrate_grammar_events`, which does not exist in this database, which raised too. The
function's real behaviour was its last line and nothing else:

    return str(event_id).startswith("gev_")

Confirmed live 2026-08-20: all 1,167 distinct referenced ids are `gev_`-prefixed, so it
returned True for all of them -- including 876 whose events had been pruned and 14 that never
existed. A validator that cannot fail is not a validator.

These tests use a fake pool that models the REAL schema, so the original implementation
cannot pass them: its first query names a column the fake does not have, exactly as in
production.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.memory.crystallization import sources


class _FakeConn:
    """Models the live schema: grammar_events has event_id AND trace_id; grammar_traces has
    only trace_id; substrate_grammar_events does not exist."""

    def __init__(self, events: set[str], traces: set[str], horizon: datetime | None,
                 fail: bool = False):
        self.events, self.traces, self.horizon, self.fail = events, traces, horizon, fail
        self.seen: list[str] = []

    async def fetchrow(self, sql, *args):
        self.seen.append(" ".join(sql.split()))
        if self.fail:
            raise RuntimeError("connection reset")
        if "grammar_traces" in sql and "event_id" in sql:
            raise RuntimeError('column "event_id" does not exist')
        if "substrate_grammar_events" in sql:
            raise RuntimeError('relation "substrate_grammar_events" does not exist')
        ref = args[0]
        if ref in self.events or ref in self.traces:
            return {"?column?": 1}
        return None

    async def fetchval(self, sql, *args):
        self.seen.append(" ".join(sql.split()))
        if self.fail:
            raise RuntimeError("connection reset")
        return self.horizon


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return False

        return _Ctx()


class _FakeCrystallization:
    def __init__(self, gevs, created_at):
        self.crystallization_id = "test-crys"
        self.source_card_ids = []
        self.source_grammar_event_ids = list(gevs)
        self.evidence = []
        self.created_at = created_at


NOW = datetime(2026, 8, 20, tzinfo=timezone.utc)
HORIZON = NOW - timedelta(days=3)


class TestTheResolverActuallyQueriesTheRightTable:
    @pytest.mark.asyncio
    async def test_a_real_event_id_resolves(self):
        conn = _FakeConn(events={"gev_real"}, traces=set(), horizon=HORIZON)
        assert await sources.resolve_grammar_event_ref(_FakePool(conn), "gev_real") is True

    @pytest.mark.asyncio
    async def test_a_trace_id_also_resolves(self):
        """A grammar reference may name the episode or a step within it."""
        conn = _FakeConn(events=set(), traces={"hub.chat:abc"}, horizon=HORIZON)
        assert await sources.resolve_grammar_event_ref(_FakePool(conn), "hub.chat:abc") is True

    @pytest.mark.asyncio
    async def test_a_gev_prefixed_id_that_does_not_exist_does_NOT_resolve(self):
        """THE test the old code could never pass. Its entire behaviour was
        `startswith("gev_")`, so this exact input returned True."""
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        assert await sources.resolve_grammar_event_ref(_FakePool(conn), "gev_nope") is False

    @pytest.mark.asyncio
    async def test_it_never_touches_the_two_tables_that_do_not_work(self):
        conn = _FakeConn(events={"gev_real"}, traces=set(), horizon=HORIZON)
        await sources.resolve_grammar_event_ref(_FakePool(conn), "gev_real")
        joined = " ".join(conn.seen)
        assert "substrate_grammar_events" not in joined
        assert "grammar_traces" not in joined
        assert "grammar_events" in joined

    @pytest.mark.asyncio
    async def test_a_probe_failure_raises_instead_of_returning_false(self):
        """An unreachable database is not evidence that a reference is bad. Returning False
        here would quarantine every proposal validated during an outage."""
        conn = _FakeConn(events=set(), traces=set(), horizon=None, fail=True)
        with pytest.raises(Exception):
            await sources.resolve_grammar_event_ref(_FakePool(conn), "gev_x")


class TestPrunedIsNotTheSameAsMissing:
    """`grammar_events` is the only bounded source store, so "absent" is ambiguous and the
    two meanings must not collapse. Live 2026-08-20 this rule split 876 aged-out refs from
    14 genuinely-missing ones."""

    @pytest.mark.asyncio
    async def test_a_ref_from_before_the_horizon_is_pruned_not_an_error(self):
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization(["gev_old"], created_at=HORIZON - timedelta(days=5))
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.pruned == ["grammar_event:gev_old"]
        assert res.unresolved == []
        assert res.errors == []
        assert res.valid is True

    @pytest.mark.asyncio
    async def test_a_ref_from_after_the_horizon_is_a_real_error(self):
        """This proposal was minted while the event should still have been on disk."""
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization(["gev_gone"], created_at=HORIZON + timedelta(days=1))
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.unresolved == ["grammar_event:gev_gone"]
        assert res.pruned == []
        assert res.valid is False

    @pytest.mark.asyncio
    async def test_an_unknown_horizon_never_silently_becomes_pruned(self):
        """None means "cannot classify", not "no retention". Guessing pruned here would
        re-create the old bug: every bad ref would be excused as aged-out."""
        conn = _FakeConn(events=set(), traces=set(), horizon=None)
        crys = _FakeCrystallization(["gev_gone"], created_at=NOW - timedelta(days=400))
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.pruned == []
        assert res.unresolved == ["grammar_event:gev_gone"]
        assert res.valid is False

    @pytest.mark.asyncio
    async def test_a_resolving_ref_is_never_classified_as_pruned(self):
        """Even for a proposal older than the horizon: found is found."""
        conn = _FakeConn(events={"gev_still_here"}, traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization(["gev_still_here"],
                                    created_at=HORIZON - timedelta(days=5))
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.pruned == []
        assert res.unresolved == []
        assert res.valid is True

    @pytest.mark.asyncio
    async def test_the_horizon_is_fetched_once_not_per_ref(self):
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization([f"gev_{i}" for i in range(25)],
                                    created_at=HORIZON - timedelta(days=5))
        await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert sum("MIN(created_at)" in q for q in conn.seen) == 1

    @pytest.mark.asyncio
    async def test_no_grammar_refs_means_no_horizon_query_at_all(self):
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization([], created_at=NOW)
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert conn.seen == []
        assert res.valid is True

    @pytest.mark.asyncio
    async def test_a_probe_failure_is_recorded_as_an_error_not_swallowed(self):
        """The failure mode that started all of this: an exception that becomes a pass."""
        conn = _FakeConn(events=set(), traces=set(), horizon=None, fail=True)
        crys = _FakeCrystallization(["gev_x"], created_at=NOW)
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.valid is False
        assert res.unresolved == ["grammar_event:gev_x"]
        assert any("probe failed" in e for e in res.errors), res.errors
