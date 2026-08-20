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


class _FakeEvidence:
    def __init__(self, source_kind, source_id):
        self.source_kind = source_kind
        self.source_id = source_id


class _FakeCrystallization:
    """Carries evidence BY DEFAULT, mirroring live rows.

    The first version of this fixture hardcoded `self.evidence = []`, so all 12 tests ran
    with an empty evidence list and the evidence loop was never executed once. That is
    exactly where the feature was defeated: refs excused by the grammar loop were re-flagged
    as fatal one loop later, and live 61 of 61 affected crystallizations still quarantined.
    Twelve green tests, zero production effect. A fake wrong in the same direction as the
    code is worse than no fake.

    Live 2026-08-20 there are 2,586 `grammar_event` rows in memory_crystallization_sources,
    and every affected crystallization carried the same ids in BOTH carriers -- so mirroring
    that overlap is the realistic default, not an edge case.
    """

    def __init__(self, gevs, created_at=None, evidence=None, mirror_evidence=True):
        self.crystallization_id = "test-crys"
        self.source_card_ids = []
        self.source_grammar_event_ids = list(gevs)
        if evidence is not None:
            self.evidence = list(evidence)
        elif mirror_evidence:
            self.evidence = [_FakeEvidence("grammar_event", g) for g in gevs]
        else:
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


class TestAnAbsentGrammarRefIsReportedNeverFatal:
    """`grammar_events` is the only source store with retention, so every grammar reference
    is perishable by construction. Treating absence as a defect quarantines an ever-growing
    share of good proposals.

    An earlier version tried to split "aged out" from "never existed" by comparing the
    crystallization's created_at to the live retention horizon. Review killed it with live
    data: crystallizations COPY REFS FORWARD (one inherited seven ids verbatim from a
    proposal minted 25h earlier, and the same ids got opposite verdicts from the two
    carriers), and refs lag their carrier by p95 18.3h / max 43.6h against a 3-day window.
    All 14 refs that rule called "genuinely missing" were in fact aged out -- a 100%
    false-positive rate on its own error bucket.
    """

    @pytest.mark.asyncio
    async def test_an_absent_ref_is_reported_and_does_not_invalidate(self):
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization(["gev_old"])
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.absent_grammar_refs == ["grammar_event:gev_old"]
        assert res.unresolved == []
        assert res.errors == []
        assert res.valid is True

    @pytest.mark.asyncio
    async def test_the_evidence_loop_cannot_re_flag_what_the_grammar_loop_excused(self):
        """THE bug that made the whole feature inert in production.

        The same id appears in `source_grammar_event_ids` AND in `evidence`. The old code
        walked them in separate loops, so the second loop appended the id to `unresolved` and
        `errors`, and the route quarantined the proposal anyway -- 61 of 61 live.
        """
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization(["gev_dup"])  # mirrored into evidence by default
        assert [e.source_id for e in crys.evidence] == ["gev_dup"], "fixture lost its evidence"
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.valid is True, res.errors
        assert res.unresolved == []
        assert res.absent_grammar_refs == ["grammar_event:gev_dup"]

    @pytest.mark.asyncio
    async def test_a_grammar_ref_present_only_in_evidence_is_still_checked(self):
        """Not every grammar ref is in `source_grammar_event_ids`. Skipping the evidence
        carrier entirely would silently stop resolving those."""
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization([], evidence=[_FakeEvidence("grammar_event", "gev_ev")])
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.absent_grammar_refs == ["grammar_event:gev_ev"]
        assert res.valid is True

    @pytest.mark.asyncio
    async def test_ids_are_deduplicated_across_both_carriers(self):
        """Live, ids repeat within one array and across both carriers. Counting them twice
        inflates whatever number an operator reads off the response."""
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization(
            ["gev_a", "gev_a", "gev_b"],
            evidence=[_FakeEvidence("grammar_event", "gev_a"),
                      _FakeEvidence("grammar_event", "gev_b")],
        )
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.absent_grammar_refs == ["grammar_event:gev_a", "grammar_event:gev_b"]

    @pytest.mark.asyncio
    async def test_a_resolving_ref_is_not_reported_as_absent(self):
        conn = _FakeConn(events={"gev_here"}, traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization(["gev_here"])
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.absent_grammar_refs == []
        assert res.valid is True

    @pytest.mark.asyncio
    async def test_non_grammar_evidence_still_invalidates(self):
        """Only the grammar store is bounded. Absence in an unbounded store really does mean
        the reference is broken, and must still quarantine."""
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization([], evidence=[_FakeEvidence("memory_card", "")])
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.valid is False
        assert res.unresolved == ["memory_card:"]

    @pytest.mark.asyncio
    async def test_a_probe_failure_is_unverified_not_absent_and_not_fatal(self):
        """"I could not look" is not "it is gone", and making it fatal quarantines every
        proposal validated during a database blip -- persisted to disk, on rows that may be
        `active`."""
        conn = _FakeConn(events=set(), traces=set(), horizon=None, fail=True)
        crys = _FakeCrystallization(["gev_x"])
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert res.unverified_grammar_refs == ["grammar_event:gev_x"]
        assert res.absent_grammar_refs == []
        assert res.unresolved == []
        assert res.valid is True

    @pytest.mark.asyncio
    async def test_no_grammar_refs_makes_no_grammar_queries_at_all(self):
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization([], evidence=[])
        res = await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert conn.seen == []
        assert res.valid is True

    @pytest.mark.asyncio
    async def test_the_retention_horizon_is_never_consulted(self):
        """Pins the removal of the inference, so it cannot quietly come back."""
        assert not hasattr(sources, "grammar_retention_horizon")
        conn = _FakeConn(events=set(), traces=set(), horizon=HORIZON)
        crys = _FakeCrystallization(["gev_a"])
        await sources.resolve_crystallization_sources(_FakePool(conn), crys)
        assert not any("MIN(created_at)" in q for q in conn.seen), conn.seen
