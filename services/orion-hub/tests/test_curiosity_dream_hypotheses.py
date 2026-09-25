"""Hub claims dream hypotheses only when the kickoff will actually show them.

Claiming stamps `offered_at` and each hypothesis is offered once, so a claim on
a run whose prompt drops the section would burn it unseen.
"""

from __future__ import annotations

import asyncio

from orion.core.bus.bus_schemas import ServiceRef
from orion.curiosity.worldview import WorldviewSnapshot
from scripts.curiosity_investigation import CuriosityInvestigation

SOURCE = ServiceRef(name="orion-hub", version="0.1.0", node="athena")


class _Conn:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch(self, sql, *args):
        self.calls += 1
        return [{"hypothesis_id": "dh-1", "claim": "a claim that is long enough", "why": ""}]


class _Pool:
    def __init__(self, conn) -> None:
        self.conn = conn

    def acquire(self):
        conn = self.conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return False

        return _Ctx()


def _loop(conn, *, enabled=True, reader=object()) -> CuriosityInvestigation:
    return CuriosityInvestigation(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=60.0, daily_cap=3,
        timeout_sec=60.0, session_id="s", pool_provider=lambda: _Pool(conn),
        source_ref=SOURCE, reader=reader,
        dream_hypotheses_enabled=enabled, dream_hypotheses_per_run=2,
    )


def test_claims_when_enabled_and_graph_writable():
    conn = _Conn()
    got = asyncio.run(_loop(conn)._take_dream_hypotheses(WorldviewSnapshot(), "abcd1234abcd"))
    assert [h.hypothesis_id for h in got] == ["dh-1"] and conn.calls == 1


def test_no_claim_when_flag_off_graph_off_unreadable_or_no_run():
    for loop, view, run_id in (
        (_loop(_Conn(), enabled=False), WorldviewSnapshot(), "abcd1234abcd"),
        (_loop(_Conn(), reader=None), WorldviewSnapshot(), "abcd1234abcd"),
        (_loop(_Conn()), WorldviewSnapshot(unavailable_reason="down"), "abcd1234abcd"),
        (_loop(_Conn()), WorldviewSnapshot(), ""),
    ):
        assert asyncio.run(loop._take_dream_hypotheses(view, run_id)) == ()
        assert loop._pool_provider().conn.calls == 0
