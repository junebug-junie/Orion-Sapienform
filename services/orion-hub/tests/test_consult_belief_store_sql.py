"""The consult belief SQL against a real Postgres (take -> offers counted -> stamp | release).

Runs on a TEMP copy of ``orion_consult`` (pg_temp shadows public) with the belief migration applied
to it, inside a transaction that is rolled back: nothing reaches the live table. Skipped where no
local Postgres answers (CI)."""

from __future__ import annotations

import pathlib

import pytest
import pytest_asyncio

from orion.consult.beliefs import MAX_BELIEF_OFFERS
from scripts import consult_loop as cl

asyncpg = pytest.importorskip("asyncpg")

DSN = "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"
MIGRATION = pathlib.Path(__file__).resolve().parents[3] / "services/orion-sql-db/manual_migration_orion_consult_belief_v1.sql"


@pytest_asyncio.fixture
async def conn():
    try:
        c = await asyncpg.connect(DSN, timeout=3)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no local Postgres: {exc}")
    tx = c.transaction()
    await tx.start()
    try:
        # The pre-belief shape (v1 columns), then this patch's migration on top of it.
        await c.execute(
            "CREATE TEMP TABLE orion_consult AS SELECT consult_id, source_kind, source_ref, source_detail, question, "
            "status, status_reason, ending, turns, claude_cost_usd, answer, brief_json, brief_text, brief_sent_at, "
            "email_status, decision, decision_reason, decided_at, orion_told_run_id, pending_prompt, "
            "pending_correlation_id, gate_retry_part, claude_resends, orion_told_at, created_at, updated_at "
            "FROM public.orion_consult WHERE false"
        )
        await c.execute(MIGRATION.read_text())
        for cid, status, decision in (("c1", "answered", None), ("c2", "brief_sent", "yes"),
                                      ("c3", "brief_sent", None), ("c4", "failed", None)):
            await c.execute(
                "INSERT INTO orion_consult (consult_id, source_kind, source_ref, question, status, ending, decision, "
                "turns, claude_cost_usd, claude_resends, created_at, updated_at) "
                "VALUES ($1, 'curiosity_run', $1, 'q', $2, $3, $4, 0, 0, 0, now(), now())",
                cid, status, "answered" if status == "answered" else "agreed_proposal", decision,
            )
        yield c
    finally:
        await tx.rollback()
        await c.close()


async def _ids(rows):
    return sorted(r["consult_id"] for r in rows)


@pytest.mark.asyncio
async def test_take_counts_offers_and_stops_after_the_cap(conn):
    for n in range(1, MAX_BELIEF_OFFERS + 1):
        taken = await cl.take_outcomes(conn, run_id=f"r{n}")
        assert await _ids(taken) == ["c1", "c2"], "answered + decided only"
        assert {r["belief_offers"] for r in taken} == {n}
        assert await _ids(await cl.outcomes_offered_to(conn, f"r{n}")) == ["c1", "c2"]
        await cl.release_outcomes(conn, f"r{n}", consult_ids=["c1", "c2"], refund=False)
    assert await cl.take_outcomes(conn, run_id="r9") == [], "offered the cap: Hub stops asking"


@pytest.mark.asyncio
async def test_a_stamped_outcome_is_never_offered_again_and_a_refund_does_not_count(conn):
    await cl.take_outcomes(conn, run_id="r1")
    await cl.stamp_belief(conn, "c1", run_id="r1", finding_id="consult:c1:answer", prior_ids=["p1", "p2"])
    await cl.release_outcomes(conn, "r1", refund=True)  # c2: the run never reached its turn
    row1 = await cl.get(conn, "c1")
    assert row1["belief_finding_id"] == "consult:c1:answer" and list(row1["belief_prior_ids"]) == ["p1", "p2"]
    assert row1["belief_recorded_run_id"] == "r1" and row1["belief_recorded_at"] is not None
    assert row1["orion_told_run_id"] == "r1", "a recorded outcome keeps its claim"
    row2 = await cl.get(conn, "c2")
    assert row2["orion_told_run_id"] is None and row2["belief_offers"] == 0
    assert await cl.outcomes_offered_to(conn, "r1") == []
    assert await _ids(await cl.take_outcomes(conn, run_id="r2")) == ["c2"]


@pytest.mark.asyncio
async def test_an_outcome_shown_before_offers_were_counted_is_offered_again(conn):
    await conn.execute("UPDATE orion_consult SET orion_told_run_id = 'old-run', orion_told_at = now() WHERE consult_id = 'c1'")
    taken = await cl.take_outcomes(conn, run_id="r1")
    assert await _ids(taken) == ["c1", "c2"] and {r["belief_offers"] for r in taken} == {1}
