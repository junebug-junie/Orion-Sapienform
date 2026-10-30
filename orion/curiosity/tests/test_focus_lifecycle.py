"""Curiosity focus lifecycle: queue -> take -> outcome, replace, Juniper protection, expiry,
cancel-before-seen release, failed run not requeued, clear.

Every scenario runs on two backends: the in-memory fake the Hub tests use
(``focus_fixtures.FocusTable``) and a real Postgres with the real migration applied
(ORION_MEMORY_EPISODE_TEST_DATABASE_URL: admin DSN of a THROWAWAY server; each test makes and
drops its own database. CI: .github/workflows/orion-memory-episode-tests.yml, which fails on any
skip). So the SQL is proven on Postgres and the fake cannot drift from it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

from orion.curiosity import focus as cf
from orion.curiosity.tests.focus_fixtures import FocusPool, FocusTable

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
MIGRATION = Path(__file__).resolve().parents[3] / cf.MIGRATION
REFS_MIGRATION = Path(__file__).resolve().parents[3] / cf.REFS_MIGRATION
RUN_A, RUN_B = "a1a1a1a1a1a1", "b2b2b2b2b2b2"


def _backends():
    yield pytest.param("fake", id="fake")
    yield pytest.param("pg", id="postgres", marks=pytest.mark.skipif(
        not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set"))


def run(backend: str, scenario) -> None:
    """Run ``scenario(pool)`` on one backend."""

    async def _go():
        if backend == "fake":
            await scenario(FocusPool(FocusTable()))
            return
        import asyncpg

        name = f"focus_{uuid.uuid4().hex[:10]}"
        admin = await asyncpg.connect(ADMIN_DSN)
        await admin.execute(f'CREATE DATABASE "{name}"')
        await admin.close()
        pool = await asyncpg.create_pool(dsn=ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}", min_size=1, max_size=3)
        try:
            async with pool.acquire() as conn:
                await conn.execute(MIGRATION.read_text())
                await conn.execute(REFS_MIGRATION.read_text())
            await scenario(pool)
        finally:
            await pool.close()
            admin = await asyncpg.connect(ADMIN_DSN)
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
            await admin.close()

    asyncio.run(_go())


def q(pool, question="what is the bus's real fan-out?", *, why="I keep guessing", origin="self",
      chat="chat-1", ttl=cf.FOCUS_TTL):
    return cf.queue_focus(pool, question=question, why=why, origin=origin, chat_correlation_id=chat, ttl=ttl)


async def _all(pool):
    rows, _ = await cf.read_recent(pool, since=None, limit=50)
    return {r.focus_id: r for r in rows}


@pytest.mark.parametrize("backend", _backends())
def test_queue_is_one_pending_row_and_a_retry_is_the_same_row(backend):
    async def scenario(pool):
        first = await q(pool)
        again = await q(pool)
        assert first.focus.status == cf.PENDING and not first.already_queued and first.replaced is None
        assert again.already_queued and again.focus.focus_id == first.focus.focus_id
        assert first.focus.expires_at - first.focus.created_at == cf.FOCUS_TTL
        assert [r.status for r in (await _all(pool)).values()] == [cf.PENDING]
        assert (await cf.read_pending(pool)).focus_id == first.focus.focus_id

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_a_new_focus_replaces_the_pending_one_and_names_it(backend):
    async def scenario(pool):
        old = await q(pool, "question one")
        new = await q(pool, "question two")
        assert new.replaced is not None and new.replaced.question == "question one"
        rows = await _all(pool)
        assert rows[old.focus.focus_id].status == cf.REPLACED
        assert rows[old.focus.focus_id].replaced_by == new.focus.focus_id
        assert rows[new.focus.focus_id].status == cf.PENDING
        assert "replaced by a newer focus" in cf.focus_story(rows[old.focus.focus_id])

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_orion_cannot_replace_a_focus_juniper_asked_for(backend):
    async def scenario(pool):
        jun = await q(pool, "look at the thermal logs", origin="juniper")
        with pytest.raises(cf.FocusRefused) as exc:
            await q(pool, "my own idea", origin="self")
        assert cf.JUNIPER_PROTECTED in str(exc.value)
        rows = await _all(pool)
        assert list(rows) == [jun.focus.focus_id] and rows[jun.focus.focus_id].status == cf.PENDING
        # Juniper's own new ask may replace hers; and then Orion still may not.
        jun2 = await q(pool, "actually, the fan curve", origin="juniper")
        assert jun2.replaced.focus_id == jun.focus.focus_id

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_juniper_may_replace_orions_choice(backend):
    async def scenario(pool):
        mine = await q(pool, "my idea", origin="self")
        hers = await q(pool, "her ask", origin="juniper")
        assert hers.replaced.focus_id == mine.focus.focus_id

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_take_once_then_turn_end_stamps_the_disposition(backend):
    async def scenario(pool):
        queued = await q(pool)
        taken = await cf.take_focus(pool, run_id=RUN_A)
        assert taken.focus_id == queued.focus.focus_id and taken.status == cf.TAKEN
        assert taken.offered_run_id == RUN_A
        assert await cf.take_focus(pool, run_id=RUN_B) is None
        stamped = await cf.stamp_focus_outcome(pool, run_id=RUN_A, turn_ok=True,
                                               disposition="explored", disposition_raw=None)
        assert stamped == queued.focus.focus_id
        row = (await _all(pool))[stamped]
        assert row.status == cf.RAN and row.turn_ok is True and row.disposition == "explored"
        story = cf.focus_story(row)
        assert f"taken by run {RUN_A}" in story and "disposition: explored" in story

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_no_disposition_reads_unknown_not_explored(backend):
    async def scenario(pool):
        await q(pool)
        await cf.take_focus(pool, run_id=RUN_A)
        await cf.stamp_focus_outcome(pool, run_id=RUN_A, turn_ok=True, disposition=None, disposition_raw=None)
        row = next(iter((await _all(pool)).values()))
        assert row.status == cf.RAN and row.disposition is None
        assert "disposition: unknown (the run recorded none)" in cf.focus_story(row)
        assert "explored" not in cf.disposition_words(row)

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_a_run_cancelled_before_it_saw_the_focus_gives_it_back(backend):
    async def scenario(pool):
        queued = await q(pool)
        await cf.take_focus(pool, run_id=RUN_A)
        assert await cf.release_focus(pool, run_id=RUN_A) == cf.PENDING
        again = await cf.take_focus(pool, run_id=RUN_B)
        assert again.focus_id == queued.focus.focus_id and again.offered_run_id == RUN_B

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_a_release_after_a_newer_focus_was_queued_reads_replaced(backend):
    async def scenario(pool):
        old = await q(pool, "old")
        await cf.take_focus(pool, run_id=RUN_A)
        new = await q(pool, "newer")  # the slot is free while the old one is taken
        assert new.replaced is None
        assert await cf.release_focus(pool, run_id=RUN_A) == cf.REPLACED
        rows = await _all(pool)
        assert rows[old.focus.focus_id].status == cf.REPLACED
        assert rows[old.focus.focus_id].replaced_by == new.focus.focus_id
        assert rows[new.focus.focus_id].status == cf.PENDING

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_a_failed_run_is_never_requeued(backend):
    async def scenario(pool):
        await q(pool)
        await cf.take_focus(pool, run_id=RUN_A)
        await cf.stamp_focus_outcome(pool, run_id=RUN_A, turn_ok=False, disposition=None, disposition_raw=None)
        assert await cf.release_focus(pool, run_id=RUN_A) is None  # the turn ended: Orion saw it
        assert await cf.take_focus(pool, run_id=RUN_B) is None
        row = next(iter((await _all(pool)).values()))
        assert row.status == cf.FAILED and "not re-queued" in cf.focus_story(row)

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_a_run_that_ended_failed_after_it_started_is_failed_not_pending(backend):
    async def scenario(pool):
        await q(pool)
        await cf.take_focus(pool, run_id=RUN_A)
        assert await cf.fail_focus(pool, run_id=RUN_A) is not None
        assert await cf.take_focus(pool, run_id=RUN_B) is None
        assert next(iter((await _all(pool)).values())).status == cf.FAILED

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_a_retry_that_succeeds_wins_and_a_late_failure_does_not_downgrade(backend):
    async def scenario(pool):
        await q(pool)
        await cf.take_focus(pool, run_id=RUN_A)
        await cf.stamp_focus_outcome(pool, run_id=RUN_A, turn_ok=False, disposition=None, disposition_raw=None)
        await cf.stamp_focus_outcome(pool, run_id=RUN_A, turn_ok=True,
                                     disposition="set_aside:not today", disposition_raw=None)
        await cf.stamp_focus_outcome(pool, run_id=RUN_A, turn_ok=False, disposition=None, disposition_raw=None)
        assert await cf.fail_focus(pool, run_id=RUN_A) is None
        row = next(iter((await _all(pool)).values()))
        assert row.status == cf.RAN and row.disposition == "set_aside:not today"
        assert "set aside: not today" in cf.focus_story(row)

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_an_expired_focus_is_not_taken_and_reads_expired(backend):
    async def scenario(pool):
        queued = await q(pool, ttl=timedelta(seconds=-1))
        assert (await _all(pool))[queued.focus.focus_id].status == cf.EXPIRED  # before any sweep
        assert await cf.read_pending(pool) is None
        assert await cf.take_focus(pool, run_id=RUN_A) is None
        row = (await _all(pool))[queued.focus.focus_id]
        assert row.status == cf.EXPIRED and "no run took it" in cf.focus_story(row)
        # The slot is free again for a new focus (the expired row does not block it).
        assert (await q(pool, "next")).replaced is None

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_an_expired_juniper_focus_no_longer_blocks_orion(backend):
    async def scenario(pool):
        await q(pool, "hers", origin="juniper", ttl=timedelta(seconds=-1))
        mine = await q(pool, "mine", origin="self")
        assert mine.focus.status == cf.PENDING and mine.replaced is None

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_clear_only_clears_a_pending_focus(backend):
    async def scenario(pool):
        queued = await q(pool)
        assert await cf.clear_focus(pool, focus_id=uuid.UUID(queued.focus.focus_id)) is True
        assert await cf.clear_focus(pool, focus_id=uuid.UUID(queued.focus.focus_id)) is False
        assert await cf.take_focus(pool, run_id=RUN_A) is None
        assert "cleared by Juniper" in cf.focus_story((await _all(pool))[queued.focus.focus_id])

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_read_for_runs_maps_a_run_to_its_focus(backend):
    async def scenario(pool):
        queued = await q(pool)
        await cf.take_focus(pool, run_id=RUN_A)
        found = await cf.read_for_runs(pool, [RUN_A, RUN_B])
        assert list(found) == [RUN_A] and found[RUN_A].focus_id == queued.focus.focus_id

    run(backend, scenario)


# --- pure helpers -------------------------------------------------------------------------------


@pytest.mark.parametrize("raw,want", [
    ("explored", ("explored", None)),
    ("  Explored ", ("explored", None)),
    ("set_aside: the GPU logs were empty", ("set_aside:the GPU logs were empty", None)),
    ("set aside - not enough material", ("set_aside:not enough material", None)),
    ("set_aside", (None, "set_aside")),
    ("set_aside:   ", (None, "set_aside:")),
    ("partly", (None, "partly")),
    ("", (None, None)),
    (None, (None, None)),
])
def test_parse_disposition(raw, want):
    assert cf.parse_disposition(raw) == want


def test_focus_id_is_stable_and_origin_sensitive():
    a = cf.focus_id_for("q", "w", "self", "c")
    assert a == cf.focus_id_for("q", "w", "self", "c") != cf.focus_id_for("q", "w", "juniper", "c")


def test_disposition_cypher_refuses_a_non_hex_run_id():
    with pytest.raises(ValueError):
        cf.disposition_cypher("x' OR 1=1 //")
    assert "focus_disposition IS NOT NULL" in cf.disposition_cypher("abcdef123456")


class _Reader:
    def __init__(self, rows=None, raises=False):
        self.rows, self.raises, self.seen = rows or [], raises, []

    def query(self, cypher):
        self.seen.append(cypher)
        if self.raises:
            raise RuntimeError("graph down")
        return self.rows


def test_read_disposition_unknown_when_unreadable_or_absent():
    assert cf.read_disposition(None, RUN_A) == (None, None, False)
    assert cf.read_disposition(_Reader(raises=True), RUN_A) == (None, None, False)
    assert cf.read_disposition(_Reader([]), RUN_A) == (None, None, True)
    assert cf.read_disposition(_Reader([{"focus_disposition": "explored"}]), RUN_A) == ("explored", None, True)


def test_the_kickoff_section_shows_the_question_and_asks_for_a_disposition():
    row = cf.FocusRow(focus_id="f", question="why  does the bus\nspike?", why="it bit us", origin="self",
                      status=cf.TAKEN)
    text = "\n".join(cf.format_focus_section(row))
    assert cf.FOCUS_HEADER in text and "why does the bus spike?" in text and "it bit us" in text
    assert 'focus_disposition: "explored"' in text and "set_aside" in text
    assert "recorded as unknown, not as explored" in text and "offer, not an order" in text
    jun = "\n".join(cf.format_focus_section(cf.FocusRow(**{**row.__dict__, "origin": "juniper"})))
    assert "Juniper asked you" in jun
    assert cf.format_focus_section(None) == []


# --- review fixes (2026-10-16) ------------------------------------------------------------------


async def _age_taken(pool, hours: float) -> None:
    """Move every taken row's offered_at back by ``hours`` (Postgres has no settable clock)."""
    if isinstance(pool, FocusPool):
        for r in pool.table.rows:
            if r["offered_at"] is not None:
                r["offered_at"] -= timedelta(hours=hours)
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE curiosity_focus_request SET offered_at = offered_at - make_interval(hours => $1) "
            "WHERE offered_at IS NOT NULL", int(hours))


@pytest.mark.parametrize("backend", _backends())
def test_an_unreadable_graph_on_a_retry_keeps_the_known_disposition(backend):
    async def scenario(pool):
        await q(pool)
        await cf.take_focus(pool, run_id=RUN_A)
        await cf.stamp_focus_outcome(pool, run_id=RUN_A, turn_ok=True, disposition="explored", disposition_raw=None)
        await cf.stamp_focus_outcome(pool, run_id=RUN_A, turn_ok=True, disposition=None, disposition_raw=None,
                                     graph_read=False)
        assert next(iter((await _all(pool)).values())).disposition == "explored"

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_a_taken_focus_whose_end_was_never_recorded_is_failed_after_48h(backend):
    async def scenario(pool):
        await q(pool, "stuck")
        await cf.take_focus(pool, run_id=RUN_A)
        await _age_taken(pool, 49)
        await q(pool, "next")
        await cf.take_focus(pool, run_id=RUN_B)  # the take path runs the sweep
        rows = {r.question: r for r in (await _all(pool)).values()}
        assert rows["stuck"].status == cf.FAILED and rows["stuck"].turn_ok is None
        assert "end was never recorded" in cf.focus_story(rows["stuck"])
        # A late turn end still records what really happened.
        await cf.stamp_focus_outcome(pool, run_id=RUN_A, turn_ok=True, disposition="explored", disposition_raw=None)
        assert {r.question: r for r in (await _all(pool)).values()}["stuck"].status == cf.RAN

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_a_fresh_taken_focus_is_not_swept(backend):
    async def scenario(pool):
        await q(pool, "fresh")
        await cf.take_focus(pool, run_id=RUN_A)
        await _age_taken(pool, 47)
        await cf.take_focus(pool, run_id=RUN_B)
        assert next(iter((await _all(pool)).values())).status == cf.TAKEN

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_clear_does_not_hide_an_expiry(backend):
    async def scenario(pool):
        queued = await q(pool, ttl=timedelta(seconds=-1))
        assert await cf.clear_focus(pool, focus_id=uuid.UUID(queued.focus.focus_id)) is False
        assert (await _all(pool))[queued.focus.focus_id].status == cf.EXPIRED

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_repeating_a_replaced_focus_is_refused_not_reported_queued(backend):
    async def scenario(pool):
        await q(pool, "A")
        await q(pool, "B")
        with pytest.raises(cf.FocusRefused, match="now replaced; nothing was queued"):
            await q(pool, "A")
        pending = await cf.read_pending(pool)
        assert pending.question == "B"

    run(backend, scenario)


def test_the_kickoff_template_value_is_a_placeholder_that_reads_unknown():
    from orion.curiosity.kickoff_prompt import _outcome_section

    text = "\n".join(_outcome_section(run_id="abcdef123456", focus=True))
    assert 'focus_disposition: "explored"' not in text
    copied = '<explored | set_aside: why not>'
    assert f'focus_disposition: "{copied}"' in text
    assert cf.parse_disposition(copied) == (None, copied)


@pytest.mark.parametrize("backend", _backends())
def test_keep_pending_never_replaces_a_pending_focus(backend):
    """Orion Prime 2A: a question Prime chose alone never displaces one chosen in a conversation."""
    async def scenario(pool):
        chat = await q(pool, "the one from chat", origin="self")
        with pytest.raises(cf.FocusRefused) as exc:
            await cf.queue_focus(pool, question="prime's own", why="a concern", origin="self",
                                 chat_correlation_id="prime:t1", keep_pending=True)
        assert cf.PENDING_KEPT in str(exc.value)
        rows = await _all(pool)
        assert list(rows) == [chat.focus.focus_id] and rows[chat.focus.focus_id].status == cf.PENDING

    run(backend, scenario)


@pytest.mark.parametrize("backend", _backends())
def test_keep_pending_queues_into_an_empty_slot(backend):
    async def scenario(pool):
        got = await cf.queue_focus(pool, question="prime's own", why="a concern", origin="self",
                                   chat_correlation_id="prime:t1", keep_pending=True)
        assert got.focus.status == cf.PENDING and cf.is_prime_focus(got.focus)

    run(backend, scenario)


def test_a_prime_focus_is_never_described_as_chosen_in_a_chat():
    row = cf.FocusRow(focus_id="f", question="q?", why="w", origin="self", chat_correlation_id="prime:abc",
                      created_at=None, expires_at=None, status=cf.PENDING)
    text = "\n".join(cf.format_focus_section(row))
    assert "Between conversations" in text and "In a conversation with Juniper" not in text
    assert "between conversations" in cf.focus_story(row)
    chat = dataclasses.replace(row, chat_correlation_id="chat-1")
    assert "In a conversation with Juniper" in "\n".join(cf.format_focus_section(chat))


@pytest.mark.parametrize("backend", _backends())
def test_refs_are_stored_read_back_and_shown_with_the_command_that_opens_them(backend):
    """Orion Prime step 1: the ids a question cites travel on the row to the run's focus section."""
    refs = [{"kind": "arc", "id": "8d87adeca2ce06a5"}, {"kind": "act", "id": "act_42c0e5f55fbd5d881786ebd4"}]

    async def scenario(pool):
        queued = await cf.queue_focus(pool, question="what drives this arc?", why="it keeps returning",
                                      origin="self", chat_correlation_id="prime:t1", keep_pending=True, refs=refs)
        assert list(queued.focus.refs) == refs
        taken = await cf.take_focus(pool, run_id=RUN_A)
        assert taken is not None and list(taken.refs) == refs
        plain = await q(pool, question="no refs here", chat="chat-2")
        assert plain.focus.refs == () and cf.format_focus_section(plain.focus)[3:5] != []
        assert not any("orion.dig" in line for line in cf.format_focus_section(plain.focus))
        section = "\n".join(cf.format_focus_section(taken))
        assert "python -P -m orion.dig arc 8d87adeca2ce06a5" in section
        assert "python -P -m orion.dig act act_42c0e5f55fbd5d881786ebd4" in section

    run(backend, scenario)
