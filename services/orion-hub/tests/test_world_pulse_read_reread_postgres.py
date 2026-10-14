"""Operator reread + policy withdrawal against real SQL (disposable cluster).

RUN_READING_POSTGRES=1 python -m pytest services/orion-hub/tests/test_world_pulse_read_reread_postgres.py -q
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from orion.core.schemas.substrate_graph_journal import SubstrateGraphDecisionV1, SubstrateGraphProposalV1
from orion.schemas.reading import SourceFetchEvidenceV1
from orion.schemas.reading_turn import ReadingRunBriefV1
from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1, WorldPulseReadSeedV1
from orion.substrate.graph_journal import journal_row
from orion.world_pulse_read import operator as op
from orion.world_pulse_read import queue
from orion.world_pulse_read.assertions import MEANING_POLICY, POLICY, READING_ACTOR
from orion.world_pulse_read.durable import bind_turn, consume_turn
from orion.world_pulse_read.meaning_links import WITHDRAW_ACTOR, list_policy_links, withdrawal_for
from test_reading_postgres import db, local_pg, request  # noqa: F401 -- fixture reuse

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reading_dns")]

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def _handoff(seed, *, sha=None):
    return WorldPulseReadHandoffV1(
        seed_ref=seed, what_i_learned="Orion learned something source-backed.",
        trace_id=str(uuid4()), created_at=datetime.now(timezone.utc),
        read_evidence=[SourceFetchEvidenceV1(tool_name="WebFetch", url=seed.url, content_chars=900,
                                             content_sha256=sha)],
    )


async def _bind(conn, seed_id, stage):
    return await bind_turn(conn, ReadingRunBriefV1(
        seed_id=seed_id, stage=stage, prompt="Saved prompt", session_id="reading", timeout_sec=900,
    ), str(uuid4()))


async def _row(conn, seed_id):
    return await conn.fetchrow("SELECT * FROM world_pulse_read_seed WHERE seed_id=$1", seed_id)


async def _read_done(conn, seed_id, *, stage2="done"):
    seed = await queue.claim_next_seed(conn)
    assert seed.seed_id == seed_id
    req = await _bind(conn, seed_id, 1)
    await consume_turn(conn, req.run_id)
    await queue.mark_seed_done(conn, seed_id, trace_id="t1", handoff=_handoff(seed))
    await conn.execute("UPDATE world_pulse_read_seed SET stage2_status=$2, stage2_completed_at=now() "
                       "WHERE seed_id=$1", seed_id, stage2)
    return seed, req


def test_reread_puts_a_finished_read_back_through_both_stages_with_a_new_run(local_pg):
    async def run():
        conn, _ = await db(local_pg)
        made = await queue.enqueue_reading(conn, request("https://example.org/old"))
        seed, first = await _read_done(conn, made["seed_id"])
        old_handoff = (await _row(conn, seed.seed_id))["handoff_json"]

        assert await op.reread_read(conn, seed.seed_id) == {"action": "requeued_reread", "stage": 1}
        row = await _row(conn, seed.seed_id)
        assert (row["status"], row["stage2_status"], row["attempts"], row["last_error"]) == (
            "pending", "pending", 0, op.REREAD_MARKER)
        # The old read stays readable until Stage 1 replaces it.
        assert row["handoff_json"] == old_handoff
        # Stage 2 cannot run on the old handoff while Stage 1 is pending.
        assert await queue.claim_next_stage2_seed(conn) is None
        again = await queue.claim_next_seed(conn)
        assert again.seed_id == seed.seed_id
        second = await _bind(conn, seed.seed_id, 1)
        assert second.run_id != first.run_id
        await queue.mark_seed_done(conn, seed.seed_id, trace_id="t2", handoff=_handoff(seed, sha="ab" * 32))
        row = await _row(conn, seed.seed_id)
        assert row["last_error"] is None
        assert json.loads(row["handoff_json"])["read_evidence"][0]["content_sha256"] == "ab" * 32
        assert row["stage2_status"] == "pending"
        await conn.close()
    asyncio.run(run())


def test_reread_survives_the_stale_digest_sweep_a_new_old_item_does_not(local_pg):
    async def run():
        conn, _ = await db(local_pg)
        item = WorldPulseReadSeedV1(seed_id="digest_item:old-read", kind="digest_item", run_id="r",
                                    url="https://example.org/digest-read", title="Old item")
        fresh = WorldPulseReadSeedV1(seed_id="digest_item:never-read", kind="digest_item", run_id="r",
                                     url="https://example.org/digest-unread", title="Unread")
        await queue.enqueue_seeds(conn, [item])
        await _read_done(conn, item.seed_id)
        await queue.enqueue_seeds(conn, [fresh])
        await conn.execute("UPDATE world_pulse_read_seed SET created_at = now() - interval '30 days'")
        await op.reread_read(conn, item.seed_id)
        assert await queue.skip_stale_digest_items(conn, max_age_sec=86400) == 1
        assert (await _row(conn, fresh.seed_id))["status"] == "skipped"
        assert (await _row(conn, item.seed_id))["status"] == "pending"
        # The already-read sweep does not fold it onto itself either.
        assert await queue.skip_already_read_stage1(conn) == 0
        assert (await _row(conn, item.seed_id))["status"] == "pending"
        await conn.close()
    asyncio.run(run())


def test_reread_refusals(local_pg):
    async def run():
        conn, _ = await db(local_pg)
        made = await queue.enqueue_reading(conn, request("https://example.org/busy"))
        with pytest.raises(op.OperatorActionError) as err:
            await op.reread_read(conn, made["seed_id"])
        assert err.value.code == "stage1_not_done"
        await _read_done(conn, made["seed_id"], stage2="pending")
        with pytest.raises(op.OperatorActionError) as err:
            await op.reread_read(conn, made["seed_id"])
        assert err.value.code == "stage2_not_terminal"
        await conn.execute("UPDATE world_pulse_read_seed SET stage2_status='done' WHERE seed_id=$1", made["seed_id"])
        await _bind(conn, made["seed_id"], 2)
        with pytest.raises(op.OperatorActionError) as err:
            await op.reread_read(conn, made["seed_id"])
        assert err.value.code == "active_durable_binding"
        await conn.execute("UPDATE reading_durable_turn SET consumed_at=now()")
        # Another non-alias row holding the URL (in flight) blocks it.
        await conn.execute(
            "INSERT INTO world_pulse_read_seed (seed_id, kind, run_id, url, status, priority) "
            "VALUES ('finding:other', 'finding', 'r', 'https://example.org/busy', 'pending', 0)")
        with pytest.raises(op.OperatorActionError) as err:
            await op.reread_read(conn, made["seed_id"])
        assert err.value.code == "url_held_by_another_row"
        await conn.close()
    asyncio.run(run())


# ── withdrawal by policy ────────────────────────────────────────────────────


def _proposal(n):
    return SubstrateGraphProposalV1(
        proposal_id=f"reading-p{n}", proposal_kind="relationship_assertion", target_id=f"assertion-{n}",
        actor=READING_ACTOR, subject_node_id=f"concept-s{n}", subject_kind="concept",
        object_node_id=f"concept-o{n}", object_kind="concept", predicate="associated_with",
        statement_key=f"s{n}|associated_with|o{n}|", statement_text="s", anchor_scope="orion",
        authority="local_inferred", recorded_at=NOW)


def _accept(n, policy, seed, at=NOW):
    return SubstrateGraphDecisionV1(
        proposal_id=f"reading-p{n}", proposal_kind="relationship_assertion", target_id=f"assertion-{n}",
        actor=READING_ACTOR, decision_id=f"reading-d{n}", expected_prior_revision=0,
        resulting_state="provisional", policy=policy, authority="local_inferred",
        evidence_refs=[f"reading_text:{'a' * 64}#bytes=0-10", f"reading_seed:{seed}"], recorded_at=at)


async def _journal_db(local_pg):
    conn, _ = await db(local_pg)
    await conn.execute((Path(__file__).resolve().parents[3] / "services/orion-sql-db/manual_migration_substrate_graph_journal_v1.sql").read_text())
    from orion.substrate.graph_journal import _INSERT

    async def append(event):
        return (await conn.execute(_INSERT, *journal_row(event))).endswith(" 1")
    return conn, append


def test_withdraw_targets_only_standing_links_of_the_named_policy(local_pg):
    async def run():
        conn, append = await _journal_db(local_pg)
        for n in range(1, 5):
            await append(_proposal(n))
        await append(_accept(1, MEANING_POLICY, "seed-a"))
        await append(_accept(2, MEANING_POLICY, "seed-b", at=NOW - timedelta(days=2)))
        await append(_accept(3, POLICY, "seed-a"))           # the exact-name rule: not touched
        await append(_accept(4, MEANING_POLICY, "seed-a"))
        # Juniper reviewed 4 later: her decision stands.
        await append(SubstrateGraphDecisionV1(
            proposal_id="reading-p4", proposal_kind="relationship_assertion", target_id="assertion-4",
            actor="operator", decision_id="review-4", expected_prior_revision=1, resulting_state="canonical",
            policy="operator_review", authority="human_verified", recorded_at=NOW + timedelta(hours=1)))

        links = {l.decided.target_id: l for l in await list_policy_links(conn)}
        assert set(links) == {"assertion-1", "assertion-2", "assertion-4"}
        assert links["assertion-4"].status == "superseded" and withdrawal_for(links["assertion-4"], recorded_at=NOW) is None
        scoped = await list_policy_links(conn, since=NOW - timedelta(hours=1), seed_ids=["seed-a"])
        assert [l.decided.target_id for l in scoped] == ["assertion-1", "assertion-4"]

        w = withdrawal_for(links["assertion-1"], recorded_at=NOW, reason="test")
        assert (w.resulting_state, w.expected_prior_revision, w.policy, w.actor) == (
            "rejected", 1, f"{MEANING_POLICY}_withdrawn", WITHDRAW_ACTOR)
        assert await append(w) is True
        assert await append(withdrawal_for(links["assertion-1"], recorded_at=NOW)) is False  # idempotent id
        after = {l.decided.target_id: l.status for l in await list_policy_links(conn)}
        assert after == {"assertion-1": "withdrawn", "assertion-2": "standing", "assertion-4": "superseded"}
        both = await list_policy_links(conn, policies=[MEANING_POLICY, POLICY], seed_ids=["seed-a"])
        assert {l.decided.target_id for l in both} == {"assertion-1", "assertion-3", "assertion-4"}
        await conn.close()
    asyncio.run(run())


def test_withdraw_cli_dry_run_writes_nothing(local_pg):
    async def run():
        import io

        from scripts.reading_meaning_links import parse_args, run as cli_run

        conn, append = await _journal_db(local_pg)
        await append(_proposal(1))
        await append(_accept(1, MEANING_POLICY, "seed-a"))
        written = []

        async def jappend(event):
            written.append(event)
            return await append(event)
        journal = SimpleNamespace(append=jappend)
        out = io.StringIO()
        dry = await cli_run(conn, parse_args(["withdraw"]), journal=journal, out=out)
        assert dry["dry_run"] and dry["withdrawals_written"] == 0 and written == []
        done = await cli_run(conn, parse_args(["withdraw", "--yes"]), journal=journal, out=io.StringIO())
        assert done["withdrawals_written"] == 1 and len(written) == 1
        listed = await cli_run(conn, parse_args(["list"]), journal=journal, out=io.StringIO())
        assert listed["by_status"] == {"withdrawn": 1}
        await conn.close()
    asyncio.run(run())
