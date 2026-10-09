"""Operator list/detail/cancel/retry/submit against real SQL (disposable cluster).

RUN_READING_POSTGRES=1 python -m pytest services/orion-hub/tests/test_world_pulse_read_operator_postgres.py -q
"""
import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from orion.schemas.reading import SourceFetchEvidenceV1
from orion.schemas.reading_turn import ReadingRunBriefV1
from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1, WorldPulseReadSeedV1
from orion.world_pulse_read import operator as op
from orion.world_pulse_read import queue
from orion.world_pulse_read.durable import OPERATOR_CANCEL_REASON, bind_turn
from orion.world_pulse_read.events import REQUESTED_CHANNEL
from test_reading_postgres import db, local_pg, request  # noqa: F401 -- fixture reuse

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reading_dns")]


def _handoff(seed, *, evidence=True):
    return WorldPulseReadHandoffV1(
        seed_ref=seed, what_i_learned="Orion learned something source-backed.",
        trace_id=str(uuid4()), created_at=datetime.now(timezone.utc),
        read_evidence=[SourceFetchEvidenceV1(tool_name="WebFetch", url=seed.url, content_chars=900)] if evidence else [],
    )


async def _bind(conn, seed_id, stage):
    return await bind_turn(conn, ReadingRunBriefV1(
        seed_id=seed_id, stage=stage, prompt="Saved prompt", session_id="reading", timeout_sec=900,
    ), str(uuid4()))


async def _row(conn, seed_id):
    return await conn.fetchrow("SELECT * FROM world_pulse_read_seed WHERE seed_id=$1", seed_id)


def _run(coro):
    asyncio.run(coro)


async def _operator_db(local_pg):
    conn, schema = await db(local_pg)
    # Match sql-writer's journal_entries columns the operator detail reads.
    await conn.execute(
        "ALTER TABLE journal_entries ADD COLUMN created_at timestamptz NOT NULL DEFAULT now(), "
        "ADD COLUMN title text"
    )
    return conn, schema


def test_list_phases_hide_stale_digest_items_and_carry_outputs(local_pg):
    async def run():
        conn, _ = await _operator_db(local_pg)
        done = await queue.enqueue_reading(conn, request("https://example.org/done"))
        pending = await queue.enqueue_reading(conn, request("https://example.org/pending"))
        seed = await queue.claim_next_seed(conn)
        assert seed.seed_id == done["seed_id"]
        handoff = _handoff(seed)
        await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)
        stale = WorldPulseReadSeedV1(seed_id="digest_item:old", kind="digest_item", run_id="r",
                                     url="https://example.org/stale", title="Old item")
        await queue.enqueue_seeds(conn, [stale])
        await queue.mark_seed_skipped(conn, stale.seed_id, reason=queue.STALE_DIGEST_ITEM_LAST_ERROR)
        await _bind(conn, pending["seed_id"], 1)

        everything = await op.list_reads(conn)
        assert {i["seed_id"] for i in everything["items"]} == {done["seed_id"], pending["seed_id"]}
        assert everything["total"] == 2
        with_stale = await op.list_reads(conn, include_stale=True)
        assert with_stale["total"] == 3

        active = await op.list_reads(conn, phase="active")
        ids = {i["seed_id"]: i for i in active["items"]}
        # Stage 1 done + Stage 2 pending is still in flight.
        assert set(ids) == {done["seed_id"], pending["seed_id"]}
        assert ids[pending["seed_id"]]["active_run_id"].startswith("reading-")
        assert ids[done["seed_id"]]["reading_status"] == "stage1_completed"

        output = (await op.list_reads(conn, phase="with_output"))["items"]
        assert [i["seed_id"] for i in output] == [done["seed_id"]]
        assert output[0]["preview"] == "Orion learned something source-backed."
        assert output[0]["requested_by"] == "juniper"
        assert output[0]["has_handoff"] is True

        assert (await op.list_reads(conn, phase="skipped", include_stale=True))["total"] == 1
        assert (await op.list_reads(conn, kind="digest_item"))["total"] == 0
        with pytest.raises(op.OperatorActionError):
            await op.list_reads(conn, phase="bogus")
        await conn.close()
    _run(run())


def test_detail_marks_rejected_handoff_and_joins_journal_aliases_bindings(local_pg):
    async def run():
        conn, _ = await _operator_db(local_pg)
        first = await queue.enqueue_reading(conn, request("https://example.org/a"))
        alias = await queue.enqueue_reading(conn, request("https://example.org/a"))
        assert alias["duplicate_of"] == first["seed_id"]
        seed = await queue.claim_next_seed(conn)
        handoff = _handoff(seed, evidence=False)
        await _bind(conn, seed.seed_id, 1)
        # Rejected read: the handoff is stored but Stage 1 failed.
        await conn.execute(
            "UPDATE world_pulse_read_seed SET handoff_json=$2::jsonb, trace_id=$3 WHERE seed_id=$1",
            seed.seed_id, handoff.model_dump_json(), handoff.trace_id,
        )
        await queue.mark_seed_failed(conn, seed.seed_id, error="no_read_evidence")
        await conn.execute(
            "INSERT INTO journal_entries(entry_id, source_ref, body) VALUES ($1, $2, 'Journal body'), ('other', 'unrelated', 'x')",
            "j1", f"world_pulse_read:{handoff.trace_id}",
        )
        detail = await op.read_detail(conn, seed.seed_id)
        assert detail["handoff"]["what_i_learned"].startswith("Orion learned")
        assert detail["handoff_accepted"] is False
        assert detail["reading_status"] == "failed"
        assert detail["last_error"] == "no_read_evidence"
        assert [j["body"] for j in detail["journal"]] == ["Journal body"]
        assert [a["seed_id"] for a in detail["aliases"]] == [alias["seed_id"]]
        assert detail["durable_turns"][0]["stage"] == 1
        assert detail["request"]["invocation_context"] == "unified_chat"
        listed = {i["seed_id"]: i for i in (await op.list_reads(conn))["items"]}
        # A rejected write-up is never quoted in the list as if it were learned.
        assert listed[seed.seed_id]["has_handoff"] is True
        assert listed[seed.seed_id]["preview"] == ""
        with pytest.raises(op.OperatorActionError) as err:
            await op.read_detail(conn, "missing")
        assert err.value.http_status == 404
        await conn.close()
    _run(run())


def test_detail_survives_missing_journal_table(local_pg):
    async def run():
        conn, _ = await _operator_db(local_pg)
        await conn.execute("DROP TABLE journal_entries")
        made = await queue.enqueue_reading(conn, request())
        seed = await queue.claim_next_seed(conn)
        handoff = _handoff(seed)
        await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)
        detail = await op.read_detail(conn, made["seed_id"])
        assert detail["journal"] == []
        assert detail["handoff_accepted"] is True
        await conn.close()
    _run(run())


def test_cancel_pending_unbound_skips_without_charge(local_pg):
    async def run():
        conn, _ = await _operator_db(local_pg)
        made = await queue.enqueue_reading(conn, request())
        assert await op.cancel_read(conn, made["seed_id"]) == {"action": "skipped", "stage": 1}
        row = await _row(conn, made["seed_id"])
        assert (row["status"], row["last_error"], row["attempts"]) == ("skipped", OPERATOR_CANCEL_REASON, 0)
        with pytest.raises(op.OperatorActionError) as err:
            await op.cancel_read(conn, made["seed_id"])
        assert err.value.code == "not_active"
        await conn.close()
    _run(run())


def test_cancel_bound_returns_durable_plan_and_leaves_row(local_pg):
    async def run():
        conn, _ = await _operator_db(local_pg)
        made = await queue.enqueue_reading(conn, request())
        await queue.claim_next_seed(conn)
        binding = await _bind(conn, made["seed_id"], 1)
        plan = await op.cancel_read(conn, made["seed_id"])
        assert plan == {"action": "cancel_durable_run", "stage": 1, "run_id": binding.run_id}
        assert (await _row(conn, made["seed_id"]))["status"] == "claimed"
        await conn.close()
    _run(run())


def test_cancel_claimed_unbound_is_refused(local_pg):
    async def run():
        conn, _ = await _operator_db(local_pg)
        made = await queue.enqueue_reading(conn, request())
        await queue.claim_next_seed(conn)
        with pytest.raises(op.OperatorActionError) as err:
            await op.cancel_read(conn, made["seed_id"])
        assert err.value.code == "claimed_without_binding_retry_shortly"
        assert (await _row(conn, made["seed_id"]))["status"] == "claimed"
        await conn.close()
    _run(run())


def test_cancel_stage2_pending_and_alias_refusal(local_pg):
    async def run():
        conn, _ = await _operator_db(local_pg)
        made = await queue.enqueue_reading(conn, request("https://example.org/s2"))
        alias = await queue.enqueue_reading(conn, request("https://example.org/s2"))
        seed = await queue.claim_next_seed(conn)
        handoff = _handoff(seed)
        await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)
        with pytest.raises(op.OperatorActionError) as err:
            await op.cancel_read(conn, alias["seed_id"])
        assert err.value.code == "alias_row_act_on_target"
        assert await op.cancel_read(conn, made["seed_id"]) == {"action": "skipped", "stage": 2}
        row = await _row(conn, made["seed_id"])
        assert (row["status"], row["stage2_status"], row["stage2_error"]) == ("done", "skipped", OPERATOR_CANCEL_REASON)
        await conn.close()
    _run(run())


def test_retry_stage1_resets_budget_and_guards(local_pg):
    async def run():
        conn, _ = await _operator_db(local_pg)
        made = await queue.enqueue_reading(conn, request("https://example.org/r"))
        seed = await queue.claim_next_seed(conn)
        await queue.mark_seed_failed(conn, seed.seed_id, error="no_read_evidence")
        with pytest.raises(op.OperatorActionError) as err:
            await op.retry_read(conn, made["seed_id"], stage=2)
        assert err.value.code == "stage1_not_done"
        # Another active row for the same URL blocks a duplicate read.
        other = await queue.enqueue_reading(conn, request("https://example.org/r"))
        assert other["duplicate_of"] is None
        with pytest.raises(op.OperatorActionError) as err:
            await op.retry_read(conn, made["seed_id"], stage=1)
        assert err.value.code == "url_already_active"
        await op.cancel_read(conn, other["seed_id"])
        assert await op.retry_read(conn, made["seed_id"], stage=1) == {"action": "requeued", "stage": 1}
        row = await _row(conn, made["seed_id"])
        assert (row["status"], row["attempts"], row["last_error"]) == ("pending", 0, None)
        with pytest.raises(op.OperatorActionError) as err:
            await op.retry_read(conn, made["seed_id"], stage=1)
        assert err.value.code == "stage1_not_terminal"
        await conn.close()
    _run(run())


def test_retry_refuses_unconsumed_binding_and_stale_digest(local_pg):
    async def run():
        conn, _ = await _operator_db(local_pg)
        made = await queue.enqueue_reading(conn, request())
        await queue.claim_next_seed(conn)
        await _bind(conn, made["seed_id"], 1)
        await conn.execute("UPDATE world_pulse_read_seed SET status='failed' WHERE seed_id=$1", made["seed_id"])
        with pytest.raises(op.OperatorActionError) as err:
            await op.retry_read(conn, made["seed_id"], stage=1)
        assert err.value.code == "active_durable_binding"

        stale = WorldPulseReadSeedV1(seed_id="digest_item:s", kind="digest_item", run_id="r",
                                     url="https://example.org/old-digest")
        await queue.enqueue_seeds(conn, [stale])
        await queue.mark_seed_skipped(conn, stale.seed_id, reason=queue.STALE_DIGEST_ITEM_LAST_ERROR)
        with pytest.raises(op.OperatorActionError) as err:
            await op.retry_read(conn, stale.seed_id, stage=1)
        assert err.value.code == "stale_digest_item_would_be_reskipped"
        await conn.close()
    _run(run())


def test_retry_refuses_failed_digest_item_past_the_stale_sweep_age(local_pg):
    """A failed (not swept) old digest item would be re-skipped by the next tick's sweep."""
    max_age = 5 * 86400.0

    async def run():
        conn, _ = await _operator_db(local_pg)
        old = WorldPulseReadSeedV1(seed_id="digest_item:old-failed", kind="digest_item", run_id="r",
                                   url="https://example.org/old-failed")
        young = WorldPulseReadSeedV1(seed_id="digest_item:young-failed", kind="digest_item", run_id="r",
                                     url="https://example.org/young-failed")
        await queue.enqueue_seeds(conn, [old, young])
        await conn.execute(
            "UPDATE world_pulse_read_seed SET created_at = now() - interval '6 days' WHERE seed_id=$1",
            old.seed_id,
        )
        for sid in (old.seed_id, young.seed_id):
            await queue.mark_seed_failed(conn, sid, error="fetch_failed")

        with pytest.raises(op.OperatorActionError) as err:
            await op.retry_read(conn, old.seed_id, stage=1, digest_item_max_age_sec=max_age)
        assert err.value.code == "stale_digest_item_would_be_reskipped"
        assert (await _row(conn, old.seed_id))["status"] == "failed"

        assert await op.retry_read(conn, young.seed_id, stage=1, digest_item_max_age_sec=max_age) == {
            "action": "requeued", "stage": 1,
        }
        assert await queue.skip_stale_digest_items(conn, max_age_sec=max_age) == 0
        assert (await _row(conn, young.seed_id))["status"] == "pending"
        await conn.close()
    _run(run())


@pytest.mark.parametrize("evidence", [True, False])
def test_retry_stage2_requires_read_evidence(local_pg, evidence):
    async def run():
        conn, _ = await _operator_db(local_pg)
        made = await queue.enqueue_reading(conn, request())
        seed = await queue.claim_next_seed(conn)
        handoff = _handoff(seed, evidence=evidence)
        await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)
        await queue.mark_stage2_failed(conn, seed.seed_id, error="bad_json")
        if not evidence:
            with pytest.raises(op.OperatorActionError) as err:
                await op.retry_read(conn, made["seed_id"], stage=2)
            assert err.value.code == "no_read_evidence"
        else:
            assert await op.retry_read(conn, made["seed_id"], stage=2) == {"action": "requeued", "stage": 2}
            row = await _row(conn, made["seed_id"])
            assert (row["status"], row["stage2_status"], row["stage2_attempts"]) == ("done", "pending", 0)
        await conn.close()
    _run(run())


def test_submit_uses_ingress_with_operator_provenance(local_pg):
    class Bus:
        def __init__(self):
            self.published = []

        async def publish(self, channel, envelope):
            self.published.append((channel, envelope))

    async def run():
        conn, _ = await _operator_db(local_pg)
        bus = Bus()
        receipt = await op.submit_read(conn, url="https://example.org/paper#frag", why_now="Check it", bus=bus)
        assert receipt["status"] == "queued"
        row = await _row(conn, receipt["seed_id"])
        stored = json.loads(row["request_json"])
        assert (stored["requested_by"], stored["invocation_context"]) == ("juniper", "operator")
        assert row["url"] == "https://example.org/paper"
        assert [c for c, _ in bus.published] == [REQUESTED_CHANNEL]
        with pytest.raises(ValueError):
            await op.submit_read(conn, url="http://127.0.0.1/admin")
        await conn.close()
    _run(run())
