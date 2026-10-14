"""The reread backfill's whole lifecycle against real SQL, with a stand-in for Hub's workers:
requeue -> worker reads -> settle -> record -> restore a failed read -> report -> rollback.

RUN_READING_POSTGRES=1 python -m pytest services/orion-hub/tests/test_world_pulse_read_reread_backfill_postgres.py -q
"""
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from orion.core.schemas.substrate_graph_journal import SubstrateGraphDecisionV1, SubstrateGraphProposalV1
from orion.schemas.reading import SourceFetchEvidenceV1
from orion.schemas.reading_turn import ReadingRunBriefV1
from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1
from orion.substrate.graph_journal import _INSERT, journal_row
from orion.world_pulse_read import queue
from orion.world_pulse_read.assertions import MEANING_POLICY, READING_ACTOR
from orion.world_pulse_read.durable import bind_turn, consume_turn
from orion.world_pulse_read.fetch_text import text_sha256
from scripts import reading_reread_backfill as bf
from test_reading_postgres import local_pg, request  # noqa: F401 -- fixture reuse

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reading_dns")]

PAGE = "The Falcon ASR model transcribes speech in many languages with a small footprint."
MIGRATION = Path(__file__).resolve().parents[3] / "services/orion-sql-db/manual_migration_substrate_graph_journal_v1.sql"


def _handoff(seed, sha=None):
    return WorldPulseReadHandoffV1(
        seed_ref=seed, what_i_learned="Orion learned something source-backed.", trace_id=str(uuid4()),
        created_at=datetime.now(timezone.utc),
        read_evidence=[SourceFetchEvidenceV1(tool_name="WebFetch", url=seed.url, content_chars=len(PAGE),
                                             content_sha256=sha)])


async def _finish(conn, seed):
    req = await bind_turn(conn, ReadingRunBriefV1(seed_id=seed.seed_id, stage=1, prompt="p", session_id="r",
                                                  timeout_sec=900), str(uuid4()))
    # Live shape: Hub's success path leaves both stages' finished runs unconsumed.
    await bind_turn(conn, ReadingRunBriefV1(seed_id=seed.seed_id, stage=2, prompt="p", session_id="r",
                                            timeout_sec=900), str(uuid4()))
    await queue.mark_seed_done(conn, seed.seed_id, trace_id=str(uuid4()), handoff=_handoff(seed))
    await conn.execute("UPDATE world_pulse_read_seed SET stage2_status='done', stage2_completed_at=now() "
                       "WHERE seed_id=$1", seed.seed_id)


async def _worker(pool, good_seed, stop):
    """Hub's Stage 1 + Stage 2 workers, reduced to the writes they make."""
    while not stop.is_set():
        async with pool.acquire() as conn:
            seed = await queue.claim_next_seed(conn)
            if seed is None:
                await asyncio.sleep(0.02)
                continue
            req = await bind_turn(conn, ReadingRunBriefV1(seed_id=seed.seed_id, stage=1, prompt="p", session_id="r",
                                                          timeout_sec=900), str(uuid4()))
            assert req.brief.prompt == "p" and await conn.fetchval(
                "SELECT attempt FROM reading_durable_turn WHERE run_id=$1", req.run_id) == 1  # never the old run
            if seed.seed_id != good_seed:
                await queue.mark_seed_failed(conn, seed.seed_id, error="turn_error:failed")
                await consume_turn(conn, req.run_id)  # Hub's _fail_seed consumes on failure
                continue
            sha = text_sha256(PAGE)
            await conn.execute("INSERT INTO reading_document_snapshot (sha256, content, content_chars, first_source) "
                               "VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING", sha, PAGE, len(PAGE), seed.url)
            await queue.mark_seed_done(conn, seed.seed_id, trace_id="t-new", handoff=_handoff(seed, sha))
            now = datetime.now(timezone.utc)
            for event in (
                SubstrateGraphProposalV1(
                    proposal_id="reading-p-new", proposal_kind="relationship_assertion", target_id="assertion-new",
                    actor=READING_ACTOR, subject_node_id="concept-falcon-asr", subject_kind="concept",
                    object_node_id="concept-speech-recognition", object_kind="concept", predicate="associated_with",
                    statement_key="k|associated_with|o|", statement_text="s", anchor_scope="orion",
                    authority="local_inferred", recorded_at=now),
                SubstrateGraphDecisionV1(
                    proposal_id="reading-p-new", proposal_kind="relationship_assertion", target_id="assertion-new",
                    actor=READING_ACTOR, decision_id="reading-d-new", expected_prior_revision=0,
                    resulting_state="provisional", policy=MEANING_POLICY, authority="local_inferred",
                    evidence_refs=[f"reading_text:{sha}#bytes=0-9", f"reading_seed:{seed.seed_id}"], recorded_at=now)):
                await conn.execute(_INSERT, *journal_row(event))
            claim = {"subject_id": "concept-falcon-asr", "predicate": "associated_with",
                     "object_id": "concept-speech-recognition", "statement_text": "s", "quote": PAGE[:40],
                     "receipt": {"outcome": "accepted_provisional", "reason": "accepted_meaning_target"}}
            await conn.execute("UPDATE world_pulse_read_seed SET stage2_status='done', stage2_completed_at=now(), "
                               "stage2_result_json=$2::jsonb WHERE seed_id=$1", seed.seed_id,
                               json.dumps({"summary": "s", "relationship_claims": [claim]}))


def test_backfill_rereads_records_restores_reports_and_rolls_back(local_pg, tmp_path, monkeypatch):
    async def run():
        schema = "reading_" + uuid4().hex
        admin = await asyncpg.connect(**local_pg)
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        await admin.close()
        pool = await asyncpg.create_pool(**local_pg, min_size=1, max_size=4, server_settings={"search_path": schema})
        async with pool.acquire() as conn:
            await queue.ensure_seed_queue_schema(conn)
            await conn.execute(MIGRATION.read_text())
            good = await queue.enqueue_reading(conn, request("https://example.org/falcon"))
            bad = await queue.enqueue_reading(conn, request("https://example.org/broken"))
            gone = await queue.enqueue_reading(conn, request("https://example.org/gone"))
            for _ in range(3):
                await _finish(conn, await queue.claim_next_seed(conn))
            rows = {s: bf._row_dict(await conn.fetchrow(bf.SEED_ROW_SQL, s))
                    for s in (good["seed_id"], bad["seed_id"], gone["seed_id"])}
        out = tmp_path / "bf"
        out.mkdir()
        targets = [{"seed_id": s, "url": rows[s]["url"], "probe": "gone" if s == gone["seed_id"] else "fetchable",
                    "http_status": 404 if s == gone["seed_id"] else 200, "before_claims": 0, "before_accepted": 0}
                   for s in rows]
        (out / "targets.json").write_text(json.dumps(targets))
        (out / "snapshot.jsonl").write_text("".join(json.dumps({"seed_id": s, "seed_row": r}, default=str) + "\n"
                                                    for s, r in rows.items()))
        (out / "seeds.txt").write_text("".join(s + "\n" for s in rows))

        async def quiet(*_a, **_k):
            return []
        monkeypatch.setattr(bf, "_pool_quiet", quiet)
        stop = asyncio.Event()
        worker = asyncio.create_task(_worker(pool, good["seed_id"], stop))
        args = bf.parse_args(["run", "--yes", "--skip-deploy-check", "--out", str(out), "--poll-sec", "0.05",
                              "--per-read-timeout-min", "1"])
        assert await bf.cmd_run(pool, args, out) == 0
        stop.set()
        await worker

        results = {r["seed_id"]: r for r in bf._load_jsonl(out / "results.jsonl")}
        assert set(results) == {good["seed_id"], bad["seed_id"]}  # gone is never re-read
        g, b = results[good["seed_id"]], results[bad["seed_id"]]
        assert g["outcome"] == "stage2_done" and g["retained_chars"] == [len(PAGE)]
        assert g["links_meaning_rule"] == 1 and len(g["run_ids"]) == 1
        assert b["outcome"] == "stage1_failed" and b["restored_old_row"] is True
        async with pool.acquire() as conn:
            back = bf._row_dict(await conn.fetchrow(bf.SEED_ROW_SQL, bad["seed_id"]))
            untouched = await conn.fetchrow(bf.SEED_ROW_SQL, gone["seed_id"])
        assert (back["status"], back["stage2_status"], back["handoff_json"]) == (
            "done", "done", rows[bad["seed_id"]]["handoff_json"])
        assert untouched["status"] == "done" and untouched["last_error"] is None
        log = (out / "progress.log").read_text().splitlines()
        assert all(" | eta " in ln and "/h | errors " in ln and "| anomalies " in ln for ln in log)
        assert any("stage2_done" in ln and "meaning_links=1" in ln for ln in log)

        # Every requeue is recorded in_flight before it is waited on, then settled.
        lines = bf._load_jsonl(out / "results.jsonl")
        assert [r["outcome"] for r in lines].count("in_flight") == 2 and len(lines) == 4
        # Resume: nothing left to do, nothing appended.
        assert await bf.cmd_run(pool, args, out) == 0
        assert len(bf._load_jsonl(out / "results.jsonl")) == 4

        assert bf.cmd_report(out) == 0
        table = list(csv.DictReader((out / "before_after.csv").open()))
        assert {r["outcome"] for r in table} == {"stage2_done", "stage1_failed", "skipped_gone"}
        assert "DONE_WITH_ERRORS" in (out / "report.md").read_text()

        dry = bf.parse_args(["rollback", "--out", str(out)])
        assert await bf.cmd_rollback(pool, dry, out) == 0
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM substrate_graph_journal WHERE actor <> $1",
                                       READING_ACTOR) == 0
        real = bf.parse_args(["rollback", "--yes", "--out", str(out)])
        assert await bf.cmd_rollback(pool, real, out) == 0
        async with pool.acquire() as conn:
            state = await conn.fetchval(
                "SELECT payload->>'resulting_state' FROM substrate_graph_journal WHERE event_kind='decision' "
                "AND target_id='assertion-new' ORDER BY revision DESC LIMIT 1")
        assert state == "rejected"
        await pool.close()
    asyncio.run(run())


def test_progress_line_has_every_field_the_protocol_names(tmp_path):
    p = bf.Progress(tmp_path, total=4, done=1)
    p.done = 2
    p.errors, p.anomalies = 1, 2
    line = p.line("requeued reading:x", url="https://e.org")
    for part in ("requeued reading:x", "| 50% |", "| eta ", "| 2/4 |", "/h |", "errors 1", "anomalies 2", "url=https://e.org"):
        assert part in line
    assert (tmp_path / "progress.log").read_text().strip() == line


def test_queue_gate_counts_live_work_but_not_stale_digest_items_or_the_target(local_pg):
    async def run():
        schema = "reading_" + uuid4().hex
        conn = await asyncpg.connect(**local_pg, server_settings={"search_path": schema})
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await queue.ensure_seed_queue_schema(conn)
        target = await queue.enqueue_reading(conn, request("https://example.org/target"))
        assert await conn.fetchval(bf.QUEUE_BUSY_SQL, [target["seed_id"]], 86400.0) == 0
        await conn.execute(
            "INSERT INTO world_pulse_read_seed (seed_id, kind, run_id, url, status, priority, created_at) VALUES "
            "('digest_item:stale', 'digest_item', 'r', 'https://example.org/stale', 'pending', 10, now() - interval '9 days')")
        assert await conn.fetchval(bf.QUEUE_BUSY_SQL, [target["seed_id"]], 86400.0) == 0
        await queue.enqueue_reading(conn, request("https://example.org/live"))
        assert await conn.fetchval(bf.QUEUE_BUSY_SQL, [target["seed_id"]], 86400.0) == 1
        await conn.close()
    asyncio.run(run())


@pytest.mark.parametrize("prior", ["timeout_still_running", "in_flight"])
def test_a_timed_out_read_is_waited_for_on_resume_never_requeued(local_pg, tmp_path, monkeypatch, prior):
    async def run():
        schema = "reading_" + uuid4().hex
        admin = await asyncpg.connect(**local_pg)
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        await admin.close()
        pool = await asyncpg.create_pool(**local_pg, min_size=1, max_size=4, server_settings={"search_path": schema})
        async with pool.acquire() as conn:
            await queue.ensure_seed_queue_schema(conn)
            await conn.execute(MIGRATION.read_text())
            made = await queue.enqueue_reading(conn, request("https://example.org/slow"))
            await _finish(conn, await queue.claim_next_seed(conn))
            snap = {"seed_id": made["seed_id"],
                    "seed_row": bf._row_dict(await conn.fetchrow(bf.SEED_ROW_SQL, made["seed_id"]))}
            from orion.world_pulse_read.operator import reread_read
            await reread_read(conn, made["seed_id"])  # the earlier, timed-out invocation
        out = tmp_path / "bf"
        out.mkdir()
        (out / "targets.json").write_text(json.dumps([{"seed_id": made["seed_id"], "url": "https://example.org/slow",
                                                       "probe": "fetchable", "before_claims": 0, "before_accepted": 0}]))
        (out / "snapshot.jsonl").write_text(json.dumps(snap, default=str) + "\n")
        (out / "results.jsonl").write_text(json.dumps({"seed_id": made["seed_id"], "outcome": prior,
                                                       "started_at": datetime.now(timezone.utc).isoformat()}) + "\n")

        async def quiet(*_a, **_k):
            return []
        monkeypatch.setattr(bf, "_pool_quiet", quiet)
        stop = asyncio.Event()
        worker = asyncio.create_task(_worker(pool, made["seed_id"], stop))
        args = bf.parse_args(["run", "--yes", "--skip-deploy-check", "--out", str(out), "--poll-sec", "0.05",
                              "--per-read-timeout-min", "1"])
        assert await bf.cmd_run(pool, args, out) == 0
        stop.set()
        await worker
        last = bf._load_jsonl(out / "results.jsonl")[-1]
        assert last["outcome"] == "stage2_done"
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM reading_durable_turn WHERE seed_id=$1 AND stage=1",
                                       made["seed_id"]) == 2  # the original + ONE reread attempt
        assert "resume waiting on" in (out / "progress.log").read_text()
        await pool.close()
    asyncio.run(run())


async def _schema_pool(local_pg):
    schema = "reading_" + uuid4().hex
    admin = await asyncpg.connect(**local_pg)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    await admin.close()
    pool = await asyncpg.create_pool(**local_pg, min_size=1, max_size=8, server_settings={"search_path": schema})
    async with pool.acquire() as conn:
        await queue.ensure_seed_queue_schema(conn)
        await conn.execute(MIGRATION.read_text())
    return pool


async def _targets(pool, out, urls):
    rows = {}
    async with pool.acquire() as conn:
        for url in urls:
            made = await queue.enqueue_reading(conn, request(url))
            await _finish(conn, await queue.claim_next_seed(conn))
            rows[made["seed_id"]] = bf._row_dict(await conn.fetchrow(bf.SEED_ROW_SQL, made["seed_id"]))
    out.mkdir()
    (out / "targets.json").write_text(json.dumps([
        {"seed_id": s, "url": r["url"], "probe": "fetchable", "before_claims": 0, "before_accepted": 0}
        for s, r in rows.items()]))
    (out / "snapshot.jsonl").write_text("".join(json.dumps({"seed_id": s, "seed_row": r}, default=str) + "\n"
                                                for s, r in rows.items()))
    return list(rows)


def _quiet(monkeypatch):
    async def quiet(*_a, **_k):
        return []
    monkeypatch.setattr(bf, "_pool_quiet", quiet)
    monkeypatch.setattr(bf, "_pool_busy", quiet)


async def _complete(conn):
    """What Hub's two workers write for one successful re-read of the next claimable seed."""
    seed = await queue.claim_next_seed(conn)
    assert seed is not None
    seed_id = seed.seed_id
    req = await bind_turn(conn, ReadingRunBriefV1(seed_id=seed_id, stage=1, prompt="p", session_id="r",
                                                  timeout_sec=900), str(uuid4()))
    assert await conn.fetchval("SELECT attempt FROM reading_durable_turn WHERE run_id=$1", req.run_id) == 1
    await queue.mark_seed_done(conn, seed_id, trace_id="t-new", handoff=_handoff(seed))
    await conn.execute("UPDATE world_pulse_read_seed SET stage2_status='done', stage2_completed_at=now() "
                       "WHERE seed_id=$1", seed_id)


def test_concurrency_keeps_three_reads_in_flight_never_four(local_pg, tmp_path, monkeypatch):
    """--concurrency 3: three requeued at once before any settles; the fourth starts only
    after one settles; each seed is requeued exactly once."""
    async def run():
        pool = await _schema_pool(local_pg)
        out = tmp_path / "bf"
        ids = await _targets(pool, out, [f"https://example.org/p{i}" for i in range(4)])
        _quiet(monkeypatch)
        stop, peak = asyncio.Event(), []

        async def worker():
            reached = False
            while not stop.is_set():
                async with pool.acquire() as conn:
                    pending = [r["seed_id"] for r in await conn.fetch(
                        "SELECT seed_id FROM world_pulse_read_seed WHERE status='pending' ORDER BY seed_id")]
                    peak.append(len(pending))
                    # Hold every read until three are in flight at once, then drain.
                    reached = reached or len(pending) >= 3
                    if pending and reached:
                        await _complete(conn)
                await asyncio.sleep(0.02)

        w = asyncio.create_task(worker())
        args = bf.parse_args(["run", "--yes", "--skip-deploy-check", "--out", str(out), "--poll-sec", "0.05",
                              "--pause-sec", "0.05", "--per-read-timeout-min", "1", "--concurrency", "3"])
        assert await asyncio.wait_for(bf.cmd_run(pool, args, out), 30) == 0
        stop.set()
        await w
        assert max(peak) == 3
        latest = {r["seed_id"]: r for r in bf._load_jsonl(out / "results.jsonl")}
        assert {latest[s]["outcome"] for s in ids} == {"stage2_done"}
        async with pool.acquire() as conn:
            for s in ids:  # the original + ONE reread attempt, never a double queue
                assert await conn.fetchval("SELECT count(*) FROM reading_durable_turn WHERE seed_id=$1 AND stage=1",
                                           s) == 2
        assert "concurrency=3" in (out / "progress.log").read_text()
        await pool.close()
    asyncio.run(run())


def test_resume_adopts_a_requeue_the_stopped_run_never_recorded(local_pg, tmp_path, monkeypatch):
    """A run stopped after reread_read but before any record (every pre-concurrency run):
    the in-flight seed and a seed whose re-read finished while stopped are waited for and
    recorded, never requeued; the untouched one is read normally."""
    from orion.world_pulse_read.operator import reread_read

    async def run():
        pool = await _schema_pool(local_pg)
        out = tmp_path / "bf"
        flying, finished, fresh = await _targets(
            pool, out, ["https://example.org/a", "https://example.org/b", "https://example.org/c"])
        (out / "run_meta.json").write_text(json.dumps({"run_started_at": datetime.now(timezone.utc).isoformat()}))
        async with pool.acquire() as conn:
            await reread_read(conn, finished)
            await _complete(conn)  # Hub finished it while the backfill was stopped
            await reread_read(conn, flying)
        ts = "2026-10-14T03:01:07+00:00"
        (out / "progress.log").write_text(f"{ts} requeued {flying[:60]} | 0% | eta unknown | 0/3 | 0.0/h | "
                                          "errors 0 | anomalies 0 | url=x consumed_old_runs=2\n")
        _quiet(monkeypatch)
        stop = asyncio.Event()

        async def worker():
            while not stop.is_set():
                async with pool.acquire() as conn:
                    if await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed WHERE status='pending'"):
                        await _complete(conn)
                await asyncio.sleep(0.02)

        w = asyncio.create_task(worker())
        args = bf.parse_args(["run", "--yes", "--skip-deploy-check", "--out", str(out), "--poll-sec", "0.05",
                              "--pause-sec", "0.05", "--per-read-timeout-min", "1", "--concurrency", "3"])
        assert await asyncio.wait_for(bf.cmd_run(pool, args, out), 30) == 0
        stop.set()
        await w
        lines = bf._load_jsonl(out / "results.jsonl")
        latest = {r["seed_id"]: r for r in lines}
        assert {latest[s]["outcome"] for s in (flying, finished, fresh)} == {"stage2_done"}
        # Only the fresh seed was requeued by this run.
        assert [r["seed_id"] for r in lines if r["outcome"] == "in_flight"] == [fresh]
        assert bf._parse_ts(latest[flying]["started_at"]) == bf._parse_ts(ts)  # recovered from progress.log
        async with pool.acquire() as conn:
            for s in (flying, finished, fresh):
                assert await conn.fetchval("SELECT count(*) FROM reading_durable_turn WHERE seed_id=$1 AND stage=1",
                                           s) == 2
        log = (out / "progress.log").read_text()
        assert f"resume waiting on {flying[:60]}" in log and f"resume waiting on {finished[:60]}" in log
        assert "resuming=2" in log
        await pool.close()
    asyncio.run(run())


def test_own_leases_are_not_contention_but_anyone_elses_are():
    own = {"reading-aaa"}
    state = {"leases": [
        {"holder": "durable-runs:reading-aaa", "status": "granted", "priority": "background", "role": "agent"},
        {"holder": "durable-runs:reading-bbb", "status": "dead_letter", "priority": "background"}],
        "queue_depth": {"agent": 1}, "backlog_depth": {}, "cards": [], "roles": []}
    assert bf._foreign_busy(state, own, own_waiting=1) == []
    assert bf._foreign_busy(state, own, own_waiting=0) == ["queue_depth[agent]=1"]
    state["leases"].append({"holder": "orion-mind", "status": "granted", "priority": "interactive",
                            "work_class": "metacog", "role": "agent"})
    assert bf._foreign_busy(state, own, own_waiting=1) == ["granted interactive metacog lease held by orion-mind"]


def test_concurrency_above_one_is_refused_without_parallel_hub(monkeypatch):
    import subprocess as sp

    calls = []

    def fake_run(cmd, **_k):
        calls.append(cmd)
        return sp.CompletedProcess(cmd, 1, "", "")
    monkeypatch.setattr(bf.subprocess, "run", fake_run)
    assert bf.hub_parallel_check(1) == [] and not calls
    assert "max_in_flight" in bf.hub_parallel_check(3)[0]


def test_a_line_cut_off_by_a_kill_does_not_break_resume(tmp_path):
    p = tmp_path / "results.jsonl"
    p.write_text(json.dumps({"seed_id": "a", "outcome": "in_flight"}) + "\n" + '{"seed_id": "b", "outc')
    assert bf._load_jsonl(p) == [{"seed_id": "a", "outcome": "in_flight"}]
