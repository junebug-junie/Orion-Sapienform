"""Checkpoint pruning: keep what LangGraph needs to resume, drop the rest.

Pure tests run everywhere. The lifecycle tests run on REAL Postgres (ORION_ADMISSION_TEST_DSN; skipped
without it): a real ``PostgresSaver`` writes real history, the prune runs, and the same threads must
read back the same state and still resume -- including a thread paused on ``interrupt()`` whose
pending write lives on its newest checkpoint.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import TypedDict
from urllib.parse import quote

import pytest

from app.checkpoint_prune import Plan, _interval, thread_family

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "scripts"))
import prune_checkpoints  # noqa: E402

DSN = os.environ.get("ORION_ADMISSION_TEST_DSN")


# --- pure ---------------------------------------------------------------------------------------


def test_thread_family_strips_run_ids_and_dates():
    assert thread_family("temporal_self:orion:2026-11-11") == "temporal_self:orion"
    assert thread_family("dream-carry-5c22ff218ee85da29dc125bc78758eb1") == "dream-carry"
    assert thread_family("situation:juniper:2026-11-12") == "situation:juniper"
    assert thread_family("02e4bddcb536") == "<bare run id>"


def test_plan_totals_and_only_threads_with_work_are_pruned():
    rows = [
        {"thread_id": "a", "ckpt_total": 5, "ckpt_del": 4, "writes_total": 3, "writes_del": 3,
         "blobs_total": 10, "blobs_del": 6, "blob_bytes_del": 600},
        {"thread_id": "b", "ckpt_total": 1, "ckpt_del": 0, "writes_total": 0, "writes_del": 0,
         "blobs_total": 2, "blobs_del": 0, "blob_bytes_del": 0},
        {"thread_id": "c", "ckpt_total": 1, "ckpt_del": 0, "writes_total": 2, "writes_del": 0,
         "blobs_total": 3, "blobs_del": 1, "blob_bytes_del": 9},
    ]
    plan = Plan(rows=rows)
    assert plan.threads_to_prune == ["a", "c"]
    t = plan.totals()
    assert (t["ckpt_del"], t["writes_del"], t["blobs_del"], t["threads"], t["threads_to_prune"]) == (4, 3, 7, 3, 2)
    assert plan.largest_threads(1)[0]["thread_id"] == "a"


def test_like_prefix_escapes_wildcards():
    assert prune_checkpoints.like_prefix("temporal_self:") == "temporal\\_self:%"
    assert prune_checkpoints.like_prefix("a%b") == "a\\%b%"


def test_timeouts_are_validated_before_reaching_sql():
    assert _interval("5s") == "5s"
    with pytest.raises(ValueError):
        _interval("5s'; drop table checkpoints; --")


def test_cli_without_dsn_refuses(monkeypatch, capsys):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    assert prune_checkpoints.main([]) == 2


# --- real Postgres --------------------------------------------------------------------------------


# Non-primitive values on purpose: the saver inlines str/int/float/bool in the checkpoint row and
# only writes dicts/lists to checkpoint_blobs -- the table this prune rule is about.
class _S(TypedDict, total=False):
    tick: dict
    counter: dict
    constant: list
    answer: str


def _graphs(saver):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def bump(s: _S) -> dict:
        return {"counter": {"n": int((s.get("counter") or {}).get("n", 0)) + 1}}

    def ask(s: _S) -> dict:
        return {"answer": interrupt("who?")}

    a = StateGraph(_S)
    a.add_node("bump", bump)
    a.add_edge(START, "bump")
    a.add_edge("bump", END)
    b = StateGraph(_S)
    b.add_node("bump", bump)
    b.add_node("ask", ask)
    b.add_edge(START, "bump")
    b.add_edge("bump", "ask")
    b.add_edge("ask", END)
    return a.compile(checkpointer=saver), b.compile(checkpointer=saver)


@pytest.fixture()
def pg():
    if not DSN:
        pytest.skip("ORION_ADMISSION_TEST_DSN not set")
    import psycopg
    from psycopg.rows import dict_row
    from langgraph.checkpoint.postgres import PostgresSaver

    schema = f"ckprune_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in DSN else "?"
    dsn = f"{DSN}{sep}options={quote(f'-csearch_path={schema}')}"
    conn = psycopg.connect(dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    saver = PostgresSaver(conn)
    saver.setup()
    try:
        yield saver, conn, dsn
    finally:
        conn.close()
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(f"DROP SCHEMA {schema} CASCADE")


def _cfg(t):
    return {"configurable": {"thread_id": t}}


def _counts(conn, thread):
    q = lambda table: conn.execute(f"SELECT count(*) AS n FROM {table} WHERE thread_id = %s", (thread,)).fetchone()["n"]
    return q("checkpoints"), q("checkpoint_writes"), q("checkpoint_blobs")


def _seed_history(saver, conn):
    a, b = _graphs(saver)
    a.invoke({"constant": ["kept"], "tick": {"i": 0}, "counter": {"n": 0}}, _cfg("temporal_self:orion:2026-11-11"))
    for i in range(1, 6):
        a.invoke({"tick": {"i": i}}, _cfg("temporal_self:orion:2026-11-11"))
    b.invoke({"constant": ["c"], "tick": {"i": 0}, "counter": {"n": 0}}, _cfg("reading-abc12345"))   # pauses on interrupt
    return a, b


def test_prune_keeps_state_and_resume_on_real_postgres(pg, capsys):
    from langgraph.types import Command

    from app.checkpoint_prune import apply_plan, build_plan

    saver, conn, dsn = pg
    a, b = _seed_history(saver, conn)
    ta, tb = "temporal_self:orion:2026-11-11", "reading-abc12345"
    before_a, before_b = a.get_state(_cfg(ta)), b.get_state(_cfg(tb))
    assert before_a.values["counter"] == {"n": 6} and before_a.values["constant"] == ["kept"]
    assert before_b.next == ("ask",) and before_b.tasks[0].interrupts
    # an in-flight writer's newer blob and a version this job cannot parse: both must survive
    conn.execute("INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob) "
                 "VALUES (%s, '', 'counter', %s, 'empty', NULL), (%s, '', 'counter', 'weird', 'empty', NULL)",
                 (ta, "9" * 32 + ".0.1", ta))
    ck_a, wr_a, bl_a = _counts(conn, ta)
    assert ck_a > 6 and bl_a > 6

    # a pending write for a checkpoint not yet committed (async durability: task writes land before
    # their checkpoint row): its id is above the newest, so it must survive
    newest_id = conn.execute("SELECT max(checkpoint_id) AS m FROM checkpoints WHERE thread_id = %s", (ta,)).fetchone()["m"]
    racing_id = newest_id + "z"   # sorts after the newest, like the next uuid6
    conn.execute("INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob,"
                 " task_path) VALUES (%s, '', %s, 'racing-task', 0, 'counter', 'empty', ''::bytea, '')", (ta, racing_id))
    ck_a, wr_a, bl_a = _counts(conn, ta)

    # dry run (the CLI default) deletes nothing and reports the work
    assert prune_checkpoints.main(["--dsn", dsn, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry_run" and "deleted" not in report
    assert report["totals"]["blobs_del"] > 0 and report["totals"]["writes_del"] > 0
    assert report["totals"]["ckpt_del"] == ck_a - 1 + _counts(conn, tb)[0] - 1
    assert _counts(conn, ta) == (ck_a, wr_a, bl_a)

    # apply
    conn.autocommit = False
    plan = build_plan(conn)
    conn.commit()
    deleted = apply_plan(conn, plan, batch_threads=1)
    conn.autocommit = True
    assert deleted["checkpoints"] == report["totals"]["ckpt_del"]
    assert deleted["blobs"] == report["totals"]["blobs_del"] and deleted["batches"] == 2
    assert deleted["writes"] == report["totals"]["writes_del"]
    assert conn.execute("SELECT count(*) AS n FROM checkpoint_writes WHERE task_id = 'racing-task'").fetchone()["n"] == 1
    conn.execute("DELETE FROM checkpoint_writes WHERE task_id = 'racing-task'")

    assert _counts(conn, ta)[0] == 1 and _counts(conn, tb)[0] == 1
    versions = {r["version"] for r in conn.execute(
        "SELECT version FROM checkpoint_blobs WHERE thread_id = %s AND channel = 'counter'", (ta,)).fetchall()}
    assert "9" * 32 + ".0.1" in versions and "weird" in versions
    conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s AND version IN (%s, 'weird')", (ta, "9" * 32 + ".0.1"))

    after_a, after_b = a.get_state(_cfg(ta)), b.get_state(_cfg(tb))
    assert after_a.values == before_a.values          # 'constant' was written once, long ago: kept
    assert after_b.values == before_b.values and after_b.next == ("ask",)
    assert after_b.tasks[0].interrupts[0].value == "who?"   # pending write on the newest checkpoint kept

    # both threads still run: A continues its count, B resumes past its interrupt
    assert a.invoke({"tick": {"i": 99}}, _cfg(ta))["counter"] == {"n": 7}
    assert b.invoke(Command(resume="juniper"), _cfg(tb))["answer"] == "juniper"

    # idempotent: a second plan right after an apply finds nothing older than the newest... except
    # what the two runs above just wrote
    conn.autocommit = False
    again = build_plan(conn)
    conn.commit()
    conn.autocommit = True
    assert set(again.threads_to_prune) <= {ta, tb}


def test_prefix_limits_the_prune_to_matching_threads(pg):
    from app.checkpoint_prune import apply_plan, build_plan

    saver, conn, _ = pg
    _seed_history(saver, conn)
    other_before = _counts(conn, "reading-abc12345")
    prefix = prune_checkpoints.like_prefix("temporal_self:")
    conn.autocommit = False
    plan = build_plan(conn, prefix=prefix)
    conn.commit()
    assert plan.threads_to_prune == ["temporal_self:orion:2026-11-11"]
    apply_plan(conn, plan, prefix=prefix)
    conn.autocommit = True
    assert _counts(conn, "temporal_self:orion:2026-11-11")[0] == 1
    assert _counts(conn, "reading-abc12345") == other_before


# --- the in-service poller -----------------------------------------------------------------------


def test_poller_dry_run_reports_and_never_applies():
    import asyncio

    from app.checkpoint_prune import PrunePoller

    calls = []

    def fake(dsn, *, apply, **_):
        calls.append(apply)
        return {"status": "ok", "mode": "apply" if apply else "dry_run",
                "totals": {"threads": 3, "ckpt_del": 7, "blobs_del": 9}}

    p = PrunePoller(dsn="x", mode="dry_run", interval_sec=60, runner=fake)
    asyncio.run(p.tick())
    assert calls == [False]
    h = p.health()
    assert h["mode"] == "dry_run" and h["passes"] == 1 and h["last"]["totals"]["ckpt_del"] == 7
    assert h["last"]["deleted"] is None


def test_poller_apply_mode_passes_apply_and_off_does_nothing():
    import asyncio

    from app.checkpoint_prune import PrunePoller

    calls = []
    fake = lambda dsn, *, apply, **_: calls.append(apply) or {"status": "ok", "totals": {}, "deleted": {"checkpoints": 1}}
    asyncio.run(PrunePoller(dsn="x", mode="apply", interval_sec=60, runner=fake).tick())
    asyncio.run(PrunePoller(dsn="x", mode="off", interval_sec=60, runner=fake).tick())
    assert calls == [True]
    with pytest.raises(ValueError):
        PrunePoller(dsn="x", mode="delete_everything", interval_sec=60)


def test_poller_survives_a_failed_pass():
    import asyncio

    from app.checkpoint_prune import PrunePoller

    def boom(dsn, *, apply, **_):
        raise RuntimeError("db away")

    p = PrunePoller(dsn="x", mode="dry_run", interval_sec=60, runner=boom)
    assert asyncio.run(p.tick()) is None
    assert "db away" in p.health()["last_error"] and p.passes == 0


def test_poller_loop_waits_first_delay_and_stops():
    import asyncio

    from app.checkpoint_prune import PrunePoller

    calls = []

    async def go():
        stop = asyncio.Event()
        p = PrunePoller(dsn="x", mode="dry_run", interval_sec=0.01, first_delay_sec=0.01,
                        runner=lambda dsn, *, apply, **_: calls.append(1) or {"status": "ok", "totals": {}})
        task = asyncio.create_task(p.run_forever(stop))
        await asyncio.sleep(0.08)
        stop.set()
        await asyncio.wait_for(task, 1)

    asyncio.run(go())
    assert len(calls) >= 2


def test_settings_ship_dry_run(monkeypatch):
    from app.settings import Settings

    monkeypatch.setenv("POSTGRES_URI", "postgresql://x")
    monkeypatch.delenv("DURABLE_RUNS_CHECKPOINT_PRUNE_MODE", raising=False)
    s = Settings()
    assert s.checkpoint_prune_mode == "dry_run" and s.checkpoint_prune_interval_sec == 3600.0
    example = (SERVICE_ROOT / ".env_example").read_text()
    assert "DURABLE_RUNS_CHECKPOINT_PRUNE_MODE=dry_run" in example
    compose = (SERVICE_ROOT / "docker-compose.yml").read_text()
    assert "DURABLE_RUNS_CHECKPOINT_PRUNE_MODE:-dry_run" in compose


def test_update_state_threads_read_back_and_resume_after_prune(pg):
    """admission_runtime's own write path: aupdate_state(..., as_node=...) checkpoints."""
    from langgraph.types import Command

    from app.checkpoint_prune import apply_plan, build_plan

    saver, conn, _ = pg
    _a, b = _graphs(saver)
    tb = "reverie-visual-0123456789abcdef"
    b.invoke({"constant": ["c"], "tick": {"i": 0}, "counter": {"n": 0}}, _cfg(tb))      # paused on interrupt
    b.update_state(_cfg(tb), {"counter": {"n": 40}}, as_node="bump")
    b.update_state(_cfg(tb), {"tick": {"i": 7}})
    before = b.get_state(_cfg(tb))
    conn.autocommit = False
    plan = build_plan(conn)
    conn.commit()
    apply_plan(conn, plan)
    conn.autocommit = True
    assert _counts(conn, tb)[0] == 1
    after = b.get_state(_cfg(tb))
    assert after.values == before.values and after.next == before.next
    out = b.invoke(Command(resume="juniper"), _cfg(tb))
    assert out["counter"] == {"n": 40} and out["answer"] == "juniper"


def test_apply_stops_between_batches_when_asked(pg):
    from app.checkpoint_prune import apply_plan, build_plan

    saver, conn, _ = pg
    _seed_history(saver, conn)
    conn.autocommit = False
    plan = build_plan(conn)
    conn.commit()
    asked = iter([False, True])
    deleted = apply_plan(conn, plan, batch_threads=1, should_stop=lambda: next(asked))
    conn.autocommit = True
    assert deleted["batches"] == 1 and deleted["stopped"] is True


def test_no_durable_runs_graph_uses_a_delta_channel():
    """A DeltaChannel's value is rebuilt from parent checkpoints and their writes -- exactly what the
    prune deletes. If a graph adopts one (e.g. add_messages / MessagesState), prune must change first."""
    hits = []
    for path in (SERVICE_ROOT / "app").rglob("*.py"):
        if path.name == "checkpoint_prune.py":
            continue
        text = path.read_text()
        for needle in ("DeltaChannel", "add_messages", "MessagesState"):
            if needle in text:
                hits.append(f"{path.name}: {needle}")
    assert hits == []
