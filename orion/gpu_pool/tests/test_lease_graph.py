from __future__ import annotations

import asyncio

from datetime import datetime, timedelta, timezone

import pytest

from orion.gpu_pool.config import load_pool_config
from orion.gpu_pool.lease_graph import InvalidTransition, build_lease_graph, initial_state, transition

CFG = load_pool_config()
T0 = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def ev(kind, t=0, **kw):
    return {"type": kind, "at": (T0 + timedelta(seconds=t)).isoformat(), **kw}


def step(state, *events):
    for e in events:
        upd = transition(state, e, CFG)
        hist = state["history"] + upd.pop("history")
        state = {**state, **upd, "history": hist}
    return state


def fresh(kind="request"):
    return initial_state("L", {"work_class": "metacog", "kind": kind}, T0)


def test_grant_then_release_ok_is_final():
    s = step(fresh(), ev("grant", role="metacog"), ev("release_ok", 3))
    assert s["status"] == "released" and s["generation"] == 1 and s["role"] == "metacog"


def test_failures_retry_with_backoff_then_dead_letter():
    s = step(fresh(), ev("grant", role="metacog"), ev("release_failed", 1, reason="upstream_error"))
    assert s["status"] == "retry_wait" and s["attempt"] == 2
    assert datetime.fromisoformat(s["not_before"]) == T0 + timedelta(seconds=1 + CFG.defaults.retry.delay(1))
    s = step(s, ev("requeue", 20), ev("grant", 21, role="fast"), ev("expire", 60))
    assert s["status"] == "retry_wait" and s["attempt"] == 3
    s = step(s, ev("requeue", 80), ev("grant", 81, role="fast"), ev("release_failed", 82, reason="timeout"))
    assert s["status"] == "dead_letter" and s["reason"] == "timeout"


def test_recall_then_abort_retries_and_generation_fences_regrants():
    s = step(fresh(), ev("grant", role="chat"), ev("recall", 1, recall_by=(T0 + timedelta(seconds=61)).isoformat()))
    assert s["status"] == "recalling"
    s = step(s, ev("abort", 61), ev("requeue", 70), ev("grant", 71, role="agent"))
    assert s["status"] == "granted" and s["generation"] == 2 and s["role"] == "agent"


def test_operator_replay_of_dead_letter_resets_attempts():
    s = step(fresh(), ev("backlog"), ev("dead_letter", 9, reason="backlog_max_age"), ev("replay", 10))
    assert s["status"] == "queued" and s["attempt"] == 1 and s["replays"] == 1


def test_heartbeat_extends_by_kind_ttl():
    s = step(fresh("hold"), ev("grant", role="agent"), ev("heartbeat", 50))
    assert datetime.fromisoformat(s["expires_at"]) == T0 + timedelta(seconds=50 + CFG.defaults.hold_lease_ttl_sec)


def test_illegal_transition_is_rejected():
    with pytest.raises(InvalidTransition):
        transition(fresh(), ev("release_ok"), CFG)


def test_history_is_the_walker_path():
    s = step(fresh(), ev("backlog"), ev("requeue", 5), ev("grant", 6, role="metacog"), ev("release_ok", 9))
    assert [h["event"] for h in s["history"]] == ["admit", "backlog", "requeue", "grant", "release_ok"]


def test_graph_survives_restart_via_checkpoint():
    asyncio.run(_restart_scenario())


async def _restart_scenario():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    saver = MemorySaver()
    cfg = {"configurable": {"thread_id": "L"}}
    graph = build_lease_graph(lambda: CFG, saver)
    await graph.ainvoke(fresh(), cfg)
    await graph.ainvoke(Command(resume=ev("backlog")), cfg)

    rebuilt = build_lease_graph(lambda: CFG, saver)  # "restart": same checkpoints, new graph
    snap = await rebuilt.aget_state(cfg)
    assert snap.values["status"] == "backlogged" and snap.next == ("wait",)
    await rebuilt.ainvoke(Command(resume=ev("requeue", 5)), cfg)
    await rebuilt.ainvoke(Command(resume=ev("grant", 6, role="fast")), cfg)
    out = await rebuilt.ainvoke(Command(resume=ev("release_ok", 9)), cfg)
    assert out["status"] == "released"
    assert (await rebuilt.aget_state(cfg)).next == ()
