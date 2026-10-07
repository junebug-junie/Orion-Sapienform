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


def fresh(kind="request", retryable=True):
    return initial_state("L", {"work_class": "metacog", "kind": kind, "retryable": retryable}, T0)


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
        transition(fresh(), ev("heartbeat"), CFG)


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


def test_non_retryable_failure_ends_instead_of_regranting_a_ghost():
    for failure in ("release_failed", "expire"):
        s = step(fresh(retryable=False), ev("grant", role="metacog"), ev(failure, 1, reason="x"))
        assert s["status"] == "released" and s["reason"].startswith(failure)
    s = step(fresh(retryable=False), ev("grant", role="chat"),
             ev("recall", 1, recall_by=T0.isoformat()), ev("abort", 61))
    assert s["status"] == "released"


def _recalling_hold(retryable=True):
    return step(fresh("hold", retryable=retryable), ev("grant", role="agent"),
                ev("recall", 1, recall_by=(T0 + timedelta(seconds=6)).isoformat(), reason="urgent_preempt"))


def test_urgent_preempt_requeues_a_retryable_hold_in_place_without_spending_an_attempt():
    s = step(_recalling_hold(), ev("abort", 6, reason="urgent_preempt"))
    assert s["status"] == "queued" and s["attempt"] == 1 and s["role"] is None
    assert s["created_at"] == T0.isoformat() and s["queued_since"] == (T0 + timedelta(seconds=6)).isoformat()
    assert s["not_before"] is None and s["recall_by"] is None and s["expires_at"] is None
    assert s["reason"] == "urgent_preempt"
    assert s["history"][-1] == {"event": "abort", "from": "recalling", "status": "queued",
                                "at": (T0 + timedelta(seconds=6)).isoformat(), "role": "agent",
                                "reason": "urgent_preempt", "attempt": 1}
    s = step(s, ev("grant", 20, role="agent"))                 # back in line, re-granted normally
    assert s["status"] == "granted" and s["generation"] == 2 and s["attempt"] == 1


@pytest.mark.parametrize("recall_reason", ["max_hold", "owner_waiting", "card_unlent", "draining"])
def test_a_recalled_hold_aborted_past_its_grace_requeues_in_place_every_time(recall_reason):
    """Live 2026-09-26..28: each abort of a recalled durable-run hold spent a pool attempt, so the
    third recall (every run longer than gpu2's max_hold_sec) dead-lettered the hold and the run saw
    ``unavailable:recall_grace_exceeded``. The pool took the seat back; the hold keeps its place."""
    s = step(fresh("hold"), ev("grant", role="agent-gpu2"))
    for cycle in range(1, 5):
        t = cycle * 1000
        s = step(s, ev("recall", t, recall_by=(T0 + timedelta(seconds=t + 600)).isoformat(), reason=recall_reason),
                 ev("abort", t + 600, reason="recall_grace_exceeded"))
        assert s["status"] == "queued" and s["attempt"] == 1, cycle
        assert s["reason"] == "recall_grace_exceeded" and s["created_at"] == T0.isoformat()
        assert s["role"] is None and s["not_before"] is None and s["expires_at"] is None
        s = step(s, ev("grant", t + 700, role="agent"))
        assert s["status"] == "granted" and s["generation"] == cycle + 1


def test_plain_abort_of_a_request_lease_still_spends_an_attempt():
    s = step(fresh("request"), ev("grant", role="chat"),
             ev("recall", 1, recall_by=(T0 + timedelta(seconds=61)).isoformat(), reason="owner_waiting"),
             ev("abort", 61))
    assert s["status"] == "retry_wait" and s["attempt"] == 2 and s["not_before"] is not None


def test_a_lost_hold_heartbeat_still_spends_a_pool_attempt():
    s = step(fresh("hold"), ev("grant", role="agent"), ev("expire", 91))
    assert s["status"] == "retry_wait" and s["attempt"] == 2


def test_non_retryable_urgent_preempt_ends_like_any_abort():
    s = step(_recalling_hold(retryable=False), ev("abort", 6, reason="urgent_preempt"))
    assert s["status"] == "released" and s["reason"] == "abort:urgent_preempt"


def test_caller_can_finish_a_lease_from_any_waiting_state():
    for pre in ([], [ev("backlog")], [ev("grant", role="m"), ev("release_failed", 1)]):
        s = step(fresh(), *pre, ev("release_ok", 5))
        assert s["status"] == "released", pre


def test_operator_hold_has_no_heartbeat_expiry():
    st = initial_state("H", {"work_class": "experiment", "kind": "hold", "operator": True}, T0)
    s = step(st, ev("grant", role="experiment"))
    assert s["expires_at"] is None
