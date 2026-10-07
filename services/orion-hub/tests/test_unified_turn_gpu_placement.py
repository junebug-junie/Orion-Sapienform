"""The unified turn tells the situation brief which model it runs on from its GPU pool lease.

Spec docs/superpowers/specs/2026-09-24-gpu-pool-design.md, reader impacts item 5: under the pool a
route's /routes default is not where a spilled call runs, so a turn holding a lease (a durable
run's hold, ``payload["gpu_lease"]``) must hand the brief its granted role and the model the pool
discovered there, read from Hub's own live pool feed.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import orion.hub.turn_orchestrator as turn_orchestrator


def _run(payload):
    captured = {}

    async def _fake_build(ctx, runtime_ns, **_kwargs):
        captured["ctx"] = ctx
        return None, {"compact_text": "Situation: stub"}

    orig = turn_orchestrator.build_situation_for_ctx
    turn_orchestrator.build_situation_for_ctx = _fake_build
    try:
        asyncio.run(turn_orchestrator._build_situation_prompt_fragment(
            session_id="s1", user_message="hello", payload=payload, settings=SimpleNamespace(),
            correlation_id="corr-1",
        ))
    finally:
        turn_orchestrator.build_situation_for_ctx = orig
    return captured["ctx"]


def _state(age_sec: float = 1.0):
    generated = datetime.now(timezone.utc) - timedelta(seconds=age_sec)
    return {"generated_at": generated.isoformat(), "roles": [
        {"role": "agent", "kind": "llm", "cards": ["gpu1"], "url": "http://x:8015", "status": "confirmed",
         "profile_name": "agent-flex", "model_file": "Agent-27B.gguf"},
        {"role": "chat", "kind": "llm", "cards": ["gpu0"], "url": "http://x:8011", "status": "confirmed",
         "profile_name": "chat-deep", "model_file": "Chat-35B.gguf"},
    ]}


def test_spilled_hold_hands_the_brief_the_granted_roles_model(monkeypatch):
    from scripts import gpu_pool_routes

    monkeypatch.setattr(gpu_pool_routes.feed, "state", _state())
    # An agent-class run whose hold was granted chat (spill to gpu0).
    ctx = _run({"gpu_lease": {"lease_id": "L", "generation": 1, "role": "chat", "holder": "durable-runs:r"}})
    assert ctx["gpu_placement"] == {"role": "chat", "model": "Chat-35B.gguf", "profile": "chat-deep",
                                    "status": "confirmed"}
    assert "runtime_line_owner" not in ctx


def test_hold_before_the_feed_has_state_names_the_role_only(monkeypatch):
    from scripts import gpu_pool_routes

    monkeypatch.setattr(gpu_pool_routes.feed, "state", None)
    ctx = _run({"gpu_lease": {"lease_id": "L", "generation": 1, "role": "agent", "holder": "durable-runs:r"}})
    assert ctx["gpu_placement"] == {"role": "agent", "model": None, "profile": None, "status": None}


def test_stale_pool_snapshot_never_names_a_model(monkeypatch):
    """The feed keeps its last snapshot when the pool goes quiet; an old one is not evidence."""
    from scripts import gpu_pool_routes

    monkeypatch.setattr(gpu_pool_routes.feed, "state", _state(age_sec=3600))
    ctx = _run({"gpu_lease": {"lease_id": "L", "generation": 1, "role": "chat", "holder": "durable-runs:r"}})
    assert ctx["gpu_placement"]["model"] is None


def test_no_hold_hands_the_model_line_to_the_harness():
    """Without a lease the harness prompt states the default of the route the motor really uses;
    the brief must not add a second default for ORION_SITUATION_RUNTIME_ROUTE next to it."""
    for payload in ({}, {"gpu_lease": None}):
        ctx = _run(payload)
        assert "gpu_placement" not in ctx
        assert ctx["runtime_line_owner"] == "harness"
