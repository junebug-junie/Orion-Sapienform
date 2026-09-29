"""The unified turn tells the situation brief which model it runs on from its GPU pool lease.

Spec docs/superpowers/specs/2026-09-24-gpu-pool-design.md, reader impacts item 5: under the pool a
route's /routes default is not where a spilled call runs, so a turn holding a lease (a durable
run's hold, ``payload["gpu_lease"]``) must hand the brief its granted role and the model the pool
discovered there, read from Hub's own live pool feed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import orion.hub.turn_orchestrator as turn_orchestrator


def _run(payload):
    captured = {}

    async def _fake_build(ctx, runtime_ns):
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


def _state():
    return {"roles": [
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
    assert ctx["gpu_placement"] == {"role": "chat", "model": "Chat-35B.gguf", "profile": "chat-deep"}


def test_hold_before_the_feed_has_state_names_the_role_only(monkeypatch):
    from scripts import gpu_pool_routes

    monkeypatch.setattr(gpu_pool_routes.feed, "state", None)
    ctx = _run({"gpu_lease": {"lease_id": "L", "generation": 1, "role": "agent", "holder": "durable-runs:r"}})
    assert ctx["gpu_placement"] == {"role": "agent", "model": None, "profile": None}


def test_no_hold_leaves_the_brief_on_the_route_default():
    assert "gpu_placement" not in _run({})
    assert "gpu_placement" not in _run({"gpu_lease": None})
