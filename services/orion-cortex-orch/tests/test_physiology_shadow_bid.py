"""Body economy Patch C3a: cortex-orch bids for a tier 1/2 verb before dispatch, in SHADOW.

* Juniper's turn never bids (cortex-gateway ingress, chat verbs / the pinned chat lane, interactive
  or urgent priority) -- checked on the pure rule AND end to end through ``handle``;
* a verb called from inside a durable run never bids again (the run bid at admission);
* a self-initiated tier 1/2 verb bids with its ``physiology`` block, and the dispatch is unchanged
  whatever the decision says (funded, deferred, or no answer); the reservation is released after.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app import main as orch_main
from app import physiology_bid as pb
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs.models import VerbResultV1
from orion.physiology import client
from orion.schemas.cortex.contracts import CortexClientRequest
from orion.schemas.physiology_want import WantDecisionV1

CORR = "22222222-2222-2222-2222-222222222222"


def _payload(verb, *, mode="brain", options=None):
    return {"mode": mode, "verb": verb, "route_intent": "none", "packs": [], "options": options or {},
            "recall": {"enabled": False, "required": False, "mode": "hybrid", "profile": None},
            "context": {"messages": [{"role": "user", "content": "x"}], "raw_user_text": "x", "user_message": "x",
                        "metadata": {}}}


def _req(verb, **kw) -> CortexClientRequest:
    return CortexClientRequest.model_validate(_payload(verb, **kw))


@pytest.mark.parametrize("source,verb,options,reason", [
    ("cortex-gateway", "dream_cycle", {}, "turn_ingress"),                    # Juniper's chat ingress
    ("actions", "chat_general", {}, "interactive_lane"),                      # a chat verb
    ("actions", "dream_cycle", {"execution_lane": "chat"}, "interactive_lane"),
    ("actions", "dream_cycle", {"priority": "interactive"}, "interactive_priority"),
    ("actions", "dream_cycle", {"priority": "urgent"}, "interactive_priority"),
    ("orion-durable-runs", "journal.compose", {}, "inside_durable_run"),      # already bid at admission
    # cortex-exec's capability bridge inside Juniper's turn, and any other nested call it makes (review 1, 2)
    ("cortex-exec", "skills.perception.look_at_camera.v1",
     {"policy_dispatch_only": True, "source": "cortex_exec_capability_bridge"}, "nested_exec_call"),
    ("cortex-exec", "journal.compose", {"policy_dispatch_only": True}, "nested_exec_call"),
    ("cortex-exec", "skills.runtime.image_prune.v1", {}, "nested_exec_call"),
    ("actions", "dream_cycle", {"policy_dispatch_only": True}, "nested_exec_call"),
    ("actions", "dream_cycle", {}, None),
    ("dream", "dream_cycle", {"execution_lane": "background"}, None),
])
def test_bid_skip_reason(source, verb, options, reason):
    assert pb.bid_skip_reason(_req(verb, options=options), source) == reason


def test_want_carries_the_verbs_physiology_block_and_the_request_correlation():
    from orion.cognition.plan_loader import load_verb_physiology

    block = load_verb_physiology("dream_cycle")
    now = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)
    w = pb.want_for_dispatch("dream_cycle", block, CORR, now)
    assert (w.family, w.tier, w.lanes, w.correlation_id, w.producer) == (
        "dream.cycle", 1, ["metacog"], CORR, "orion-cortex-orch.dispatch")
    assert w.want_id == f"orch:{CORR}:dream_cycle" and w.opened_at == now


def _decision(want, outcome="funded"):
    return WantDecisionV1(decision_id=f"{want.want_id}@x", want_id=want.want_id, family=want.family,
                          producer=want.producer, verb=want.verb, tier=want.tier,
                          decided_at=datetime.now(timezone.utc), outcome=outcome,
                          reason="net_positive" if outcome == "funded" else "price",
                          reserved={"lane:metacog": 60.0} if outcome == "funded" else {}, want=want)


@pytest.fixture
def harness(monkeypatch):
    calls = {"bids": [], "released": [], "dispatched": []}
    state = {"outcome": "funded", "enabled": True}

    async def fake_abid(want, **_kw):
        calls["bids"].append(want)
        return None if state["outcome"] is None else _decision(want, state["outcome"])

    async def fake_arelease(want_id, **_kw):
        calls["released"].append(want_id)
        return 1

    async def fake_call_verb_runtime(*_a, **kwargs):
        calls["dispatched"].append(kwargs["client_request"].model_dump(mode="json"))
        req = kwargs["client_request"]
        return VerbResultV1(verb=req.verb or "unknown", ok=True,
                            output={"result": {"status": "success", "steps": []}}, request_id="r")

    monkeypatch.setattr(client, "abid", fake_abid)
    monkeypatch.setattr(client, "arelease", fake_arelease)
    monkeypatch.setattr(orch_main, "call_verb_runtime", fake_call_verb_runtime)
    monkeypatch.setattr(orch_main, "svc", SimpleNamespace(bus=object()))
    monkeypatch.setattr(orch_main, "is_active", lambda *_a, **_k: True)
    real = orch_main.get_settings()
    monkeypatch.setattr(orch_main, "get_settings",
                        lambda: real.model_copy(update={"physiology_shadow_bids_enabled": state["enabled"]}))
    return calls, state


def _handle(source, verb, **kw):
    env = BaseEnvelope(kind="cortex.orch.request", source=ServiceRef(name=source, version="0", node="n"),
                       correlation_id=CORR, payload=_payload(verb, **kw))

    async def go():
        out = await orch_main.handle(env)
        await asyncio.gather(*list(orch_main._shadow_release_tasks))
        return out
    return asyncio.run(go())


@pytest.mark.parametrize("source,verb,options", [
    ("cortex-gateway", "dream_cycle", {}),                       # a tier-1 verb on Juniper's turn
    ("cortex-gateway", "skills.imagination.render_scene.v1", {}),  # a tier-2 verb on Juniper's turn
    ("actions", "chat_general", {}),                             # tier 0 chat verb
    ("actions", "dream_cycle", {"priority": "interactive"}),
    ("cortex-exec", "skills.perception.look_at_camera.v1", {"policy_dispatch_only": True}),
])
def test_turn_and_interactive_dispatches_never_bid_end_to_end(harness, source, verb, options):
    calls, _ = harness
    _handle(source, verb, options=options)
    assert calls["bids"] == [] and calls["released"] == []
    assert len(calls["dispatched"]) == 1


@pytest.mark.parametrize("outcome", ["funded", "deferred", None])
def test_self_initiated_tier1_verb_bids_and_dispatch_is_unchanged(harness, outcome):
    calls, state = harness
    state["enabled"] = False
    _handle("actions", "dream_cycle")                 # today's dispatch, no bid at all
    [baseline] = calls["dispatched"]
    assert calls["bids"] == []
    state["enabled"], state["outcome"] = True, outcome
    _handle("actions", "dream_cycle")
    [want] = calls["bids"]
    assert (want.family, want.tier, want.correlation_id) == ("dream.cycle", 1, CORR)
    # Dispatched exactly once more, byte-identical to the no-bid dispatch (nothing leaks into it).
    assert len(calls["dispatched"]) == 2 and calls["dispatched"][1] == baseline
    # A funded decision holds a reservation, and so may a bid the client gave up on: both are
    # released by want id once the dispatch returned. A deferral holds none.
    assert calls["released"] == ([] if outcome == "deferred" else [want.want_id])


def test_disabled_flag_bids_nothing(harness):
    calls, _ = harness
    assert asyncio.run(pb.shadow_bid(_req("dream_cycle"), source_name="actions", correlation_id=CORR,
                                      enabled=False)) is None
    assert calls["bids"] == []


def test_a_verb_with_no_or_tier0_physiology_never_bids(harness):
    calls, _ = harness
    for verb in ("chat_general", "no_such_verb_anywhere"):
        assert asyncio.run(pb.shadow_bid(_req(verb), source_name="actions", correlation_id=CORR, enabled=True)) is None
    assert calls["bids"] == []
    assert pb._physiology("no_such_verb_anywhere") == (None, "no_physiology")   # not "physiology_invalid"


def test_flag_ships_on():
    from pathlib import Path

    from app.settings import Settings

    assert Settings.model_fields["physiology_shadow_bids_enabled"].default is True
    svc = Path(__file__).resolve().parents[1]
    assert "\nORION_PHYSIOLOGY_SHADOW_BIDS_ENABLED=true\n" in (svc / ".env_example").read_text()
    assert "ORION_PHYSIOLOGY_SHADOW_BIDS_ENABLED: ${ORION_PHYSIOLOGY_SHADOW_BIDS_ENABLED:-true}" in (
        svc / "docker-compose.yml").read_text()
