"""orion.gpu_pool.route_view: the per-route view every former GET /routes reader now builds itself.

Stage 6.3. The view is built from the pool state the reader got over RPC, with the pool's own
config riding along (``include_config=True``) -- so these tests feed it exactly that shape.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from orion.gpu_pool.config import load_pool_config
from orion.gpu_pool.route_view import (
    SOURCE_POOL, SOURCE_UNAVAILABLE, build_route_view, config_from_state, fetch_route_view, route_entry,
)
from orion.llm.routes import LLM_ROUTE_DISPLAY_ORDER

CFG = load_pool_config()


def _config_payload() -> dict[str, Any]:
    # Exactly what the pool sends (services/orion-gpu-pool/app/runtime.py snapshot()).
    return CFG.model_dump(mode="json", by_alias=True, exclude={"digest"})


def _role(role: str, status: str = "confirmed", **kw: Any) -> dict[str, Any]:
    return {"role": role, "kind": "llm", "cards": ["x"], "url": f"http://h:{CFG.roles[role].port}",
            "status": status, "model_file": f"{role}.gguf", "ctx_per_slot": 8192, "vision": False,
            "checked_at": "2026-09-30T00:00:00Z", **kw}


def _state(roles, *, gpu0_lent: bool = False, config: bool = True) -> dict[str, Any]:
    state = {"generated_at": "2026-09-30T00:00:00Z", "cards": [
        {"card": "gpu0", "vram_gb": 32, "lendable": True, "lent": gpu0_lent},
        {"card": "gpu1", "vram_gb": 32}, {"card": "gpu2", "vram_gb": 32}, {"card": "gpu3", "vram_gb": 32},
    ], "roles": roles}
    if config:
        state["config"] = _config_payload()
    return state


def _by_id(view):
    return {r["id"]: r for r in view["routes"]}


def test_config_round_trips_from_the_pool_payload():
    cfg = config_from_state({"config": _config_payload()})
    assert cfg is not None and set(cfg.routes) == set(CFG.routes)


def test_view_from_state_uses_the_config_the_pool_sent():
    view = build_route_view(_state([_role("chat"), _role("fast", vision=True)]))
    assert view["source"] == SOURCE_POOL
    routes = _by_id(view)
    assert set(routes) == set(CFG.routes)
    chat = routes["chat"]
    assert (chat["status"], chat["model"], chat["n_ctx"], chat["role"]) == ("up", "chat.gguf", 8192, "chat")
    # Both classes now use the same fast role; the route still retains metacog priority.
    assert routes["metacog"]["status"] == "up" and routes["metacog"]["role"] == "fast"
    assert routes["quick"]["vision"] is True
    assert routes["quick_background"]["priority"] == "background" and routes["harness"]["priority"] == "system"
    assert routes["chat-burst"]["status"] == "operator_closed" and routes["chat-burst"]["gate_open"] is False


ROUTES_ADDED_AFTER_GOLDEN = {"agent-deep",          # hecate, 2026-10-08
                             "metacog_perception",  # Orion's sight, 2026-10-23 (pinned below)
                             "quick_turn",          # repair-pressure probe, 2026-10-23 (pinned below)
                             "quick_embodied",      # Orion's town speech, 2026-10-23
                             "prime",               # Orion Prime's thoughts, 2026-10-30 (pinned below)
                             "prime_starved",       # Prime after a try that got no seat, 2026-11-03 (pinned below)
                             "judge",               # finalize 5b judge off a gpu2 hold, 2026-11-12 (pinned below)
                             # operator work (evals), 2026-11-12 (pinned below)
                             "operator", "operator_chat", "operator_agent", "operator_agent_gpu2",
                             "operator_agent_deep", "operator_metacog", "operator_fast"}
# Roles whose server type changed after the freeze report the current one wherever they serve.
BACKEND_CHANGED_AFTER_GOLDEN = {"chat": "strata", "agent": "strata"}   # Strata on gpu0 10-30, gpu1 10-31
MIGRATED_ROUTES = {"metacog", "metacog_background", "metacog_turn"}  # all now share fast
_GOLDEN = json.loads((Path(__file__).parent / "fixtures_routes_compat_golden.json").read_text())


@pytest.mark.parametrize("case", sorted(_GOLDEN))
def test_view_matches_the_old_gateway_generator_output(case):
    """Frozen output of the pre-6.3 gateway ``build_routes_compat`` (origin/main a005658db, run on
    these states with the repo's config/gpu_pool.yaml). The moved generator must reproduce it route
    for route; the only addition is ``role``. A reader's state carries the pool's config, so the
    view is built from that; the no-state case is the gateway path (its own config)."""
    golden = _GOLDEN[case]
    if golden["state"] is None:
        view = build_route_view(None, CFG)
    else:
        view = build_route_view({**golden["state"], "config": _config_payload()})
    got = [{k: v for k, v in entry.items() if k != "role"} for entry in view["routes"]]
    # Routes added after the freeze have no old-generator output to match; each is pinned on its own.
    frozen = {r["id"] for r in golden["routes"]}
    assert {r["id"] for r in got} - frozen == ROUTES_ADDED_AFTER_GOLDEN
    # served_by is "<host>-worker-<role>", the role whose backend the entry reports (route_view._entry)
    expected = [{**r, "backend": BACKEND_CHANGED_AFTER_GOLDEN.get(r["served_by"].rsplit("-worker-", 1)[1], r["backend"])}
                if r["backend"] else r for r in golden["routes"]]
    assert [r for r in got if r["id"] in frozen - MIGRATED_ROUTES] == [
        r for r in expected if r["id"] not in MIGRATED_ROUTES]


def test_down_route_carries_no_model_ctx_or_vision():
    routes = _by_id(build_route_view(_state([_role("chat", "down")])))
    assert routes["chat"]["status"] == "down"
    assert (routes["chat"]["model"], routes["chat"]["n_ctx"], routes["chat"]["vision"], routes["chat"]["role"]) == (
        None, None, None, None)


@pytest.mark.parametrize("state", [
    None,
    # A periodic broadcast frame (no config) cannot say which role a route lands on.
    {"generated_at": "2026-09-30T00:00:00Z", "cards": [], "roles": [
        {"role": "chat", "kind": "llm", "cards": ["gpu0"], "url": "http://h:8011", "status": "confirmed",
         "model_file": "chat.gguf", "ctx_per_slot": 131072}]},
    # A config the reader cannot parse.
    {"roles": [], "cards": [], "config": {"not": "a pool config"}},
])
def test_no_usable_pool_state_is_unknown_never_a_guessed_default(state):
    view = build_route_view(state)
    assert view["source"] == SOURCE_UNAVAILABLE
    assert [r["id"] for r in view["routes"]] == list(LLM_ROUTE_DISPLAY_ORDER)
    for entry in view["routes"]:
        assert entry["status"] == "unknown"
        assert entry["model"] is None and entry["n_ctx"] is None and entry["vision"] is None
        assert entry["role"] is None


def test_route_entry_lookup():
    view = build_route_view(_state([_role("chat")]))
    assert route_entry(view, "chat")["model"] == "chat.gguf"
    assert route_entry(view, "no-such-route") is None
    assert route_entry(None, "chat") is None


class _Decoded:
    ok = True

    def __init__(self, payload):
        self.envelope = type("E", (), {"payload": payload})()


class _Bus:
    def __init__(self, payload=None, exc: Exception | None = None):
        self.sent = []
        self._payload = payload
        self._exc = exc
        self.codec = type("C", (), {"decode": lambda _s, data: _Decoded(self._payload)})()

    async def rpc_request(self, channel, env, **kw):
        self.sent.append((channel, env, kw))
        if self._exc:
            raise self._exc
        return {"data": b"raw"}


@pytest.mark.asyncio
async def test_fetch_route_view_asks_the_pool_for_its_config():
    bus = _Bus(_state([_role("chat")]))
    view = await fetch_route_view(bus, source="test")
    assert view["source"] == SOURCE_POOL and route_entry(view, "chat")["status"] == "up"
    (channel, env, _kw), = bus.sent
    assert channel == "orion:gpu_pool:state:request"
    assert env.payload == {"include_leases": False, "include_config": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("bus", [None, _Bus(exc=TimeoutError("pool silent"))])
async def test_fetch_route_view_pool_unreachable_is_all_unknown(bus):
    view = await fetch_route_view(bus, source="test")
    assert view["source"] == SOURCE_UNAVAILABLE
    assert all(r["status"] == "unknown" and r["model"] is None for r in view["routes"])


def test_metacog_perception_route_is_system_only_on_the_metacog_class():
    """Orion's sight (body-economy design A0): served by metacog roles, hidden from the human picker
    (it jumps the heat shed, like metacog_turn), and pinned to the metacog profile."""
    from orion.llm.routes import METACOG_LLM_ROUTES, SYSTEM_LLM_ROUTES
    view = build_route_view(None, CFG)
    [r] = [e for e in view["routes"] if e["id"] == "metacog_perception"]
    [turn] = [e for e in view["routes"] if e["id"] == "metacog_turn"]
    assert r["priority"] == "system" == turn["priority"]
    assert {k: v for k, v in r.items() if k not in ("id",)} == {k: v for k, v in turn.items() if k not in ("id",)}
    assert "metacog_perception" in METACOG_LLM_ROUTES and "metacog_perception" in SYSTEM_LLM_ROUTES
    assert CFG.routes["metacog_perception"].priority == "perception"


def test_quick_turn_route_is_system_only_on_the_fast_class():
    """The pre-turn repair-pressure probe's route: served exactly like `quick` (same class, same
    roles, same model), but interactive in the pool so the heat reflex never sheds it, and hidden
    from the human picker (route_view reports priority "system")."""
    from orion.llm.routes import METACOG_LLM_ROUTES, SYSTEM_LLM_ROUTES
    view = build_route_view(None, CFG)
    [r] = [e for e in view["routes"] if e["id"] == "quick_turn"]
    [quick] = [e for e in view["routes"] if e["id"] == "quick"]
    assert r["priority"] == "system" and quick["priority"] != "system"   # quick stays pickable
    same = lambda e: {k: v for k, v in e.items() if k not in ("id", "priority")}  # noqa: E731
    assert same(r) == same(quick)
    assert "quick_turn" in SYSTEM_LLM_ROUTES and "quick_turn" not in METACOG_LLM_ROUTES
    assert CFG.routes["quick_turn"].priority == "interactive"
    assert CFG.routes["quick_turn"].work_class == CFG.routes["quick"].work_class == "fast"


def test_prime_routes_are_the_chat_worker_at_system_and_starved():
    """Orion Prime's thoughts: served exactly like `chat` (same class, same roles, same warm model).
    2026-11-03: `prime` is system (it was background and lost its whole budget to the queue);
    `prime_starved` is the same call at `starved`, after a try that got no seat. Both system-only
    (orion/llm/routes.py), so neither is a human's Compute choice."""
    from orion.llm.routes import BACKGROUND_LLM_ROUTES, METACOG_LLM_ROUTES, SYSTEM_LLM_ROUTES
    view = build_route_view(None, CFG)
    [chat] = [e for e in view["routes"] if e["id"] == "chat"]
    same = lambda e: {k: v for k, v in e.items() if k not in ("id", "priority")}  # noqa: E731
    for route, prio in (("prime", "system"), ("prime_starved", "starved")):
        [r] = [e for e in view["routes"] if e["id"] == route]
        assert same(r) == same(chat)
        assert CFG.routes[route].priority == prio
        assert CFG.routes[route].work_class == CFG.routes["chat"].work_class == "chat"
        assert r["priority"] == "system"   # definitional: hidden from the picker
        assert route in SYSTEM_LLM_ROUTES and route not in BACKGROUND_LLM_ROUTES
        assert route not in METACOG_LLM_ROUTES


def test_judge_route_is_agent_work_that_never_lands_on_the_gpu2_seat():
    """The finalize 5b judge off a hold (2026-11-12, orion/harness/finalize.py): gpu1, then chat's card
    only while lent; never agent-gpu2 or hecate. System priority, system-only, a cortex-exec override."""
    from orion.llm.routes import BACKGROUND_LLM_ROUTES, CORTEX_OVERRIDE_SYSTEM_ROUTES, SYSTEM_LLM_ROUTES
    spec = CFG.routes["judge"]
    assert (spec.work_class, spec.priority) == ("judge", "system")
    assert CFG.classes["judge"].roles == ["agent", "chat"]
    assert "judge" in SYSTEM_LLM_ROUTES and "judge" in CORTEX_OVERRIDE_SYSTEM_ROUTES
    assert "judge" not in BACKGROUND_LLM_ROUTES
    [r] = [e for e in build_route_view(None, CFG)["routes"] if e["id"] == "judge"]
    assert r["priority"] == "system"


def test_operator_routes_are_background_and_land_only_on_their_pinned_role():
    """Leased evals (2026-11-12): every operator route is background; a pinned one names one role."""
    routes = _by_id(build_route_view(_state([_role(r) for r in ("chat", "agent", "agent-gpu2", "fast")],
                                            gpu0_lent=True)))
    from orion.gpu_pool.config import load_pool_config

    cfg = load_pool_config()
    for rid in ("operator", "operator_chat", "operator_agent", "operator_agent_gpu2", "operator_agent_deep",
                "operator_metacog", "operator_fast"):
        assert cfg.routes[rid].priority == "background", rid
        assert cfg.routes[rid].work_class == rid
    assert routes["operator_chat"]["role"] == "chat"
    assert routes["operator_agent_gpu2"]["role"] == "agent-gpu2"
