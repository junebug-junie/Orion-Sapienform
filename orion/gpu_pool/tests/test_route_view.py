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
    view = build_route_view(_state([_role("chat"), _role("fast", vision=True), _role("metacog", "down")]))
    assert view["source"] == SOURCE_POOL
    routes = _by_id(view)
    assert set(routes) == set(CFG.routes)
    chat = routes["chat"]
    assert (chat["status"], chat["model"], chat["n_ctx"], chat["role"]) == ("up", "chat.gguf", 8192, "chat")
    # metacog's own role is down, so the route lands on the first up role it may use (fast).
    assert routes["metacog"]["status"] == "up" and routes["metacog"]["role"] == "fast"
    assert routes["quick"]["vision"] is True
    assert routes["quick_background"]["priority"] == "background" and routes["harness"]["priority"] == "system"
    assert routes["chat-burst"]["status"] == "operator_closed" and routes["chat-burst"]["gate_open"] is False


ROUTES_ADDED_AFTER_GOLDEN = {"agent-deep"}   # hecate, 2026-10-08
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
    assert [r for r in got if r["id"] in frozen] == golden["routes"]


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
