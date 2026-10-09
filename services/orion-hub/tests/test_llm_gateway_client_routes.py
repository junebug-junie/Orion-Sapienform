"""The Hub must not narrow the gateway's catalog behind the gateway's back.

Regression for 2026-08-19: `quick_background` had been carrying Orion's own journalling since
PR #1708 and was absent from every Hub surface. Widening this module's `VALID_ROUTE_IDS` alone
changed NOTHING VISIBLE -- two further hardcoded ("chat","quick","agent","metacog") tuples
backfilled and reordered the payload afterwards, reassembling a four-route response out of a
five-route one. Confirmed live before the fix: gateway returned 5, Hub served 4.

These tests drive `_normalize_routes_payload` with a realistic gateway response rather than
asserting that a constant equals another constant, because a re-export check would have passed
throughout the entire period the bug existed.
"""
from __future__ import annotations

import pytest

from orion.llm.routes import LLM_ROUTE_DISPLAY_ORDER
from scripts import llm_gateway_client


def _gateway_payload():
    return {
        "default_route": "quick",
        "routes": [
            {"id": "chat", "served_by": "circe-worker-1", "backend": "llamacpp",
             "status": "down", "latency_ms": 1619, "last_checked_at": "t", "vision": None,
             "priority": None, "reserved_free_slots": None},
            {"id": "quick", "served_by": "atlas-worker-fast-1", "backend": "llamacpp",
             "status": "up", "latency_ms": 71, "last_checked_at": "t", "vision": False,
             "priority": None, "reserved_free_slots": None},
            {"id": "quick_background", "served_by": "atlas-worker-fast-1", "backend": "llamacpp",
             "status": "up", "latency_ms": 71, "last_checked_at": "t", "vision": False,
             "priority": "background", "reserved_free_slots": 2},
            {"id": "metacog", "served_by": "atlas-worker-2", "backend": "llamacpp",
             "status": "up", "latency_ms": 40, "last_checked_at": "t", "vision": None,
             "priority": None, "reserved_free_slots": None},
            {"id": "agent", "served_by": "circe-worker-agent-1", "backend": "llamacpp",
             "status": "down", "latency_ms": 1600, "last_checked_at": "t", "vision": None,
             "priority": None, "reserved_free_slots": None},
            {"id": "harness", "served_by": "circe-worker-1", "backend": "llamacpp",
             "status": "down", "latency_ms": 1619, "last_checked_at": "t", "vision": None,
             "priority": "system", "reserved_free_slots": None},
        ],
    }


def _normalize(payload):
    """The pure half of fetch_routes. Exposed here so these tests need no HTTP."""
    return llm_gateway_client._normalize_routes_payload(payload)


def test_every_gateway_route_reaches_the_hub():
    out = _normalize(_gateway_payload())
    assert [r["id"] for r in out["routes"]] == list(LLM_ROUTE_DISPLAY_ORDER)
    assert "quick_background" in [r["id"] for r in out["routes"]]


def test_background_priority_is_passed_through():
    """The Hub picker filters on this. Dropping it would silently make a yielding lane look
    like a normal one and offer it to a human."""
    out = _normalize(_gateway_payload())
    bg = next(r for r in out["routes"] if r["id"] == "quick_background")
    assert bg["priority"] == "background"
    assert bg["reserved_free_slots"] == 2


def test_a_route_the_gateway_omits_is_backfilled_as_unknown():
    """Not dropped -- a route that exists but was not reported is 'unknown', not absent."""
    payload = _gateway_payload()
    payload["routes"] = [r for r in payload["routes"] if r["id"] != "agent"]
    out = _normalize(payload)
    agent = next(r for r in out["routes"] if r["id"] == "agent")
    assert agent["status"] == "unknown"
    assert agent["priority"] is None
    assert [r["id"] for r in out["routes"]] == list(LLM_ROUTE_DISPLAY_ORDER)


def test_an_unknown_route_id_from_the_gateway_is_ignored():
    payload = _gateway_payload()
    payload["routes"].append({"id": "not_a_route", "status": "up"})
    out = _normalize(payload)
    assert "not_a_route" not in [r["id"] for r in out["routes"]]


@pytest.mark.parametrize("raw,expected", [
    ("quick", "quick"),
    ("quick_background", "quick_background"),
    ("chat_quick", "quick"),        # legacy alias resolves rather than collapsing to chat
    # Unrecognised/missing default falls back to "quick", NOT "chat": chat is the most
    # contended lane (Juniper's own worker) and must not be poached by a bad default.
    ("nonsense", "quick"),
    (None, "quick"),
])
def test_default_route_is_normalized_not_name_matched(raw, expected):
    payload = _gateway_payload()
    payload["default_route"] = raw
    assert _normalize(payload)["default_route"] == expected


class TestOperatorGateIsPassedThrough:
    """`gate_open` is what the Hub's "Lend chat lane" button renders from. Dropping it in
    the reassembly below would leave the button permanently "off" while the gate is open."""

    def test_gate_open_is_passed_through_for_a_reported_route(self):
        payload = _gateway_payload()
        payload["routes"].append({"id": "chat-burst", "served_by": "circe-worker-1",
                                  "status": "operator_closed", "priority": "system",
                                  "gate_open": False})
        out = _normalize(payload)
        burst = next(r for r in out["routes"] if r["id"] == "chat-burst")
        assert burst["gate_open"] is False
        assert burst["priority"] == "system"
        payload["routes"][-1]["gate_open"] = True
        out = _normalize(payload)
        assert next(r for r in out["routes"] if r["id"] == "chat-burst")["gate_open"] is True

    def test_non_gated_and_backfilled_routes_report_none(self):
        out = _normalize(_gateway_payload())
        assert next(r for r in out["routes"] if r["id"] == "chat")["gate_open"] is None
        # chat-burst is absent from _gateway_payload(), so this is the backfill row.
        assert next(r for r in out["routes"] if r["id"] == "chat-burst")["gate_open"] is None


class TestPriorityIsFailSafe:
    """A background lane must never be presented as pickable because a field was missing.

    Two real paths deliver `quick_background` with no priority: a rolling deploy where the Hub
    ships before the gateway (old gateway sends 4 routes, no `priority` key at all), and the
    route being absent from LLM_GATEWAY_ROUTE_TABLE_JSON. In both, filtering on
    `priority == 'background'` fails OPEN and the composer offers the yielding lane to a human.
    """

    def test_old_gateway_omitting_the_route_entirely(self):
        """Rolling deploy: gateway still returns the pre-patch four routes."""
        payload = {
            "default_route": "quick",
            "routes": [
                {"id": "chat", "status": "up"}, {"id": "quick", "status": "up"},
                {"id": "agent", "status": "up"}, {"id": "metacog", "status": "up"},
            ],
        }
        out = _normalize(payload)
        bg = next(r for r in out["routes"] if r["id"] == "quick_background")
        assert bg["priority"] == "background", "backfilled row must not claim to be interactive"

    def test_gateway_reporting_the_route_without_a_priority_field(self):
        payload = _gateway_payload()
        for r in payload["routes"]:
            r.pop("priority", None)
        out = _normalize(payload)
        bg = next(r for r in out["routes"] if r["id"] == "quick_background")
        assert bg["priority"] == "background"
        assert next(r for r in out["routes"] if r["id"] == "quick")["priority"] is None

    def test_a_reported_background_priority_is_honoured_for_any_route(self):
        """The definitional set is a floor, not a ceiling: if the gateway says a route yields,
        believe it even for a route not named in BACKGROUND_LLM_ROUTES."""
        payload = _gateway_payload()
        next(r for r in payload["routes"] if r["id"] == "metacog")["priority"] = "background"
        out = _normalize(payload)
        assert next(r for r in out["routes"] if r["id"] == "metacog")["priority"] == "background"

    def test_the_picker_filter_excludes_it_in_every_case(self):
        """Mirrors pickableComputeRouteIds() in app.js, which filters on this exact field.

        Was `!= "background"` only until 2026-08-19's review caught the gap it left: app.js's
        real filter excludes BOTH `"background"` (quick_background, yields for slot slack) and
        `"system"` (harness, never a human's turn but dispatches immediately) via two separate
        checks (isBackgroundRouteEntry, isSystemRouteEntry) -- a one-value reimplementation here
        would keep passing even if a future edit to app.js dropped the system-priority check
        entirely, since this test's own local filter would silently agree with the broken code.
        """
        for payload in (_gateway_payload(), {"default_route": "quick", "routes": []}):
            out = _normalize(payload)
            pickable = [r["id"] for r in out["routes"]
                        if str(r.get("priority") or "").lower() not in ("background", "system")]
            assert "quick_background" not in pickable
            assert "harness" not in pickable

    def test_gateway_reporting_harness_without_a_priority_field(self):
        """Same fail-safe as background's version above, mirrored for `system`: a route-table
        entry carrying no `priority` key at all (easy -- neighbouring `chat`/`agent` entries in
        the same JSON blob carry none) must not make `harness` look like an ordinary lane."""
        payload = _gateway_payload()
        for r in payload["routes"]:
            r.pop("priority", None)
        out = _normalize(payload)
        harness = next(r for r in out["routes"] if r["id"] == "harness")
        assert harness["priority"] == "system"


# ── GPU pool stage 6.3: the catalog is built from pool state, not the gateway's GET /routes ──


class TestCatalogFromPoolState:
    def setup_method(self):
        llm_gateway_client.reset_route_view_cache()

    def teardown_method(self):
        llm_gateway_client.reset_route_view_cache()

    @pytest.mark.asyncio
    async def test_no_bus_is_every_route_unknown_never_a_guess(self, monkeypatch):
        monkeypatch.setattr(llm_gateway_client, "_rpc_bus", lambda: None)
        out = await llm_gateway_client.fetch_routes()
        assert out["source"] == "gpu_pool_unavailable"
        assert [r["id"] for r in out["routes"]] == list(LLM_ROUTE_DISPLAY_ORDER)
        assert all(r["status"] == "unknown" and r["vision"] is None for r in out["routes"])
        # The picker still gets its fail-safe definitional priority for a yielding lane.
        assert {r["id"]: r["priority"] for r in out["routes"]}["quick_background"] == "background"

    @pytest.mark.asyncio
    async def test_pool_silent_is_unknown_and_retried_soon(self, monkeypatch):
        calls = []

        async def _silent(bus, **kw):
            calls.append(kw)
            return None

        monkeypatch.setattr("orion.gpu_pool.placement.fetch_pool_state", _silent)
        monkeypatch.setattr(llm_gateway_client, "_rpc_bus", lambda: object())
        out = await llm_gateway_client.fetch_routes()
        assert out["source"] == "gpu_pool_unavailable"
        assert all(r["status"] == "unknown" for r in out["routes"])
        assert llm_gateway_client._cache["ttl"] == llm_gateway_client._FAILURE_CACHE_SEC
        assert calls and calls[0]["include_config"] is True

    @pytest.mark.asyncio
    async def test_one_pool_read_serves_every_tab_inside_the_cache_window(self, monkeypatch):
        from orion.gpu_pool.config import load_pool_config

        cfg = load_pool_config()
        state = {"cards": [], "roles": [
            {"role": "fast", "kind": "llm", "cards": ["gpu3"], "url": "http://h:8013", "status": "confirmed",
             "model_file": "fast.gguf", "ctx_per_slot": 4096, "vision": True}],
            "config": cfg.model_dump(mode="json", by_alias=True, exclude={"digest"})}
        calls = []

        async def _state(bus, **kw):
            calls.append(kw)
            return state

        monkeypatch.setattr("orion.gpu_pool.placement.fetch_pool_state", _state)
        monkeypatch.setattr(llm_gateway_client, "_rpc_bus", lambda: object())
        first = await llm_gateway_client.fetch_routes()
        second = await llm_gateway_client.fetch_routes()
        assert len(calls) == 1 and first == second
        quick = {r["id"]: r for r in first["routes"]}["quick"]
        # The attach-image button follows this flag.
        assert quick["status"] == "up" and quick["vision"] is True


def test_api_llm_routes_is_200_with_unknown_lanes_when_the_pool_is_unreachable(monkeypatch):
    """The endpoint used to 502 when the gateway was down; the browser now always gets a catalog,
    and an unreachable pool reads as unknown lanes (the picker's existing 'unknown' rendering)."""
    import asyncio

    from scripts import api_routes

    llm_gateway_client.reset_route_view_cache()
    monkeypatch.setattr(llm_gateway_client, "_rpc_bus", lambda: None)
    out = asyncio.run(api_routes.api_llm_routes())
    llm_gateway_client.reset_route_view_cache()
    assert out["source"] == "gpu_pool_unavailable"
    assert all(r["status"] == "unknown" for r in out["routes"])


def test_hub_no_longer_reads_the_gateway_routes_endpoint():
    """Pins the move so 6.5's zero-read window cannot be reopened by a revert."""
    import inspect

    source = inspect.getsource(llm_gateway_client)
    assert "aiohttp" not in source and "HUB_LLM_GATEWAY_URL" not in source


def test_lend_toggle_ignores_an_unknown_gate_instead_of_rendering_closed():
    """Pool unreachable -> `gate_open: null` on every lane. The composer's lend toggle must keep
    its last known state, not flip to "closed" (a guess). Source-level: the Hub has no JS runner."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "static" / "js" / "app.js").read_text()
    body = js.split("function chatBurstGateFromCatalog(catalog) {", 1)[1].split("\n  }\n", 1)[0]
    assert "typeof entry.gate_open === 'boolean'" in body
    assert "entry.gate_open === true" not in body


def test_default_route_is_the_hubs_own_constant_matching_the_composer(monkeypatch):
    """Pool state carries no gateway LLM_ROUTE_DEFAULT; the payload states the Hub's constant, and
    that constant must agree with the composer's HUB_COMPUTE_DEFAULT."""
    import asyncio
    import re
    from pathlib import Path

    llm_gateway_client.reset_route_view_cache()
    monkeypatch.setattr(llm_gateway_client, "_rpc_bus", lambda: None)
    out = asyncio.run(llm_gateway_client.fetch_routes())
    llm_gateway_client.reset_route_view_cache()
    js = (Path(__file__).resolve().parents[1] / "static" / "js" / "app.js").read_text()
    js_default = re.search(r"const HUB_COMPUTE_DEFAULT = '([^']+)';", js).group(1)
    assert out["default_route"] == llm_gateway_client.HUB_DEFAULT_ROUTE == js_default
