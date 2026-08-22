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
    ("nonsense", "chat"),
    (None, "chat"),
])
def test_default_route_is_normalized_not_name_matched(raw, expected):
    payload = _gateway_payload()
    payload["default_route"] = raw
    assert _normalize(payload)["default_route"] == expected


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
