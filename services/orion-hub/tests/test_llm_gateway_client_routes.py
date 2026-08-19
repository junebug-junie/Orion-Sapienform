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
