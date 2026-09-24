from __future__ import annotations

import pytest

from app import llm_backend as lb
from app.llm_backend import RouteTarget, plan_llm_chat
from app.models import ChatBody, ChatMessage
from app.upstream_admission import reset_upstream_admission_for_tests


@pytest.fixture
def route_table() -> dict[str, RouteTarget]:
    return {
        "chat": RouteTarget(url="http://127.0.0.1:8011", served_by="atlas-chat"),
        "metacog": RouteTarget(url="http://127.0.0.1:8012", served_by="atlas-metacog"),
        "quick": RouteTarget(url="http://127.0.0.1:8013", served_by="atlas-fast"),
        "agent": RouteTarget(url="http://127.0.0.1:8015", served_by="atlas-agent"),
        "agent-burst": RouteTarget(url="http://127.0.0.1:8016", served_by="atlas-agent-burst"),
    }


@pytest.fixture(autouse=True)
def _configure(monkeypatch, route_table: dict[str, RouteTarget]):
    monkeypatch.setattr(lb, "get_route_targets", lambda: dict(route_table))
    monkeypatch.setattr(lb.settings, "llm_lane_routing_enabled", False)
    monkeypatch.setattr(lb.settings, "llm_lane_contention_fallback_enabled", True)
    monkeypatch.setattr(
        lb.settings,
        "llm_lane_contention_fallback_json",
        '{"metacog": ["quick", "agent"], "quick": ["metacog", "agent"]}',
    )
    monkeypatch.setattr(
        lb.settings,
        "llm_lane_real_capacity_json",
        '{"metacog": 1, "quick": 4, "agent": 1}',
    )
    lb._parse_fallback_map.cache_clear()
    lb._parse_capacity_map.cache_clear()
    reset_upstream_admission_for_tests()
    yield
    reset_upstream_admission_for_tests()


def _body(route: str) -> ChatBody:
    return ChatBody(messages=[ChatMessage(role="user", content="hi")], route=route, trace_id="t-1")


def test_metacog_route_untouched_when_not_contended() -> None:
    plan = plan_llm_chat(_body("metacog"))
    assert plan.route == "metacog"


def test_metacog_swaps_to_quick_when_its_single_slot_is_busy(route_table: dict[str, RouteTarget]) -> None:
    gate = lb.get_upstream_admission()
    gate.lane(route_table["metacog"].url).inflight = 1
    plan = plan_llm_chat(_body("metacog"))
    assert plan.route == "quick"


def test_metacog_falls_through_to_agent_when_quick_is_also_busy(route_table: dict[str, RouteTarget]) -> None:
    gate = lb.get_upstream_admission()
    gate.lane(route_table["metacog"].url).inflight = 1
    gate.lane(route_table["quick"].url).inflight = 4
    plan = plan_llm_chat(_body("metacog"))
    assert plan.route == "agent"


def test_agent_never_swaps_even_when_configured_to_burst(route_table: dict[str, RouteTarget], monkeypatch) -> None:
    # agent-burst requires a durable capacity lease ordinary traffic never has
    # (capacity.py::CapacityPermit.acquire(), BURST_LLM_ROUTES) -- even if an
    # operator misconfigures the fallback JSON to include it, the parser drops
    # it (test_lane_contention.py covers this at the unit level) and agent's
    # own contended route is left alone rather than hard-failing every request.
    monkeypatch.setattr(
        lb.settings,
        "llm_lane_contention_fallback_json",
        '{"metacog": ["quick"], "quick": ["metacog"], "agent": ["agent-burst"]}',
    )
    lb._parse_fallback_map.cache_clear()
    gate = lb.get_upstream_admission()
    gate.lane(route_table["agent"].url).inflight = 1
    plan = plan_llm_chat(_body("agent"))
    assert plan.route == "agent"


def test_chat_route_never_swapped_even_when_disabled_flag_flips(route_table: dict[str, RouteTarget], monkeypatch) -> None:
    gate = lb.get_upstream_admission()
    gate.lane(route_table["chat"].url).inflight = 999
    plan = plan_llm_chat(_body("chat"))
    assert plan.route == "chat"


def test_swap_disabled_leaves_contended_metacog_alone(route_table: dict[str, RouteTarget], monkeypatch) -> None:
    monkeypatch.setattr(lb.settings, "llm_lane_contention_fallback_enabled", False)
    gate = lb.get_upstream_admission()
    gate.lane(route_table["metacog"].url).inflight = 1
    plan = plan_llm_chat(_body("metacog"))
    assert plan.route == "metacog"
