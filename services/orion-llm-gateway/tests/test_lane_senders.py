"""GPU pool stage 6.4: the lane-sender census that gates deleting lane routing."""
from __future__ import annotations

import pytest

from orion.gpu_pool.config import RouteSpec

from app import lane_senders, pool_placement
from app import llm_backend as lb
from app.llm_backend import plan_llm_chat
from app.models import ChatBody, ChatMessage


# Pinned, so a config/gpu_pool.yaml edit cannot break these for unrelated reasons. Like the live
# table: no `spark` or `background` key, so both lanes land on metacog.
_ROUTES = {
    "chat": RouteSpec(**{"class": "chat", "priority": "interactive"}),
    "quick": RouteSpec(**{"class": "fast"}),
    "agent": RouteSpec(**{"class": "agent"}),
    "metacog": RouteSpec(**{"class": "metacog"}),
}


@pytest.fixture(autouse=True)
def _live_lane_settings(monkeypatch):
    monkeypatch.setattr(pool_placement, "pool_routes", lambda: dict(_ROUTES))
    # The live gateway: lane routing on, chat lane default, quick as the route default.
    monkeypatch.setattr(lb.settings, "llm_lane_routing_enabled", True)
    monkeypatch.setattr(lb.settings, "llm_lane_default", "chat")
    monkeypatch.setattr(lb.settings, "llm_route_default", "quick")
    lane_senders.reset()
    yield
    lane_senders.reset()


def _body(**kw) -> ChatBody:
    return ChatBody(messages=[ChatMessage(role="user", content="hi")], trace_id="t-1", **kw)


def _rows():
    return lane_senders.snapshot()["by_sender"]


def test_background_lane_is_counted_as_rerouted_to_metacog():
    # orion-thought reverie / orion-dream: route=quick (or none) + llm_lane=background.
    plan = plan_llm_chat(_body(route="quick", source="orion-thought", options={"llm_lane": "background"}))
    assert plan.route == "metacog"
    snap = lane_senders.snapshot()
    assert snap["requests_total"] == 1
    assert snap["lane_requests_total"] == 1
    assert snap["rerouted_total"] == 1
    assert snap["rerouted_without_lane_total"] == 0
    [row] = _rows()
    assert row == {**row, "source": "orion-thought", "lane": "llm_lane=background", "route_in": "quick",
                   "route_chosen": "metacog", "route_without_lane_routing": "quick",
                   "lane_routing": "applied", "rerouted": True, "requests": 1}


def test_chat_lane_on_known_route_is_counted_but_not_rerouted():
    plan_llm_chat(_body(route="agent", source="cortex-exec", options={"llm_lane": "chat"}))
    plan_llm_chat(_body(route="agent", source="cortex-exec", options={"llm_lane": "chat"}))
    snap = lane_senders.snapshot()
    assert (snap["requests_total"], snap["lane_requests_total"], snap["rerouted_total"]) == (2, 2, 0)
    [row] = _rows()
    assert row["requests"] == 2 and row["rerouted"] is False and row["route_chosen"] == "agent"


def test_execution_lane_field_is_named_when_llm_lane_absent():
    plan_llm_chat(_body(route="quick", source="x", options={"execution_lane": "Agent"}))
    [row] = _rows()
    assert row["lane"] == "execution_lane=agent"
    assert row["route_chosen"] == "agent" and row["rerouted"] is True


def test_no_lane_and_no_reroute_only_bumps_the_total(caplog):
    with caplog.at_level("INFO", logger="orion-llm-gateway.lane_senders"):
        plan_llm_chat(_body(route="chat", source="orion-hub"))
    snap = lane_senders.snapshot()
    assert snap["requests_total"] == 1 and snap["by_sender"] == []
    assert "llm_gateway_lane_sender" not in caplog.text


def test_lane_routing_ignoring_options_route_is_a_reroute_without_lane(caplog):
    # With lane routing on, an empty body.route takes LLM_ROUTE_DEFAULT and ignores options.route;
    # without it, _resolve_route honours options.route. The deletion changes this call.
    with caplog.at_level("INFO", logger="orion-llm-gateway.lane_senders"):
        plan = plan_llm_chat(_body(source="harness", options={"route": "agent"}))
    assert plan.route == "quick"
    snap = lane_senders.snapshot()
    assert snap["rerouted_total"] == 1 and snap["rerouted_without_lane_total"] == 1
    [row] = _rows()
    assert row["lane"] == "none" and row["route_without_lane_routing"] == "agent"
    assert "llm_gateway_lane_sender corr=t-1 source=harness lane=none" in caplog.text
    assert "rerouted=True" in caplog.text


def test_rejected_lane_is_recorded_as_rejected(monkeypatch):
    monkeypatch.setattr(pool_placement, "pool_routes", lambda: {k: _ROUTES[k] for k in ("chat", "quick")})
    plan = plan_llm_chat(_body(route="quick", source="s", options={"llm_lane": "spark"}))
    assert plan.error is not None
    [row] = _rows()
    assert row["route_chosen"] == "rejected" and row["rerouted"] is True


def test_hold_and_disabled_paths_are_counted_without_reroute(monkeypatch):
    plan_llm_chat(_body(route="agent", source="durable", options={
        "llm_lane": "background",
        "gpu_lease": {"lease_id": "l", "holder": "h", "role": "r", "url": "http://x"}}))
    monkeypatch.setattr(lb.settings, "llm_lane_routing_enabled", False)
    plan_llm_chat(_body(route="quick", source="thought", options={"llm_lane": "background"}))
    rows = {r["source"]: r for r in _rows()}
    assert rows["durable"]["lane_routing"] == "skipped_hold" and rows["durable"]["rerouted"] is False
    assert rows["thought"]["lane_routing"] == "disabled" and rows["thought"]["route_chosen"] == "quick"
    assert lane_senders.snapshot()["rerouted_total"] == 0


def test_rows_are_bounded():
    for i in range(lane_senders._MAX_ROWS + 10):
        lane_senders.record(source=f"s{i}", lane="llm_lane=chat", route_in="quick", route_chosen="quick",
                            route_without_lane_routing="quick", lane_routing="applied")
    snap = lane_senders.snapshot()
    assert snap["lane_requests_total"] == lane_senders._MAX_ROWS + 10
    assert len(snap["by_sender"]) == lane_senders._MAX_ROWS + 1
    [other] = [r for r in snap["by_sender"] if r["source"] == "other"]
    assert other["requests"] == 10 and other["rerouted"] is False
    # A reroute in overflow keeps its own row, never hidden among the non-reroutes.
    lane_senders.record(source="late", lane=None, route_in=None, route_chosen="quick",
                        route_without_lane_routing="agent", lane_routing="applied")
    overflow = {r["rerouted"]: r["requests"] for r in lane_senders.snapshot()["by_sender"]
                if r["source"] == "other"}
    assert overflow == {False: 10, True: 1}


def test_debug_endpoint_returns_snapshot():
    from fastapi.testclient import TestClient
    from app.main import app
    lane_senders.record(source="a", lane="llm_lane=agent", route_in="quick", route_chosen="agent",
                        route_without_lane_routing="quick", lane_routing="applied")
    body = TestClient(app).get("/debug/lane-senders").json()
    assert body["rerouted_total"] == 1 and body["by_sender"][0]["source"] == "a"


def test_unknown_route_rescued_by_lane_routing_is_marked_as_failing_after_deletion():
    # Lane routing turns a misspelled route into quick; without it the pool refuses the name.
    plan = plan_llm_chat(_body(route="qiuck", source="typo"))
    assert plan.route == "quick" and plan.error is None
    [row] = _rows()
    assert row["route_without_lane_routing"] == "rejected:qiuck" and row["rerouted"] is True


def test_both_sides_rejected_is_not_a_reroute(monkeypatch):
    monkeypatch.setattr(pool_placement, "pool_routes", lambda: {k: _ROUTES[k] for k in ("chat", "quick")})
    plan_llm_chat(_body(route="nope", source="s", options={"llm_lane": "agent"}))
    [row] = _rows()
    assert row["route_chosen"] == "rejected" and row["route_without_lane_routing"] == "rejected:nope"
    assert row["rerouted"] is False and lane_senders.snapshot()["rerouted_total"] == 0


def test_blank_llm_lane_is_labelled_like_the_router_reads_it():
    # resolve_llm_lane_route reads `llm_lane or execution_lane`: a blank llm_lane is falsy.
    assert lane_senders.lane_field({"llm_lane": "", "execution_lane": "agent"}) == "execution_lane=agent"
    assert lane_senders.lane_field({"llm_lane": 0}) is None
    assert lane_senders.lane_field({"llm_lane": "Background"}) == "llm_lane=background"
    assert lane_senders.lane_field(None) is None


_MATRIX = [
    {"route": "quick"}, {"route": "agent"}, {"route": "chat"}, {}, {"route": "qiuck"},
    {"options": {"route": "agent"}}, {"options": {"routing_key": "metacog"}},
    {"route": "quick", "options": {"llm_lane": "background"}},
    {"route": "quick", "options": {"llm_lane": "spark"}},
    {"route": "quick", "options": {"execution_lane": "agent"}},
    {"route": "agent", "options": {"llm_lane": "chat"}},
    {"route": "agent", "options": {"llm_lane": "bogus"}},
    {"options": {"llm_lane": "", "route": "metacog"}},
]


@pytest.mark.parametrize("kw", _MATRIX)
def test_census_prediction_matches_plan_with_lane_routing_off(monkeypatch, kw):
    """The claim the deletion PR rests on: route_without_lane_routing is what plan_llm_chat really
    does once lane routing is gone (approximated here by LLM_LANE_ROUTING_ENABLED=false)."""
    plan_llm_chat(_body(source="m", **kw))
    [row] = _rows() or [{"route_chosen": None, "route_without_lane_routing": None}]
    monkeypatch.setattr(lb.settings, "llm_lane_routing_enabled", False)
    off = plan_llm_chat(_body(source="m", **kw))
    actual = f"rejected:{off.route}" if off.error is not None else off.route
    if row["route_without_lane_routing"] is None:
        # Not recorded: no lane and not rerouted, so today's route is already the off route.
        on_plan_route = actual
        lane_senders.reset()
        monkeypatch.setattr(lb.settings, "llm_lane_routing_enabled", True)
        assert plan_llm_chat(_body(source="m", **kw)).route == on_plan_route
    else:
        assert row["route_without_lane_routing"] == actual


def test_a_failing_counter_never_fails_the_call(monkeypatch):
    def boom(**_kw):
        raise RuntimeError("counter broke")

    monkeypatch.setattr(lane_senders, "_record", boom)
    plan = plan_llm_chat(_body(route="quick", source="s", options={"llm_lane": "background"}))
    assert plan.route == "metacog" and plan.error is None
    monkeypatch.setattr(lane_senders, "_clean", boom)
    assert lane_senders.lane_field({"llm_lane": "agent"}) is None
