from __future__ import annotations

import pytest

from app import llm_backend as lb
from app.llm_backend import RouteTarget, run_llm_chat
from app.models import ChatBody, ChatMessage


@pytest.fixture
def minimal_route_table() -> dict[str, RouteTarget]:
    return {
        "chat": RouteTarget(url="http://127.0.0.1:9", served_by="atlas-chat"),
        "quick": RouteTarget(url="http://127.0.0.1:9", served_by="atlas-fast"),
    }


def test_run_llm_chat_lane_routing_spark_missing_returns_unavailable(monkeypatch, minimal_route_table: dict[str, RouteTarget]) -> None:
    monkeypatch.setattr(lb, "get_route_targets", lambda: dict(minimal_route_table))
    monkeypatch.setattr(lb.settings, "llm_lane_routing_enabled", True)
    monkeypatch.setattr(lb.settings, "llm_lane_default", "chat")
    monkeypatch.setattr(lb.settings, "llm_route_default", "chat")
    monkeypatch.setattr(lb.settings, "llm_allow_background_to_chat_fallback", False)
    monkeypatch.setattr(lb.settings, "llm_route_spark_served_by", None)
    monkeypatch.setattr(lb.settings, "llm_route_background_served_by", None)
    monkeypatch.setattr(lb.settings, "llm_route_agent_served_by", None)

    body = ChatBody(
        messages=[ChatMessage(role="user", content="hi")],
        route="quick",
        trace_id="t-lane-1",
        options={"llm_lane": "spark", "allow_chat_fallback": False},
    )
    out = run_llm_chat(body)
    assert (out.get("text") or "") == ""
    raw = out.get("raw") if isinstance(out.get("raw"), dict) else {}
    assert raw.get("error") == "llm_route_unavailable"
    det = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    assert det.get("client_route") == "quick"
    assert det.get("route_status") in {"disallowed_chat_fallback", "missing_route"}
