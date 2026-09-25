from __future__ import annotations

from app import llm_backend as lb
from app import pool_placement
from app.llm_backend import run_llm_chat
from app.models import ChatBody, ChatMessage


def test_run_llm_chat_lane_routing_spark_missing_returns_unavailable(monkeypatch) -> None:
    routes = pool_placement.pool_routes()
    monkeypatch.setattr(pool_placement, "pool_routes", lambda: {k: routes[k] for k in ("chat", "quick")})
    monkeypatch.setattr(lb.settings, "llm_lane_routing_enabled", True)
    monkeypatch.setattr(lb.settings, "llm_lane_default", "chat")
    monkeypatch.setattr(lb.settings, "llm_route_default", "chat")

    body = ChatBody(
        messages=[ChatMessage(role="user", content="hi")],
        route="quick",
        trace_id="t-lane-1",
        options={"llm_lane": "spark"},
    )
    out = run_llm_chat(body)
    assert (out.get("text") or "") == ""
    raw = out.get("raw") if isinstance(out.get("raw"), dict) else {}
    assert raw.get("error") == "llm_route_unavailable"
    det = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    assert det.get("client_route") == "quick"
    assert det.get("route_status") == "missing_route"
