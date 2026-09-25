from __future__ import annotations

from app.lane_routes import resolve_llm_lane_route


def _resolve(
    *,
    options: dict | None,
    body_route: str | None,
    keys: set[str],
    served_by: dict[str, str | None] | None = None,
    **kwargs: object,
) -> object:
    # served_by is kept in the call sites for readability only: lane resolution picks a route
    # NAME; the GPU pool's grant supplies served_by later.
    defaults = {"llm_lane_default": "chat", "llm_route_default": "chat"}
    defaults.update(kwargs)
    return resolve_llm_lane_route(
        options,
        body_route,
        llm_lane_default=str(defaults["llm_lane_default"]),
        llm_route_default=str(defaults["llm_route_default"]),
        route_table_keys=keys,
    )


def test_chat_lane_resolves_body_route() -> None:
    d = _resolve(
        options={"llm_lane": "chat", "allow_chat_fallback": True},
        body_route="quick",
        keys={"chat", "quick"},
        served_by={"quick": "atlas-fast", "chat": "atlas-chat"},
    )
    assert d.route_status in ("ok", "invalid_lane")
    assert d.route_table_key == "quick"
    assert d.resolved_llm_lane == "chat"


def test_missing_lane_preserves_body_route_with_shipped_default() -> None:
    d = _resolve(
        options=None,
        body_route="agent",
        keys={"chat", "agent", "quick", "metacog"},
        served_by={"chat": "c1", "agent": "a1", "quick": "q1", "metacog": "m1"},
        llm_lane_default="chat",
    )
    assert d.route_status == "ok"
    assert d.requested_llm_lane == "chat"
    assert d.route_table_key == "agent"


def test_spark_lane_prefers_spark_key() -> None:
    d = _resolve(
        options={"llm_lane": "spark", "allow_chat_fallback": False},
        body_route="quick",
        keys={"spark", "chat", "quick"},
        served_by={"spark": "spark-w", "chat": "c1", "quick": "q1"},
    )
    assert d.route_table_key == "spark"
    assert d.resolved_llm_lane == "spark"
    assert d.fallback_used is False


def test_spark_falls_back_to_background_not_chat() -> None:
    d = _resolve(
        options={"llm_lane": "spark", "allow_chat_fallback": False},
        body_route="quick",
        keys={"background", "chat", "quick"},
        served_by={"background": "bg1", "chat": "c1", "quick": "q1"},
    )
    assert d.route_table_key == "background"
    assert d.resolved_llm_lane == "background"
    assert d.fallback_used is True


def test_spark_missing_disallows_chat_fallback_by_default() -> None:
    d = _resolve(
        options={"llm_lane": "spark", "allow_chat_fallback": True},
        body_route="quick",
        keys={"chat", "quick"},
        served_by={"chat": "c1", "quick": "q1"},
    )
    assert d.route_status == "missing_route"
    assert d.route_table_key is None


def test_background_metacog_alias() -> None:
    d = _resolve(
        options={"llm_lane": "background"},
        body_route=None,
        keys={"metacog", "chat"},
        served_by={"metacog": "m1", "chat": "c1"},
    )
    assert d.route_table_key == "metacog"
    assert d.resolved_llm_lane == "background"


def test_agent_prefers_agent_then_background() -> None:
    d = _resolve(
        options={"llm_lane": "agent"},
        body_route=None,
        keys={"agent", "background"},
        served_by={"agent": "a1", "background": "b1"},
    )
    assert d.route_table_key == "agent"
    d2 = _resolve(
        options={"llm_lane": "agent"},
        body_route=None,
        keys={"background"},
        served_by={"background": "b1"},
    )
    assert d2.route_table_key == "background"
    assert d2.fallback_used is True


def test_global_fallback_false_request_true_still_blocks() -> None:
    d = _resolve(
        options={"llm_lane": "background", "allow_chat_fallback": True},
        body_route="chat",
        keys={"chat"},
        served_by={"chat": "c1"},
    )
    assert d.route_status == "missing_route"


def test_no_request_option_can_reach_chat_through_a_lane_fallback() -> None:
    """The gateway's own chat fallback is gone: the pool spills a class across roles now."""
    d = _resolve(
        options={"llm_lane": "agent", "allow_chat_fallback": True},
        body_route="chat",
        keys={"chat", "quick"},
    )
    assert d.route_table_key is None
    assert d.route_status == "missing_route"
