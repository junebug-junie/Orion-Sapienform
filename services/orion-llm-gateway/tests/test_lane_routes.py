from __future__ import annotations

from app.lane_routes import resolve_llm_lane_route


def _resolve(
    *,
    options: dict | None,
    body_route: str | None,
    keys: set[str],
    served_by: dict[str, str | None],
    **kwargs: object,
) -> object:
    defaults = {
        "llm_lane_default": "chat",
        "llm_route_default": "chat",
        "llm_allow_background_to_chat_fallback": False,
        "llm_route_spark_served_by": None,
        "llm_route_background_served_by": None,
        "llm_route_agent_served_by": None,
    }
    defaults.update(kwargs)
    return resolve_llm_lane_route(
        options,
        body_route,
        llm_lane_default=str(defaults["llm_lane_default"]),
        llm_route_default=str(defaults["llm_route_default"]),
        llm_allow_background_to_chat_fallback=bool(defaults["llm_allow_background_to_chat_fallback"]),
        llm_route_spark_served_by=defaults["llm_route_spark_served_by"],  # type: ignore[arg-type]
        llm_route_background_served_by=defaults["llm_route_background_served_by"],  # type: ignore[arg-type]
        llm_route_agent_served_by=defaults["llm_route_agent_served_by"],  # type: ignore[arg-type]
        route_table_keys=keys,
        route_served_by=served_by,
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
        llm_allow_background_to_chat_fallback=False,
    )
    assert d.route_status == "disallowed_chat_fallback"
    assert d.route_table_key is None


def test_spark_emergency_chat_only_when_both_flags() -> None:
    d = _resolve(
        options={"llm_lane": "spark", "allow_chat_fallback": True},
        body_route="quick",
        keys={"chat", "quick"},
        served_by={"chat": "c1", "quick": "q1"},
        llm_allow_background_to_chat_fallback=True,
    )
    assert d.route_table_key in {"chat", "quick"}
    assert d.resolved_llm_lane == "chat"
    assert d.fallback_used is True
    assert "emergency_chat_fallback" in d.reason


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


def test_served_by_label_match() -> None:
    d = _resolve(
        options={"llm_lane": "spark"},
        body_route=None,
        keys={"foo", "bar"},
        served_by={"foo": "my-spark", "bar": "other"},
        llm_route_spark_served_by="my-spark",
    )
    assert d.route_table_key == "foo"


def test_global_fallback_false_request_true_still_blocks() -> None:
    d = _resolve(
        options={"llm_lane": "background", "allow_chat_fallback": True},
        body_route="chat",
        keys={"chat"},
        served_by={"chat": "c1"},
        llm_allow_background_to_chat_fallback=False,
    )
    assert d.route_status == "disallowed_chat_fallback"
