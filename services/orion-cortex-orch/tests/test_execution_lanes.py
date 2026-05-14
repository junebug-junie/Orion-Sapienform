from __future__ import annotations

from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest

from app.execution_lanes import resolve_execution_lane


def _req(
    *,
    mode: str = "brain",
    verb: str | None = None,
    options: dict | None = None,
) -> CortexClientRequest:
    return CortexClientRequest(
        mode=mode,  # type: ignore[arg-type]
        verb_name=verb,
        packs=[],
        options=options or {},
        context=CortexClientContext(
            messages=[],
            raw_user_text="hi",
            user_message="hi",
            session_id="s",
            user_id="u",
            trace_id="t",
            metadata={},
        ),
    )


def test_resolve_chat_general() -> None:
    d = resolve_execution_lane(_req(verb="chat_general"))
    assert d.lane == "chat" and d.reason == "verb_chat" and not d.explicit


def test_resolve_chat_quick() -> None:
    d = resolve_execution_lane(_req(verb="chat_quick"))
    assert d.lane == "chat" and d.reason == "verb_chat"


def test_resolve_introspect_spark() -> None:
    d = resolve_execution_lane(_req(verb="introspect_spark"))
    assert d.lane == "spark" and d.reason == "verb_spark"


def test_resolve_dream_cycle() -> None:
    d = resolve_execution_lane(_req(verb="dream_cycle"))
    assert d.lane == "background" and d.reason == "verb_background"


def test_resolve_log_orion_metacognition() -> None:
    d = resolve_execution_lane(_req(verb="log_orion_metacognition"))
    assert d.lane == "background" and d.reason == "verb_background"


def test_resolve_journal_compose() -> None:
    d = resolve_execution_lane(_req(verb="journal.compose"))
    assert d.lane == "background" and d.reason == "verb_background"


def test_explicit_lane_spark_wins_over_verb_chat() -> None:
    d = resolve_execution_lane(_req(verb="chat_general", options={"execution_lane": "spark"}))
    assert d.lane == "spark" and d.reason == "explicit_options" and d.explicit


def test_mode_brain_chat_lane_without_matching_verb() -> None:
    d = resolve_execution_lane(_req(mode="brain", verb="weird_verb"))
    assert d.lane == "chat" and d.reason == "mode_chat"


def test_mode_agent_background() -> None:
    d = resolve_execution_lane(_req(mode="agent", verb="agent_runtime"))
    assert d.lane == "background" and d.reason == "mode_background"


def test_fallback_background_auto_mode() -> None:
    d = resolve_execution_lane(_req(mode="auto", verb=None))
    assert d.lane == "background" and d.reason == "fallback_background"
