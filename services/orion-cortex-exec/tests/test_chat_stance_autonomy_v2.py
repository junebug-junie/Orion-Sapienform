from __future__ import annotations

from orion.autonomy.models import AutonomyStateV1
from orion.autonomy.summary import summarize_autonomy_state

from app import chat_stance


def _fake_autonomy_bundle(state: AutonomyStateV1):
    summary = summarize_autonomy_state(state)
    return {
        "lookups": [],
        "state": state,
        "backend": "graph",
        "selected_subject": "orion",
        "repository_status": {"backend": "graph", "source_path": "graphdb:test", "source_available": True},
        "summary": summary,
        "debug": {
            "orion": {"availability": "available", "present": True, "unavailable_reason": None, "subqueries": {}},
            "_runtime": {},
        },
    }


def test_chat_stance_autonomy_v2_ctx_and_inputs_when_enabled(monkeypatch) -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.4},
        active_drives=["coherence"],
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
    )
    monkeypatch.setattr(chat_stance, "_load_autonomy_state", lambda _ctx: _fake_autonomy_bundle(state))
    monkeypatch.setenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "true")
    ctx: dict = {"user_message": "hello", "correlation_id": "c-v2"}
    built = chat_stance.build_chat_stance_inputs(ctx)
    assert isinstance(ctx.get("chat_autonomy_state_v2"), dict)
    assert ctx["chat_autonomy_state_v2"].get("schema_version") == "autonomy.state.v2"
    assert "confidence" in ctx["chat_autonomy_state_v2"]
    assert isinstance(ctx.get("chat_autonomy_state_delta"), dict)
    assert "subject" in ctx["chat_autonomy_state_delta"]
    assert built["autonomy"]["state_v2"] == ctx["chat_autonomy_state_v2"]
    assert built["autonomy"]["delta"] == ctx["chat_autonomy_state_delta"]


def test_chat_stance_autonomy_v2_absent_when_disabled(monkeypatch) -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.4},
        active_drives=["coherence"],
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
    )
    monkeypatch.setattr(chat_stance, "_load_autonomy_state", lambda _ctx: _fake_autonomy_bundle(state))
    monkeypatch.delenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", raising=False)
    ctx = {"user_message": "hello"}
    chat_stance.build_chat_stance_inputs(ctx)
    assert "chat_autonomy_state_v2" not in ctx
    assert "chat_autonomy_state_delta" not in ctx


def test_chat_stance_autonomy_v2_reducer_exception_swallowed(monkeypatch) -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.4},
        active_drives=["coherence"],
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
    )
    monkeypatch.setattr(chat_stance, "_load_autonomy_state", lambda _ctx: _fake_autonomy_bundle(state))
    monkeypatch.setenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "true")

    def _boom(*_a, **_k):
        raise RuntimeError("reducer failure")

    monkeypatch.setattr(chat_stance, "reduce_autonomy_state", _boom)
    ctx: dict = {"user_message": "hello"}
    built = chat_stance.build_chat_stance_inputs(ctx)
    assert "chat_autonomy_state_v2" not in ctx
    assert "state_v2" not in built["autonomy"]
    assert "delta" not in built["autonomy"]
    assert isinstance(ctx.get("chat_autonomy_state"), dict)
    assert isinstance(ctx.get("chat_autonomy_summary"), dict)
