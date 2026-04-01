from __future__ import annotations

from app import chat_stance


class _Lookup:
    def __init__(self, subject: str, availability: str, state=None, unavailable_reason: str | None = None):
        self.subject = subject
        self.availability = availability
        self.state = state
        self.unavailable_reason = unavailable_reason


class _Repo:
    def __init__(self, lookups):
        self._lookups = lookups

    def list_latest(self, subjects, *, observer=None):
        return [self._lookups.get(s, _Lookup(s, "empty", None)) for s in subjects]

    def status(self):
        class _Status:
            backend = "graph"
            source_path = "graphdb:test"
            source_available = True

        return _Status()


def test_chat_stance_inputs_include_autonomy_summary_when_available(monkeypatch) -> None:
    from orion.autonomy.models import AutonomyStateV1

    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.9},
        active_drives=["coherence"],
        tension_kinds=["scope_sprawl"],
        source="graph",
    )
    repo = _Repo({"orion": _Lookup("orion", "available", state), "relationship": _Lookup("relationship", "empty", None)})
    monkeypatch.setattr(chat_stance, "build_autonomy_repository", lambda **_: repo)

    ctx = {"user_message": "help me synthesize this"}
    built = chat_stance.build_chat_stance_inputs(ctx)

    assert built["autonomy"]["summary"]["stance_hint"] == "favor synthesis and reduction"
    assert ctx["chat_autonomy_state"]["subject"] == "orion"
    assert "chat_autonomy_summary" in ctx


def test_chat_stance_unavailable_autonomy_keeps_behavior_stable(monkeypatch) -> None:
    repo = _Repo({})
    monkeypatch.setattr(chat_stance, "build_autonomy_repository", lambda **_: repo)

    ctx = {"user_message": "can you help"}
    chat_stance.build_chat_stance_inputs(ctx)
    fb = chat_stance.fallback_chat_stance_brief(ctx)

    assert fb.answer_strategy == "DirectAnswer"
    assert fb.task_mode == "direct_response"


def test_chat_stance_autonomy_debug_contains_unavailable_reason(monkeypatch) -> None:
    repo = _Repo({"orion": _Lookup("orion", "unavailable", None, unavailable_reason="query_error")})
    monkeypatch.setattr(chat_stance, "build_autonomy_repository", lambda **_: repo)

    ctx = {"user_message": "hello"}
    chat_stance.build_chat_stance_inputs(ctx)

    assert ctx["chat_autonomy_debug"]["orion"]["availability"] == "unavailable"
    assert ctx["chat_autonomy_debug"]["orion"]["unavailable_reason"] == "query_error"
    assert ctx["chat_autonomy_debug"]["_runtime"]["backend"] == "graph"
    assert "repository_status" in ctx["chat_autonomy_debug"]["_runtime"]
    assert ctx["chat_autonomy_backend"] == "graph"
    assert "chat_autonomy_repository_status" in ctx


def test_triage_mode_not_overridden_by_autonomy_hint(monkeypatch) -> None:
    repo = _Repo({})
    monkeypatch.setattr(chat_stance, "build_autonomy_repository", lambda **_: repo)

    ctx = {
        "user_message": "I'm super pissed; workflows are offline and gpu is dead",
        "chat_social_summary": {"social_posture": ["technical", "strained"], "relationship_facets": ["shared build"]},
        "chat_social_bridge_summary": {"posture": ["operational_frustration", "technical"], "hazards": []},
        "chat_autonomy_summary": {
            "stance_hint": "favor synthesis and reduction",
            "response_hazards": ["avoid contradictory framing"],
        },
    }
    fb = chat_stance.fallback_chat_stance_brief(ctx)
    assert fb.task_mode == "triage"
    assert fb.identity_salience == "low"
    assert "triage_operational_blockers_first" in fb.response_priorities
