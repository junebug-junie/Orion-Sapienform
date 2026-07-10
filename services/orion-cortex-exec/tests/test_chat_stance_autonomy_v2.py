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


from datetime import datetime, timezone

from orion.core.schemas.reasoning import ClaimV1
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.reasoning import InMemoryReasoningRepository


def _claim() -> ClaimV1:
    return ClaimV1(
        anchor_scope="orion",
        subject_ref="project:orion_sapienform",
        status="canonical",
        authority="local_inferred",
        confidence=0.9,
        salience=0.8,
        novelty=0.4,
        risk_tier="low",
        observed_at=datetime.now(timezone.utc),
        provenance={
            "evidence_refs": ["ev:1"],
            "source_channel": "orion:test",
            "source_kind": "unit",
            "producer": "pytest",
        },
        claim_text="Reasoning continuity is strong.",
        claim_kind="identity_signal",
    )


def test_chat_stance_empty_repo_omits_reasoning_quality_evidence(monkeypatch) -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.05},
        active_drives=["coherence"],
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
    )
    monkeypatch.setattr(chat_stance, "_load_autonomy_state", lambda _ctx: _fake_autonomy_bundle(state))
    monkeypatch.setenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "true")
    ctx: dict = {"user_message": "hello", "correlation_id": "c-omit"}
    chat_stance.build_chat_stance_inputs(ctx)
    debug = ctx.get("chat_autonomy_evidence_debug") or {}
    omitted = debug.get("omitted") or []
    assert any(o.get("kind") == "reasoning_quality" and o.get("reason") == "empty_upstream" for o in omitted)
    v2 = ctx.get("chat_autonomy_state_v2") or {}
    kinds = [e.get("kind") for e in (v2.get("evidence_refs") or [])]
    assert "reasoning_quality" not in kinds


def test_chat_stance_social_locals_reach_reducer(monkeypatch) -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive=None,
        drive_pressures={"relational": 0.0, "coherence": 0.0},
        active_drives=[],
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
    )
    monkeypatch.setattr(chat_stance, "_load_autonomy_state", lambda _ctx: _fake_autonomy_bundle(state))

    def _fake_social(_beliefs, _ctx):
        return (
            {"social_posture": [], "hazards": ["cooldown_active"], "relationship_facets": []},
            {"posture": [], "hazards": ["cooldown_active"], "framing": [], "summary": []},
        )

    monkeypatch.setattr(chat_stance, "_project_social_from_beliefs", _fake_social)
    monkeypatch.setenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "true")
    ctx: dict = {"user_message": "ping", "correlation_id": "c-haz"}
    # Intentionally do NOT pre-seed chat_social_bridge_summary — ordering bug regression.
    assert "chat_social_bridge_summary" not in ctx
    chat_stance.build_chat_stance_inputs(ctx)
    v2 = ctx["chat_autonomy_state_v2"]
    summaries = [e.get("summary") for e in (v2.get("evidence_refs") or []) if e.get("kind") == "relational_signal"]
    assert "cooldown_active" in summaries
    assert v2["drive_pressures"]["relational"] > 0.0
    assert isinstance(ctx.get("chat_autonomy_evidence_debug"), dict)
    assert isinstance(ctx.get("chat_autonomy_tension_debug"), dict)


def test_chat_stance_reasoning_upstream_emits_quality(monkeypatch) -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.05, "predictive": 0.05},
        active_drives=["coherence"],
        tension_kinds=[],
        goal_headlines=[],
        source="graph",
    )
    monkeypatch.setattr(chat_stance, "_load_autonomy_state", lambda _ctx: _fake_autonomy_bundle(state))
    monkeypatch.setenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "true")

    repo = InMemoryReasoningRepository()
    # Empty-ish path that still has an artifact but compiler may still recommend fallback
    # depending on subject_refs — force fallback via monkeypatch after compile if needed.
    repo.write_artifacts(
        ReasoningWriteRequestV1(
            context=ReasoningWriteContextV1(
                source_family="manual",
                source_kind="unit",
                source_channel="orion:test",
                producer="pytest",
            ),
            artifacts=[_claim()],
        )
    )

    original_compile = chat_stance._compile_reasoning_summary

    def _compile_force_fallback(ctx):
        out = original_compile(ctx)
        summary = dict(out.get("summary") or {})
        summary["fallback_recommended"] = True
        out = dict(out)
        out["summary"] = summary
        return out

    monkeypatch.setattr(chat_stance, "_compile_reasoning_summary", _compile_force_fallback)
    ctx: dict = {"user_message": "who?", "reasoning_repository": repo}
    chat_stance.build_chat_stance_inputs(ctx)
    v2 = ctx["chat_autonomy_state_v2"]
    kinds = [e.get("kind") for e in (v2.get("evidence_refs") or [])]
    assert "reasoning_quality" in kinds
    assert v2["drive_pressures"]["coherence"] > 0.05
