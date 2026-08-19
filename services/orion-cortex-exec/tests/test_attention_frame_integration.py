from __future__ import annotations

from pathlib import Path

import pytest

from app.chat_stance import build_chat_stance_debug_payload, build_chat_stance_inputs


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.asyncio
async def test_feature_flag_off_preserves_chat_stance_input_shape(monkeypatch) -> None:
    monkeypatch.delenv("ORION_CURIOSITY_FRAME_ENABLED", raising=False)
    ctx = {"user_message": "I am planning around Zephyr Bridge.", "skip_unified_beliefs": True}
    built = await build_chat_stance_inputs(ctx)
    assert "attention_frame" not in built
    assert "chat_attention_frame" not in ctx


@pytest.mark.asyncio
async def test_feature_flag_on_adds_attention_frame(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CURIOSITY_FRAME_ENABLED", "true")
    ctx = {"user_message": "I am planning around Zephyr Bridge.", "skip_unified_beliefs": True}
    built = await build_chat_stance_inputs(ctx)
    assert "attention_frame" in built
    assert ctx["chat_attention_frame"]["schema_version"] == "attention.frame.v1"
    assert built["attention_frame"]["open_loops"]


@pytest.mark.asyncio
async def test_frame_build_persists_salience_trace_when_enabled(monkeypatch) -> None:
    import app.chat_stance as chat_stance_module

    calls = []

    async def _fake_persist(frame):
        calls.append(frame)
        return True

    monkeypatch.setenv("ORION_CURIOSITY_FRAME_ENABLED", "true")
    monkeypatch.setattr(chat_stance_module, "persist_chat_attention_salience_trace", _fake_persist)
    ctx = {"user_message": "I am planning around Zephyr Bridge.", "skip_unified_beliefs": True}
    await build_chat_stance_inputs(ctx)
    assert len(calls) == 1
    assert calls[0].open_loops  # the real built frame, not a stub


@pytest.mark.asyncio
async def test_frame_disabled_never_calls_persist(monkeypatch) -> None:
    import app.chat_stance as chat_stance_module

    calls = []

    async def _fake_persist(frame):
        calls.append(frame)
        return True

    monkeypatch.delenv("ORION_CURIOSITY_FRAME_ENABLED", raising=False)
    monkeypatch.setattr(chat_stance_module, "persist_chat_attention_salience_trace", _fake_persist)
    ctx = {"user_message": "I am planning around Zephyr Bridge.", "skip_unified_beliefs": True}
    await build_chat_stance_inputs(ctx)
    assert calls == []


@pytest.mark.asyncio
async def test_persist_failure_does_not_clear_attention_frame_ctx(monkeypatch) -> None:
    import app.chat_stance as chat_stance_module

    async def _boom(frame):
        raise RuntimeError("db down")

    monkeypatch.setenv("ORION_CURIOSITY_FRAME_ENABLED", "true")
    monkeypatch.setattr(chat_stance_module, "persist_chat_attention_salience_trace", _boom)
    ctx = {"user_message": "I am planning around Zephyr Bridge.", "skip_unified_beliefs": True}
    built = await build_chat_stance_inputs(ctx)
    # A trace-writer exception must not undo an already-successful frame build --
    # the ctx keys and the returned inputs both still carry the real frame.
    assert "attention_frame" in built
    assert "chat_attention_frame" in ctx
    assert ctx["chat_attention_frame"]["schema_version"] == "attention.frame.v1"


def test_debug_payload_exposes_attention_frame() -> None:
    attention_frame = {
        "schema_version": "attention.frame.v1",
        "open_loops": [{"id": "loop-1", "description": "Zephyr Bridge", "target_type": "plan"}],
        "live_unknowns": ["Zephyr Bridge"],
        "candidate_actions": [],
        "selected_action": {"action_type": "watch", "open_loop_id": "loop-1", "score": 0.5},
        "suppressions": [{"reason": "user_needs_direct_answer", "target_ref": "current_turn"}],
        "deferred_items": ["loop-1"],
        "debug": {"enabled": True},
    }
    ctx = {
        "user_message": "what changed?",
        "memory_digest": "",
        "chat_stance_inputs": {
            "identity": {"orion": [], "juniper": [], "response_policy": []},
            "concept_induction": {"self": [], "relationship": [], "growth": [], "tension": []},
            "social": {"social_posture": [], "relationship_facets": [], "hazards": []},
            "social_bridge": {"posture": [], "hazards": [], "framing": [], "summary": []},
            "reflective": {"themes": [], "tensions": [], "dream_motifs": []},
            "autonomy": {"summary": {}, "debug": {}},
            "reasoning_summary": {},
            "situation": {},
            "attention_frame": attention_frame,
        },
    }
    payload = build_chat_stance_debug_payload(
        ctx=ctx,
        synthesized_brief={"task_mode": "direct_response"},
        final_brief={"task_mode": "direct_response"},
        fallback_invoked=False,
        normalized_applied=False,
        semantic_fallback=False,
        quality_modified=False,
    )
    assert payload["source_inputs"]["attention_frame"]["selected_action"]["action_type"] == "watch"
    assert payload["final_prompt_contract"]["attention_frame"]["schema_version"] == "attention.frame.v1"


def test_prompt_contracts_include_attention_policy() -> None:
    stance_prompt = (REPO_ROOT / "orion" / "cognition" / "prompts" / "chat_stance_brief.j2").read_text(encoding="utf-8")
    speech_prompt = (REPO_ROOT / "orion" / "cognition" / "prompts" / "chat_general.j2").read_text(encoding="utf-8")
    assert "attention_frame: {{ chat_attention_frame }}" in stance_prompt
    assert "curiosity:ask_selected" in stance_prompt
    assert "attention_frame: {{ chat_attention_frame }}" in speech_prompt
    assert "ask only when attention_frame.selected_action.action_type is ask" in speech_prompt
