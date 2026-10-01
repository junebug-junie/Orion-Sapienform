from __future__ import annotations

from pathlib import Path

import pytest

from app.chat_stance import build_chat_stance_debug_payload, build_chat_stance_inputs


REPO_ROOT = Path(__file__).resolve().parents[3]


async def _fake_populate_zephyr_bridge_signal(ctx) -> None:
    """Stand-in for populate_current_turn_llm_signals(): the real function
    makes a bus RPC call (no bus is bound in these unit tests, so it would
    fail-open to []). Injects the same "Zephyr Bridge" candidate the deleted
    LegacyRegexSignalDetector used to extract via regex, so these tests keep
    exercising a populated attention frame rather than an empty one."""
    ctx["current_turn_llm_signals"] = [{"phrase": "Zephyr Bridge", "type": "concept"}]


def _human_chat_ctx() -> dict:
    # A real Hub chat turn: the probe only runs on turns with a human message
    # (see current_turn_llm_signals.human_chat_turn_reason).
    return {
        "user_message": "I am planning around Zephyr Bridge.",
        "skip_unified_beliefs": True,
        "verb": "stance_react",
        "stance_inputs": {"user_message": "I am planning around Zephyr Bridge.", "utterance_origin": "juniper"},
    }


@pytest.mark.asyncio
async def test_feature_flag_off_preserves_chat_stance_input_shape(monkeypatch) -> None:
    monkeypatch.delenv("ORION_CURIOSITY_FRAME_ENABLED", raising=False)
    ctx = _human_chat_ctx()
    built = await build_chat_stance_inputs(ctx)
    assert "attention_frame" not in built
    assert "chat_attention_frame" not in ctx


@pytest.mark.asyncio
async def test_feature_flag_on_adds_attention_frame(monkeypatch) -> None:
    import app.chat_stance as chat_stance_module

    monkeypatch.setenv("ORION_CURIOSITY_FRAME_ENABLED", "true")
    monkeypatch.setattr(
        chat_stance_module, "populate_current_turn_llm_signals", _fake_populate_zephyr_bridge_signal
    )
    ctx = _human_chat_ctx()
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
    monkeypatch.setattr(
        chat_stance_module, "populate_current_turn_llm_signals", _fake_populate_zephyr_bridge_signal
    )
    ctx = _human_chat_ctx()
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
    ctx = _human_chat_ctx()
    await build_chat_stance_inputs(ctx)
    assert calls == []


@pytest.mark.asyncio
async def test_persist_failure_does_not_clear_attention_frame_ctx(monkeypatch) -> None:
    import app.chat_stance as chat_stance_module

    async def _boom(frame):
        raise RuntimeError("db down")

    monkeypatch.setenv("ORION_CURIOSITY_FRAME_ENABLED", "true")
    monkeypatch.setattr(chat_stance_module, "persist_chat_attention_salience_trace", _boom)
    monkeypatch.setattr(
        chat_stance_module, "populate_current_turn_llm_signals", _fake_populate_zephyr_bridge_signal
    )
    ctx = _human_chat_ctx()
    built = await build_chat_stance_inputs(ctx)
    # A trace-writer exception must not undo an already-successful frame build --
    # the ctx keys and the returned inputs both still carry the real frame.
    assert "attention_frame" in built
    assert "chat_attention_frame" in ctx
    assert ctx["chat_attention_frame"]["schema_version"] == "attention.frame.v1"


@pytest.mark.asyncio
async def test_llm_signal_populate_failure_does_not_break_turn(monkeypatch) -> None:
    """populate_current_turn_llm_signals() is fail-open by contract and
    should never raise, but chat_stance.py wraps the call in its own
    try/except as defense-in-depth -- a bug there must not prevent the
    frame build from running at all (it degrades to zero LLM-sourced
    signals, same as a real RPC failure)."""
    import app.chat_stance as chat_stance_module

    async def _boom(ctx) -> None:
        raise RuntimeError("unexpected populate bug")

    monkeypatch.setenv("ORION_CURIOSITY_FRAME_ENABLED", "true")
    monkeypatch.setattr(chat_stance_module, "populate_current_turn_llm_signals", _boom)
    ctx = _human_chat_ctx()
    built = await build_chat_stance_inputs(ctx)
    assert "attention_frame" in built
    assert ctx["current_turn_llm_signals"] == []


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


@pytest.mark.asyncio
async def test_real_chat_turn_calls_probe(monkeypatch) -> None:
    import app.chat_stance as chat_stance_module

    calls = []

    async def _spy(ctx) -> None:
        calls.append(ctx.get("verb"))
        await _fake_populate_zephyr_bridge_signal(ctx)

    monkeypatch.setenv("ORION_CURIOSITY_FRAME_ENABLED", "true")
    monkeypatch.setattr(chat_stance_module, "populate_current_turn_llm_signals", _spy)
    ctx = _human_chat_ctx()
    await build_chat_stance_inputs(ctx)
    assert calls == ["stance_react"]
    assert ctx["current_turn_llm_signals"] == [{"phrase": "Zephyr Bridge", "type": "concept"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ctx_extra",
    [
        # journal.compose as dispatched by orion-actions/journaler (cortex-exec-background).
        {"verb": "journal.compose", "options": {"source": "orion-actions", "policy_dispatch_only": True}},
        # render_scene, an Orion-initiated skill dispatch.
        {"verb": "skills.imagination.render_scene.v1", "options": {"policy_dispatch_only": True}},
        {"verb": "log_orion_metacognition"},
        # A unified turn with no human origin (endogenous outreach / reading).
        {"verb": "stance_react", "stance_inputs": {"user_message": "x"}},
        # A unified turn Orion started (curiosity investigation).
        {"verb": "stance_react", "stance_inputs": {"user_message": "x", "utterance_origin": "orion"}},
        # A collapse-mirror reply: unified turn, no utterance_origin (skipped on purpose).
        {"verb": "stance_react", "stance_inputs": {"user_message": "Juniper just submitted this Collapse Mirror"}},
        # Harness finalize leg of a turn: no stance_inputs, non-chat verb.
        {"verb": "harness_finalize_reflect"},
        # A scheduled workflow sent as chat_general by orion-actions' scheduler.
        {"verb": "chat_general", "options": {"source": "orion-actions", "policy_dispatch_only": True}},
    ],
)
async def test_autonomous_turn_never_calls_probe(monkeypatch, ctx_extra) -> None:
    import app.chat_stance as chat_stance_module

    calls = []

    async def _spy(ctx) -> None:
        calls.append(ctx.get("verb"))

    monkeypatch.setenv("ORION_CURIOSITY_FRAME_ENABLED", "true")
    monkeypatch.setattr(chat_stance_module, "populate_current_turn_llm_signals", _spy)
    ctx = {"user_message": "Compose today's journal entry.", "skip_unified_beliefs": True, **ctx_extra}
    built = await build_chat_stance_inputs(ctx)
    assert calls == []
    assert ctx["current_turn_llm_signals"] == []
    read = ctx["current_turn_llm_read"]
    assert read["ok"] is False and read["wants_direct_answer"] is None
    assert read["skipped"] == "not_human_turn"
    # Frame still builds; the skip is visible and fails closed like any missing read.
    assert built["attention_frame"]["debug"]["turn_read_skipped"] == "not_human_turn"
    assert built["attention_frame"]["debug"]["direct_turn_cause"] == "unavailable"
