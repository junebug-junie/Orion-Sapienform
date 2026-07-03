from __future__ import annotations

from pathlib import Path

from app.chat_stance import (
    _inject_prior_stance_to_inputs,
    compile_speech_contract,
    enforce_chat_stance_quality,
    strip_identity_recital_leadin,
    strip_transactional_closers,
)
from app.executor import _prior_stance_cache_get, _prior_stance_cache_set
from orion.schemas.chat_stance import ChatStanceBrief


def _relational_brief(**overrides) -> ChatStanceBrief:
    base = {
        "conversation_frame": "reflective",
        "task_mode": "reflective_dialogue",
        "identity_salience": "low",
        "user_intent": "Companion presence; mind off recovery.",
        "self_relevance": "Hold space; be curious.",
        "juniper_relevance": "Relational continuity matters this turn.",
        "active_relationship_facets": ["shared_history", "companionship"],
        "active_identity_facets": [],
        "active_growth_axes": [],
        "social_posture": ["presence"],
        "reflective_themes": ["recovery"],
        "active_tensions": ["existential"],
        "dream_motifs": [],
        "response_priorities": [
            "companion_presence",
            "situated_curiosity",
            "hold_space",
            "no_solutioning",
        ],
        "response_hazards": [
            "avoid_task_tracking",
            "avoid_next_steps",
            "avoid_transactional_closers",
        ],
        "answer_strategy": "RelationalHoldSpace",
        "stance_summary": "Be present; ask one situated question; do not solution.",
    }
    base.update(overrides)
    return ChatStanceBrief.model_validate(base)


def _instrumental_brief(**overrides) -> ChatStanceBrief:
    base = {
        "conversation_frame": "mixed",
        "task_mode": "direct_response",
        "identity_salience": "low",
        "user_intent": "direct question",
        "self_relevance": "answer",
        "juniper_relevance": "practical",
        "answer_strategy": "direct",
        "stance_summary": "short",
    }
    base.update(overrides)
    return ChatStanceBrief(**base)


def test_strip_identity_recital_leadin_skips_relational_turn() -> None:
    raw = "You're Juniper — the one building this with me. What's been the weirdest part?"
    brief = {
        "conversation_frame": "reflective",
        "task_mode": "reflective_dialogue",
    }
    stripped, changed = strip_identity_recital_leadin(
        raw,
        "someone to talk. im lonely",
        chat_stance_brief=brief,
    )
    assert changed is False
    assert stripped == raw


def test_enforce_preserves_relational_frame_only_brief() -> None:
    brief = ChatStanceBrief.model_validate(
        {
            "conversation_frame": "reflective",
            "task_mode": "direct_response",
            "identity_salience": "low",
            "user_intent": "Hold space.",
            "self_relevance": "x",
            "juniper_relevance": "Relational continuity matters.",
            "active_relationship_facets": ["companionship"],
            "response_priorities": ["companion_presence"],
            "response_hazards": [],
            "answer_strategy": "DirectAnswer",
            "stance_summary": "short",
        }
    )
    enriched, _ = enforce_chat_stance_quality(brief, {"user_message": "just talk to me"})
    assert enriched.active_relationship_facets == ["companionship"]
    assert enriched.juniper_relevance == "Relational continuity matters."


def test_enforce_preserves_relational_brief_on_non_identity_turn() -> None:
    brief = _relational_brief()
    ctx = {"user_message": "just be curious about it and take my mind off recovery"}

    enriched, _ = enforce_chat_stance_quality(brief, ctx)

    assert enriched.task_mode == "reflective_dialogue"
    assert enriched.conversation_frame == "reflective"
    assert "shared_history" in enriched.active_relationship_facets
    assert enriched.juniper_relevance == "Relational continuity matters this turn."
    assert "companion_presence" in enriched.response_priorities
    assert "avoid_identity_recital" not in enriched.response_priorities
    assert "identity_recital_on_ordinary_turn" not in enriched.response_hazards


def test_enforce_still_compresses_instrumental_direct_response() -> None:
    brief = ChatStanceBrief.model_validate(
        {
            "conversation_frame": "mixed",
            "task_mode": "direct_response",
            "identity_salience": "low",
            "user_intent": "Quick ack.",
            "self_relevance": "x",
            "juniper_relevance": "y",
            "active_relationship_facets": ["juniper_builder"],
            "response_priorities": ["answer_directly_first"],
            "response_hazards": [],
            "answer_strategy": "DirectAnswer",
            "stance_summary": "short",
        }
    )
    ctx = {"user_message": "hey"}

    enriched, _ = enforce_chat_stance_quality(brief, ctx)

    assert enriched.juniper_relevance == "Prioritize practical usefulness over relationship labels."
    assert enriched.active_relationship_facets == []
    assert "avoid_identity_recital" in enriched.response_priorities


def test_enforce_preserves_playful_exchange_frame() -> None:
    brief = _relational_brief(
        conversation_frame="playful_relational",
        task_mode="playful_exchange",
    )
    ctx = {"user_message": "someone to talk. im lonely"}

    enriched, _ = enforce_chat_stance_quality(brief, ctx)

    assert enriched.task_mode == "playful_exchange"
    assert enriched.active_relationship_facets


def test_stance_brief_prompt_has_connection_seek_vocabulary() -> None:
    prompt = Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    assert "interface_cost" in prompt
    assert "connection_seek" in prompt
    assert "Do not use keyword matching" in prompt
    assert "companion_presence" in prompt
    assert "LOW-BANDWIDTH / EMBODIED INTERACTION ASSESSMENT" not in prompt


def test_speech_prompt_relational_curiosity_overrides_attention_frame() -> None:
    prompt = Path("orion/cognition/prompts/chat_general.j2").read_text(encoding="utf-8")
    assert "reflective_dialogue" in prompt
    assert "playful_exchange" in prompt
    assert "advisory" in prompt.lower()
    assert "avoid_transactional_closers" in prompt or "avoid_next_steps" in prompt


def test_strip_transactional_closers_on_relational_turn() -> None:
    raw = (
        "I hear the weight of that, Juniper. 5:30am feels like a long time away, but it's coming. "
        "Let me know if you need anything to make the time pass more comfortably."
    )
    stripped, changed = strip_transactional_closers(raw, chat_stance_brief=_relational_brief())
    assert changed is True
    assert "Let me know" not in stripped
    assert stripped.endswith("it's coming.")


def test_strip_transactional_closers_skips_instrumental_turn() -> None:
    raw = "Patch is ready. Let me know when you're ready to deploy."
    brief = {
        "task_mode": "direct_response",
        "conversation_frame": "mixed",
        "response_hazards": [],
    }
    stripped, changed = strip_transactional_closers(raw, chat_stance_brief=brief)
    assert changed is False
    assert stripped == raw


def test_enforce_upgrades_companion_thread_continuation() -> None:
    brief = ChatStanceBrief(
        conversation_frame="mixed",
        task_mode="direct_response",
        identity_salience="low",
        user_intent="Vent continuation.",
        self_relevance="Answer directly.",
        juniper_relevance="Prioritize practical usefulness over relationship labels.",
        active_identity_facets=[],
        active_growth_axes=[],
        active_relationship_facets=[],
        social_posture=[],
        reflective_themes=[],
        active_tensions=[],
        dream_motifs=[],
        response_priorities=["direct_answer"],
        response_hazards=[],
        situation_relevance="background",
        temporal_context="now",
        audience_context="private",
        environmental_context="hospital",
        operational_context="none",
        situation_response_guidance=[],
        answer_strategy="Acknowledge and stay present.",
        stance_summary="Hold space.",
    )
    ctx = {
        "user_message": (
            "thanks, it's just hard... We have to be out of here by 5:30am and it will be a "
            "terrible night's sleep with nurses in and out."
        ),
        "message_history": [
            {"role": "user", "content": "just looking for a shoulder to talk. Can you help me keep my mind off?"},
            {
                "role": "assistant",
                "content": "I'm here, Juniper. Let's just sit with it — whatever it is.",
            },
        ],
    }
    enriched, _ = enforce_chat_stance_quality(brief, ctx)
    assert enriched.task_mode == "reflective_dialogue"
    assert "avoid_transactional_closers" in enriched.response_hazards
    assert "companion_presence" in enriched.response_priorities


def test_chat_stance_brief_new_fields_default_none() -> None:
    brief = _relational_brief()
    assert brief.interaction_regime is None
    assert brief.companion_closing_move is None


def test_chat_stance_brief_new_fields_roundtrip() -> None:
    brief = _relational_brief(
        interaction_regime="relational",
        companion_closing_move="end_with_a_wondering",
    )
    d = brief.model_dump(mode="json")
    assert d["interaction_regime"] == "relational"
    assert d["companion_closing_move"] == "end_with_a_wondering"
    restored = ChatStanceBrief(**d)
    assert restored.interaction_regime == "relational"
    assert restored.companion_closing_move == "end_with_a_wondering"


def test_compile_speech_contract_relational_with_closing_move() -> None:
    brief = _relational_brief(
        interaction_regime="relational",
        companion_closing_move="end_with_a_wondering",
    )
    contract = compile_speech_contract(brief)
    assert "companion turn" in contract
    assert "wondering" in contract
    assert "let me know" not in contract.lower()
    assert "if you need" not in contract.lower()


def test_compile_speech_contract_relational_default_no_move() -> None:
    brief = _relational_brief(
        interaction_regime="relational",
        response_priorities=["companion_presence", "hold_space"],
    )
    contract = compile_speech_contract(brief)
    assert "companion turn" in contract
    assert "next steps" in contract or "support closers" in contract


def test_compile_speech_contract_relational_situated_curiosity() -> None:
    brief = _relational_brief(
        interaction_regime="relational",
        response_priorities=["companion_presence", "situated_curiosity"],
    )
    contract = compile_speech_contract(brief)
    assert "grounded question" in contract
    assert "generic reversal" in contract


def test_compile_speech_contract_minimal() -> None:
    brief = _instrumental_brief(interaction_regime="minimal")
    contract = compile_speech_contract(brief)
    assert "short" in contract
    assert "replying" in contract


def test_compile_speech_contract_instrumental_direct() -> None:
    brief = _instrumental_brief(interaction_regime="instrumental")
    contract = compile_speech_contract(brief)
    assert "directly" in contract
    assert "blocker" not in contract


def test_compile_speech_contract_instrumental_triage() -> None:
    brief = _instrumental_brief(interaction_regime="instrumental", task_mode="triage")
    contract = compile_speech_contract(brief)
    assert "blocker" in contract


def test_compile_speech_contract_derives_regime_from_task_mode() -> None:
    brief = _relational_brief(interaction_regime=None)
    contract = compile_speech_contract(brief)
    assert "companion turn" in contract


def test_compile_speech_contract_all_closing_moves() -> None:
    moves = {
        "end_with_a_wondering": "wondering",
        "leave_space_without_offer": "offer",
        "ground_observation": "observation",
        "be_with_silence": "silence",
    }
    for move, expected_word in moves.items():
        brief = _relational_brief(interaction_regime="relational", companion_closing_move=move)
        contract = compile_speech_contract(brief)
        assert expected_word in contract, f"move={move!r}: expected {expected_word!r} in {contract!r}"


def test_prior_stance_cache_set_and_get() -> None:
    _prior_stance_cache_set("sess-abc", {"interaction_regime": "relational", "task_mode": "reflective_dialogue"})
    result = _prior_stance_cache_get("sess-abc")
    assert result is not None
    assert result["interaction_regime"] == "relational"


def test_prior_stance_cache_returns_none_for_missing_key() -> None:
    result = _prior_stance_cache_get("sess-does-not-exist-xyz")
    assert result is None


def test_prior_stance_cache_evicts_expired_entry(monkeypatch) -> None:
    # Override TTL to 0 to force immediate expiry. Patch the function's own module
    # namespace (__globals__) rather than a fresh `import app.executor`: under the
    # full suite the executor can be imported under duplicate module identities, and
    # __globals__ is the exact dict the helpers read _PRIOR_STANCE_TTL_SECONDS from.
    monkeypatch.setitem(_prior_stance_cache_get.__globals__, "_PRIOR_STANCE_TTL_SECONDS", 0)
    _prior_stance_cache_set("sess-ttl-test", {"interaction_regime": "relational"})
    import time as _t; _t.sleep(0.01)
    result = _prior_stance_cache_get("sess-ttl-test")
    assert result is None


def test_inject_prior_stance_to_inputs_when_present() -> None:
    ctx = {"prior_chat_stance_brief": {"interaction_regime": "relational", "task_mode": "reflective_dialogue"}}
    inputs: dict = {}
    _inject_prior_stance_to_inputs(ctx, inputs)
    assert inputs.get("prior_stance") == ctx["prior_chat_stance_brief"]


def test_inject_prior_stance_to_inputs_noop_when_absent() -> None:
    ctx: dict = {}
    inputs: dict = {}
    _inject_prior_stance_to_inputs(ctx, inputs)
    assert "prior_stance" not in inputs


def test_inject_prior_stance_to_inputs_noop_for_empty_dict() -> None:
    ctx = {"prior_chat_stance_brief": {}}
    inputs: dict = {}
    _inject_prior_stance_to_inputs(ctx, inputs)
    assert "prior_stance" not in inputs


def test_stance_brief_prompt_has_prior_stance_and_regime_fields() -> None:
    prompt = Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    assert "prior_stance" in prompt
    assert "interaction_regime" in prompt
    assert "companion_closing_move" in prompt
    assert "carry" in prompt.lower() or "carryforward" in prompt.lower() or "carry forward" in prompt.lower()


def test_inject_prior_stance_exposes_top_level_ctx_key() -> None:
    prior = {"interaction_regime": "relational", "task_mode": "reflective_dialogue"}
    ctx = {"prior_chat_stance_brief": prior}
    inputs: dict = {}
    _inject_prior_stance_to_inputs(ctx, inputs)
    assert inputs.get("prior_stance") == prior
    assert ctx.get("prior_stance") == prior


def test_prior_stance_reaches_stance_brief_render() -> None:
    from jinja2 import Template
    from app.executor import _prompt_render_ctx
    prior = {"interaction_regime": "relational", "task_mode": "reflective_dialogue"}
    ctx = {"prior_chat_stance_brief": prior}
    inputs: dict = {}
    _inject_prior_stance_to_inputs(ctx, inputs)
    render_ctx = _prompt_render_ctx(ctx)
    rendered = Template(
        Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    ).render(**render_ctx)
    assert "- prior_stance:" in rendered
    assert "relational" in rendered
    assert "carry" in rendered.lower()


def test_stance_brief_render_omits_prior_stance_when_absent() -> None:
    from jinja2 import Template
    from app.executor import _prompt_render_ctx
    render_ctx = _prompt_render_ctx({})
    rendered = Template(
        Path("orion/cognition/prompts/chat_stance_brief.j2").read_text(encoding="utf-8")
    ).render(**render_ctx)
    assert "- prior_stance:" not in rendered


def test_load_prior_stance_into_ctx_sets_top_level_from_cache() -> None:
    from app.executor import _load_prior_stance_into_ctx, _prior_stance_cache_set
    _prior_stance_cache_set("sess-load-test", {"interaction_regime": "relational", "task_mode": "reflective_dialogue"})
    ctx = {"session_id": "sess-load-test"}
    _load_prior_stance_into_ctx(ctx)
    assert ctx.get("prior_stance", {}).get("interaction_regime") == "relational"
    assert ctx.get("prior_chat_stance_brief", {}).get("interaction_regime") == "relational"


def test_load_prior_stance_into_ctx_noop_without_session_id() -> None:
    from app.executor import _load_prior_stance_into_ctx
    ctx: dict = {}
    _load_prior_stance_into_ctx(ctx)
    assert "prior_stance" not in ctx


def test_upgrade_brief_for_relational_continuation_sets_regime() -> None:
    from app.chat_stance import _upgrade_brief_for_relational_continuation
    brief = _instrumental_brief()
    upgraded = _upgrade_brief_for_relational_continuation(brief)
    assert upgraded.interaction_regime == "relational"
    assert upgraded.conversation_frame in {"reflective", "playful_relational"}


def test_compile_speech_contract_repair_concrete_overrides_relational() -> None:
    brief = _relational_brief(interaction_regime="relational", companion_closing_move="end_with_a_wondering")
    repair = {
        "mode": "repair_concrete",
        "rules": [
            "no broad architecture wandering",
            "include tests/acceptance checks",
        ],
    }
    contract = compile_speech_contract(brief, repair_contract=repair)
    assert "companion turn" not in contract.lower()
    assert "include tests/acceptance checks" in contract
    assert "no broad architecture wandering" in contract


def test_compile_speech_contract_concrete_bias_appends_to_instrumental() -> None:
    brief = _instrumental_brief(task_mode="direct_response")
    repair = {
        "mode": "concrete_bias",
        "rules": ["be more specific", "include next concrete action"],
    }
    contract = compile_speech_contract(brief, repair_contract=repair)
    assert contract.startswith("Answer directly.")
    assert "be more specific" in contract
    assert "include next concrete action" in contract


def test_compile_speech_contract_ignores_default_repair_contract() -> None:
    brief = _instrumental_brief()
    contract = compile_speech_contract(brief, repair_contract={"mode": "default"})
    assert contract == "Answer directly."

