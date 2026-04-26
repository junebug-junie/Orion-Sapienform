"""Parity: cortex_result.metadata (incl. turn_effect) is merged into Hub spark_meta for bus/concept-induction."""

from __future__ import annotations

from uuid import UUID

from scripts.chat_history import build_chat_turn_envelope
from orion.schemas.metacognitive_trace import MetacognitiveTraceV1


def test_chat_turn_spark_meta_carries_cortex_turn_effect_like_websocket_merge():
    gateway_meta = {
        "turn_effect": {"turn": {"novelty": 0.12, "coherence": -0.05}},
        "turn_effect_evidence": {"phi_before": {"novelty": 0.1}},
    }
    spark_meta = {
        "mode": "brain",
        "trace_verb": "chat_general",
        **gateway_meta,
    }
    cid = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    env = build_chat_turn_envelope(
        prompt="hi",
        response="hello",
        session_id="s1",
        correlation_id=cid,
        user_id="u1",
        source_label="test",
        spark_meta=spark_meta,
        turn_id=str(cid),
    )
    dumped = env.payload.model_dump(mode="json")
    sm = dumped.get("spark_meta") or {}
    assert sm.get("turn_effect", {}).get("turn", {}).get("novelty") == 0.12
    assert sm.get("turn_effect_evidence", {}).get("phi_before", {}).get("novelty") == 0.1


def test_chat_turn_spark_meta_preserves_turn_effect_for_quick_trace_verb():
    spark_meta = {
        "mode": "brain",
        "trace_verb": "chat_quick",
        "turn_effect": {"turn": {"coherence": -0.08}},
        "turn_effect_status": "present",
    }
    cid = UUID("11111111-2222-3333-4444-555555555555")
    env = build_chat_turn_envelope(
        prompt="hey",
        response="yo",
        session_id="s2",
        correlation_id=cid,
        user_id="u1",
        source_label="test",
        spark_meta=spark_meta,
        turn_id=str(cid),
    )
    sm = (env.payload.model_dump(mode="json").get("spark_meta") or {})
    assert sm.get("trace_verb") == "chat_quick"
    assert sm.get("turn_effect", {}).get("turn", {}).get("coherence") == -0.08
    assert sm.get("turn_effect_status") == "present"


def test_chat_turn_envelope_coerces_reasoning_trace_to_schema_model():
    cid = UUID("aaaaaaaa-1111-2222-3333-ffffffffffff")
    env = build_chat_turn_envelope(
        prompt="q",
        response="a",
        session_id="s3",
        correlation_id=cid,
        user_id="u1",
        source_label="test",
        turn_id=str(cid),
        reasoning_trace={"trace_role": "reasoning", "trace_stage": "post_answer", "content": "trace body", "model": "m1"},
    )
    assert isinstance(env.payload.reasoning_trace, MetacognitiveTraceV1)
    assert env.payload.reasoning_trace.content == "trace body"
