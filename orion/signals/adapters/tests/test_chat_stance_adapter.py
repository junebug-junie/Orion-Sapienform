import pytest

from orion.signals.adapters import chat_stance as chat_stance_module
from orion.signals.adapters.chat_stance import ChatStanceAdapter
from orion.signals.models import OrganClass
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY

_BRIEF = {
    "conversation_frame": "technical",
    "task_mode": "direct_response",
    "identity_salience": "medium",
    "user_intent": "Understand mesh health signals",
    "self_relevance": "Operator debugging stance pipeline",
    "juniper_relevance": "Co-architect session",
    "answer_strategy": "Answer directly with evidence",
    "stance_summary": "Direct technical collaboration without generic assistant tone.",
}

_CLEAN_DEBUG = {
    "enforcement": {
        "fallback_invoked": False,
        "normalized_applied": False,
        "quality_modified": False,
        "semantic_fallback": False,
    },
    "raw": {"enforcement": {"parse_error": None}},
}

_DIRTY_DEBUG = {
    "enforcement": {
        "fallback_invoked": True,
        "normalized_applied": True,
        "quality_modified": False,
        "semantic_fallback": False,
    },
    "raw": {"enforcement": {"parse_error": None}},
}


def _real_wire_payload(*, correlation_id: str, session_id: str, brief: dict, debug: dict) -> dict:
    """Mirrors the actual production shape confirmed via a live orion:cognition:trace
    capture: ChatStanceBrief/ChatStanceDebug live nested in a step's result dict
    (PascalCase), and session_id lives under payload["metadata"]["session_id"] --
    never at the payload top level in real cortex-exec output."""
    return {
        "correlation_id": correlation_id,
        "metadata": {"session_id": session_id, "chat_stance_debug_present": True},
        "steps": [
            {"step_name": "recall_context", "result": {"RecallService": {"count": 1}}},
            {
                "step_name": "synthesize_chat_stance_brief",
                "result": {"ChatStanceBrief": brief, "ChatStanceDebug": debug},
            },
        ],
    }


@pytest.fixture
def adapter() -> ChatStanceAdapter:
    return ChatStanceAdapter()


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Deterministic stand-in for vector-host: text -> a small fixed vector keyed by content."""

    def _embed(text: str, **_kwargs):
        if not text or not text.strip():
            return None
        # Two distinct fixed vectors so "same text" -> identical, "different text" -> different.
        return [1.0, 0.0, 0.0] if "Direct technical" in text else [0.0, 1.0, 0.0]

    monkeypatch.setattr(chat_stance_module, "embed_text", _embed)
    yield


def test_chat_stance_adapter_from_real_wire_shape(adapter: ChatStanceAdapter, norm_ctx: NormalizationContext) -> None:
    """Real cortex-exec envelopes nest ChatStanceBrief/ChatStanceDebug under
    payload["steps"][i]["result"] and session_id under payload["metadata"] --
    confirmed via a live bus capture. The adapter must find data there, not just
    at a flat top-level shape no real producer emits."""
    payload = _real_wire_payload(correlation_id="c1", session_id="sess-real", brief=_BRIEF, debug=_CLEAN_DEBUG)
    assert adapter.can_handle("orion:cognition:trace", payload) is True
    signal = adapter.adapt("orion:cognition:trace", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "chat_stance"
    assert signal.dimensions["confidence"] == pytest.approx(1.0)
    assert "valence" not in signal.dimensions
    # First turn in this session -> no prior embedding yet.
    assert "coherence" not in signal.dimensions
    assert any("coherence_unavailable_first_turn_in_session" in n for n in signal.notes)

    # Second turn, same session, same stance text -> coherence should now be real.
    payload_2 = _real_wire_payload(correlation_id="c2", session_id="sess-real", brief=_BRIEF, debug=_CLEAN_DEBUG)
    signal_2 = adapter.adapt("orion:cognition:trace", payload_2, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal_2.dimensions["coherence"] == pytest.approx(1.0)


def test_chat_stance_adapter_from_brief_no_debug(adapter: ChatStanceAdapter, norm_ctx: NormalizationContext) -> None:
    payload = {
        "correlation_id": "corr-stance-1",
        "metadata": {"session_id": "sess-1"},
        "chat_stance_brief": _BRIEF,
    }
    signal = adapter.adapt("cortex.exec.request", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "chat_stance"
    assert signal.organ_class == OrganClass.endogenous
    assert signal.signal_kind == "chat_stance"
    assert "valence" not in signal.dimensions
    # No chat_stance_debug on this payload -> honest neutral confidence, not a task_mode lookup.
    assert signal.dimensions["confidence"] == pytest.approx(0.5)
    assert "no_repair_telemetry_available" in signal.notes
    # First turn in the session -> no prior embedding to compare against.
    assert "coherence" not in signal.dimensions
    assert any("coherence_unavailable_first_turn_in_session" in n for n in signal.notes)
    blob = f"{signal.summary} {signal.dimensions}"
    assert "stance_summary" not in blob
    assert "Understand mesh" not in blob
    assert "generic assistant" not in blob


def test_chat_stance_no_session_id_skips_coherence_without_fallback_bucket(
    adapter: ChatStanceAdapter, norm_ctx: NormalizationContext
) -> None:
    """No session_id anywhere on the payload -> coherence must be omitted, never
    silently compared via a shared 'global' bucket across unrelated sessions."""
    payload = {"correlation_id": "c1", "chat_stance_brief": _BRIEF}
    signal = adapter.adapt("cortex.exec.request", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert "coherence" not in signal.dimensions
    assert any("coherence_unavailable_no_session_id" in n for n in signal.notes)


def test_chat_stance_confidence_penalized_by_real_repair_telemetry(
    adapter: ChatStanceAdapter,
) -> None:
    clean_payload = {
        "correlation_id": "c1",
        "metadata": {"session_id": "s1"},
        "chat_stance_brief": _BRIEF,
        "chat_stance_debug": _CLEAN_DEBUG,
    }
    dirty_payload = {
        "correlation_id": "c2",
        "metadata": {"session_id": "s2"},
        "chat_stance_brief": _BRIEF,
        "chat_stance_debug": _DIRTY_DEBUG,
    }

    clean_signal = adapter.adapt("cortex.exec.request", clean_payload, ORGAN_REGISTRY, {}, NormalizationContext())
    dirty_signal = adapter.adapt("cortex.exec.request", dirty_payload, ORGAN_REGISTRY, {}, NormalizationContext())

    assert clean_signal.dimensions["confidence"] == pytest.approx(1.0)
    assert dirty_signal.dimensions["confidence"] == pytest.approx(1.0 - 0.40 - 0.05)
    assert any("confidence_penalty_fallback_invoked" in n for n in dirty_signal.notes)
    assert any("confidence_penalty_normalized_applied" in n for n in dirty_signal.notes)


def test_chat_stance_coherence_uses_prior_turn_same_session(
    adapter: ChatStanceAdapter, norm_ctx: NormalizationContext
) -> None:
    session_payload = {"correlation_id": "c1", "metadata": {"session_id": "sess-1"}, "chat_stance_brief": _BRIEF}
    first = adapter.adapt("cortex.exec.request", session_payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert "coherence" not in first.dimensions

    same_stance_payload = {"correlation_id": "c2", "metadata": {"session_id": "sess-1"}, "chat_stance_brief": _BRIEF}
    second = adapter.adapt("cortex.exec.request", same_stance_payload, ORGAN_REGISTRY, {}, norm_ctx)
    # Same stance text as the prior turn -> identical embedding -> coherence == 1.0.
    assert second.dimensions["coherence"] == pytest.approx(1.0)

    different_brief = {**_BRIEF, "stance_summary": "Something else entirely.", "answer_strategy": "Different tack."}
    different_payload = {
        "correlation_id": "c3",
        "metadata": {"session_id": "sess-1"},
        "chat_stance_brief": different_brief,
    }
    third = adapter.adapt("cortex.exec.request", different_payload, ORGAN_REGISTRY, {}, norm_ctx)
    # Orthogonal fixed vector for non-matching text -> cosine 0 -> rescaled to 0.5.
    assert third.dimensions["coherence"] == pytest.approx(0.5)


def test_chat_stance_coherence_scoped_per_session(adapter: ChatStanceAdapter, norm_ctx: NormalizationContext) -> None:
    payload_a = {"correlation_id": "c1", "metadata": {"session_id": "sess-a"}, "chat_stance_brief": _BRIEF}
    adapter.adapt("cortex.exec.request", payload_a, ORGAN_REGISTRY, {}, norm_ctx)

    payload_b = {"correlation_id": "c2", "metadata": {"session_id": "sess-b"}, "chat_stance_brief": _BRIEF}
    signal_b = adapter.adapt("cortex.exec.request", payload_b, ORGAN_REGISTRY, {}, norm_ctx)
    # A different session has no prior embedding of its own, even though sess-a just ran.
    assert "coherence" not in signal_b.dimensions
    assert any("coherence_unavailable_first_turn_in_session" in n for n in signal_b.notes)


def test_chat_stance_session_cache_is_bounded(adapter: ChatStanceAdapter, norm_ctx: NormalizationContext) -> None:
    """A long-running gateway process must not accumulate one embedding per
    session forever -- oldest sessions get evicted past the tracked cap."""
    original_cap = chat_stance_module._MAX_TRACKED_SESSIONS
    chat_stance_module._MAX_TRACKED_SESSIONS = 3
    try:
        for i in range(5):
            payload = {
                "correlation_id": f"c{i}",
                "metadata": {"session_id": f"sess-{i}"},
                "chat_stance_brief": _BRIEF,
            }
            adapter.adapt("cortex.exec.request", payload, ORGAN_REGISTRY, {}, norm_ctx)
        tracked = norm_ctx.get_value("chat_stance", chat_stance_module._SESSION_ORDER_KEY)
        assert tracked == ["sess-2", "sess-3", "sess-4"]
        assert norm_ctx.get_value("chat_stance", "embedding:sess-0") is None
        assert norm_ctx.get_value("chat_stance", "embedding:sess-4") is not None
    finally:
        chat_stance_module._MAX_TRACKED_SESSIONS = original_cap


def test_chat_stance_debug_only(adapter: ChatStanceAdapter, norm_ctx: NormalizationContext) -> None:
    payload = {
        "correlation_id": "corr-stance-2",
        "verb": "chat_general",
        "metadata": {"chat_stance_debug_present": True},
    }
    signal = adapter.adapt("cognition.trace", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert "valence" not in signal.dimensions
    assert "coherence" not in signal.dimensions
    assert signal.dimensions["confidence"] == pytest.approx(0.5)
    assert any("stance_debug_only_no_brief" in n for n in signal.notes)
    assert any("stance_debug_flag_only_no_object" in n for n in signal.notes)


def test_chat_stance_debug_dict_present_without_brief(adapter: ChatStanceAdapter, norm_ctx: NormalizationContext) -> None:
    payload = {"correlation_id": "corr-stance-3", "chat_stance_debug": _DIRTY_DEBUG}
    signal = adapter.adapt("cognition.trace", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["confidence"] == pytest.approx(1.0 - 0.40 - 0.05)
    assert any("confidence_penalty_fallback_invoked" in n for n in signal.notes)
