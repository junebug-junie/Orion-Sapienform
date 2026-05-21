import pytest

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


@pytest.fixture
def adapter() -> ChatStanceAdapter:
    return ChatStanceAdapter()


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


def test_chat_stance_adapter_from_brief(adapter: ChatStanceAdapter, norm_ctx: NormalizationContext) -> None:
    payload = {"correlation_id": "corr-stance-1", "chat_stance_brief": _BRIEF}
    signal = adapter.adapt("cortex.exec.request", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "chat_stance"
    assert signal.organ_class == OrganClass.endogenous
    assert signal.signal_kind == "chat_stance"
    assert signal.dimensions["coherence"] > 0.5
    assert "stub adapter" not in " ".join(signal.notes)
    blob = f"{signal.summary} {signal.dimensions}"
    assert "stance_summary" not in blob
    assert "Understand mesh" not in blob
    assert "generic assistant" not in blob


def test_chat_stance_debug_only(adapter: ChatStanceAdapter, norm_ctx: NormalizationContext) -> None:
    payload = {
        "correlation_id": "corr-stance-2",
        "verb": "chat_general",
        "metadata": {"chat_stance_debug_present": True},
    }
    signal = adapter.adapt("cognition.trace", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["confidence"] == 0.4
    assert any("stance_debug_only" in n for n in signal.notes)
