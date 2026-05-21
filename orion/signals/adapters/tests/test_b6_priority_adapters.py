import pytest

from orion.signals.adapters.collapse_mirror import CollapseMirrorAdapter
from orion.signals.adapters.journaler import JournalerAdapter
from orion.signals.adapters.social_memory import SocialMemoryAdapter
from orion.signals.adapters.world_pulse import WorldPulseAdapter
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


def test_collapse_mirror_adapter(norm_ctx: NormalizationContext) -> None:
    adapter = CollapseMirrorAdapter()
    payload = {
        "event_id": "cm-1",
        "observer": "orion",
        "trigger": "gpu_spike",
        "type": "turbulence",
        "emergent_entity": "stress_field",
        "summary": "Long narrative text must not appear in signal summary field mapping",
        "mantra": "breathe",
    }
    signal = adapter.adapt("collapse.mirror.entry", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.signal_kind == "cognitive_collapse"
    assert signal.dimensions["level"] > 0.5
    assert "Long narrative" not in (signal.summary or "")
    assert "stub adapter" not in " ".join(signal.notes)


def test_journaler_adapter(norm_ctx: NormalizationContext) -> None:
    adapter = JournalerAdapter()
    signal = adapter.adapt(
        "chat.history.turn",
        {"turn_id": "jt-1", "session_id": "s-1"},
        ORGAN_REGISTRY,
        {},
        norm_ctx,
    )
    assert signal is not None
    assert signal.organ_id == "journaler"
    assert signal.signal_kind == "journal_entry"


def test_social_memory_adapter(norm_ctx: NormalizationContext) -> None:
    adapter = SocialMemoryAdapter()
    signal = adapter.adapt(
        "social.relational.update.v1",
        {"platform": "orion", "room_id": "main", "continuity_score": 0.82},
        ORGAN_REGISTRY,
        {},
        norm_ctx,
    )
    assert signal is not None
    assert signal.signal_kind == "social_bond_state"
    assert signal.dimensions["level"] > 0.5


def test_world_pulse_adapter(norm_ctx: NormalizationContext) -> None:
    adapter = WorldPulseAdapter()
    signal = adapter.adapt(
        "world_pulse.situation.brief.upsert.v1",
        {
            "topic_id": "topic-1",
            "title": "Regional weather",
            "scope": "local",
            "category": "environment",
            "current_assessment": "Storms approaching coastal grid.",
            "confidence": 0.77,
            "status": "developing",
            "last_updated": "2026-05-20T12:00:00+00:00",
            "first_seen_at": "2026-05-19T12:00:00+00:00",
            "created_at": "2026-05-19T12:00:00+00:00",
            "updated_at": "2026-05-20T12:00:00+00:00",
        },
        ORGAN_REGISTRY,
        {},
        norm_ctx,
    )
    assert signal is not None
    assert signal.signal_kind == "situation_state"
    assert "Storms approaching" not in (signal.summary or "")
