import pytest

from orion.signals.adapters.autonomy import AutonomyAdapter
from orion.signals.models import OrganClass
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


@pytest.fixture
def adapter() -> AutonomyAdapter:
    return AutonomyAdapter()


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


def test_autonomy_adapter_from_summary(adapter: AutonomyAdapter, norm_ctx: NormalizationContext) -> None:
    payload = {
        "correlation_id": "corr-auto-1",
        "chat_autonomy_summary": {
            "stance_hint": "favor synthesis and reduction",
            "dominant_drive": "coherence",
            "top_drives": ["coherence", "continuity"],
            "active_tensions": ["scope_sprawl"],
            "raw_state_present": True,
        },
    }
    signal = adapter.adapt("cortex.exec.request", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "autonomy"
    assert signal.organ_class == OrganClass.endogenous
    assert signal.signal_kind == "autonomy_state"
    assert signal.dimensions["pressure_coherence"] >= 0.65
    assert "stub adapter" not in " ".join(signal.notes)


def test_autonomy_adapter_from_state_alone_yields_neutral_low_confidence_pressures(
    adapter: AutonomyAdapter, norm_ctx: NormalizationContext
) -> None:
    """AutonomyStateV1 no longer carries dominant_drive/drive_pressures/active_drives/
    tension_kinds (removed 2026-07-30, chore/delete-orion-drives Wave 2a). A bare
    state payload (no accompanying summary_dict hint) must not crash and must not
    report a fabricated high-confidence pressure reading."""
    payload = {
        "correlation_id": "corr-auto-2",
        "chat_autonomy_state": {
            "subject": "orion",
            "model_layer": "graph",
            "entity_id": "orion:default",
            "source": "graph",
        },
    }
    signal = adapter.adapt("orion:cortex:exec:request", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["pressure_relational"] == 0.5
    assert signal.dimensions["confidence"] == 0.3
    assert any("no drive_pressures" in note for note in signal.notes)


def test_autonomy_adapter_prefers_summary_dict_hint_even_with_state_present(
    adapter: AutonomyAdapter, norm_ctx: NormalizationContext
) -> None:
    """When both a full state and a summary_dict hint are present, the summary_dict's
    dominant_drive/top_drives (an untyped dict independent of the AutonomyStateV1
    schema) is the only source that can still carry a real per-drive hint."""
    payload = {
        "correlation_id": "corr-auto-3",
        "chat_autonomy_state": {
            "subject": "orion",
            "model_layer": "graph",
            "entity_id": "orion:default",
            "source": "graph",
        },
        "chat_autonomy_summary": {
            "dominant_drive": "relational",
            "top_drives": ["relational"],
            "raw_state_present": True,
        },
    }
    signal = adapter.adapt("orion:cortex:exec:request", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["pressure_relational"] >= 0.65
    assert signal.dimensions["confidence"] == 0.75
