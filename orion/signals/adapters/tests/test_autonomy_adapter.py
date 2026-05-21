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


def test_autonomy_adapter_from_state(adapter: AutonomyAdapter, norm_ctx: NormalizationContext) -> None:
    payload = {
        "correlation_id": "corr-auto-2",
        "chat_autonomy_state": {
            "subject": "orion",
            "model_layer": "graph",
            "entity_id": "orion:default",
            "dominant_drive": "relational",
            "active_drives": ["relational", "coherence"],
            "drive_pressures": {"relational": 0.8, "coherence": 0.4},
            "tension_kinds": [],
            "source": "graph",
        },
    }
    signal = adapter.adapt("orion:cortex:exec:request", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["pressure_relational"] == 0.8
