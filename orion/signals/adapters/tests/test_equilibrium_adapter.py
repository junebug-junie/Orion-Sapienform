"""Equilibrium adapter — canonical location ``orion/signals/adapters/tests/`` (phase-2)."""

from datetime import datetime, timezone

import pytest

from orion.signals.adapters.biometrics import BiometricsAdapter
from orion.signals.adapters.equilibrium import EquilibriumAdapter
from orion.signals.causal_helpers import with_missed_parent_notes
from orion.signals.models import OrganClass
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


@pytest.fixture
def adapter() -> EquilibriumAdapter:
    return EquilibriumAdapter()


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


def test_equilibrium_adapter_snapshot_v1(adapter: EquilibriumAdapter, norm_ctx: NormalizationContext) -> None:
    from datetime import datetime, timezone

    payload = {
        "source_service": "orion-equilibrium-service",
        "producer_boot_id": "boot-eq",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grace_multiplier": 2.0,
        "windows_sec": [60, 300],
        "services": [],
        "distress_score": 0.18,
        "zen_score": 0.82,
        "correlation_id": "eq-corr",
    }
    signal = adapter.adapt("equilibrium.snapshot.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.signal_kind == "mesh_health"
    assert signal.dimensions["level"] > 0.5
    assert "stub adapter" not in " ".join(signal.notes)


def test_produces_valid_signal(adapter: EquilibriumAdapter, norm_ctx: NormalizationContext) -> None:
    payload = {"correlation_id": "abc123", "zen_score": 0.8, "distress_score": 0.2}
    signal = adapter.adapt("equilibrium.snapshot.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "equilibrium"
    assert signal.organ_class == OrganClass.hybrid


def test_causal_parents_resolved_from_prior(adapter: EquilibriumAdapter, norm_ctx: NormalizationContext) -> None:
    bio_adapter = BiometricsAdapter()
    bio_payload = {
        "metrics": {"gpu_util": {"level": 0.6, "trend": 0.5, "volatility": 0.1, "spike_rate": 0.0}}
    }
    bio_signal = bio_adapter.adapt("biometrics.induction.v1", bio_payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert bio_signal is not None

    prior = {"biometrics": bio_signal}
    eq_payload = {
        "correlation_id": "eq123",
        "source_service": "orion-equilibrium-service",
        "producer_boot_id": "boot",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grace_multiplier": 2.0,
        "windows_sec": [60],
        "zen_score": 0.7,
        "distress_score": 0.3,
    }
    eq_signal = adapter.adapt("equilibrium.snapshot.v1", eq_payload, ORGAN_REGISTRY, prior, norm_ctx)
    assert eq_signal is not None
    assert bio_signal.signal_id in eq_signal.causal_parents


def test_missed_parent_note_when_biometrics_absent(adapter: EquilibriumAdapter, norm_ctx: NormalizationContext) -> None:
    eq_payload = {
        "correlation_id": "eq456",
        "source_service": "orion-equilibrium-service",
        "producer_boot_id": "boot2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grace_multiplier": 2.0,
        "windows_sec": [60],
        "zen_score": 0.7,
        "distress_score": 0.3,
    }
    eq_signal = adapter.adapt("equilibrium.snapshot.v1", eq_payload, ORGAN_REGISTRY, {}, norm_ctx)
    enriched = with_missed_parent_notes(eq_signal, {}, ORGAN_REGISTRY)
    assert any("missed causal link" in n for n in enriched.notes)
    assert "biometrics" in enriched.notes[-1]
