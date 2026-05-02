"""Tests for the biometrics adapter — reference adapter test suite."""
import pytest
from datetime import datetime, timezone

from orion.signals.adapters.biometrics import BiometricsAdapter
from orion.signals.models import OrganClass
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


@pytest.fixture
def adapter():
    return BiometricsAdapter()


@pytest.fixture
def norm_ctx():
    return NormalizationContext()


def make_induction_payload(metrics: dict | None = None) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "athena",
        "service_name": "orion-biometrics",
        "service_version": "0.1.0",
        "window_sec": 30,
        "metrics": metrics or {
            "gpu_util": {"level": 0.7, "trend": 0.6, "volatility": 0.1, "spike_rate": 0.0},
            "cpu": {"level": 0.4, "trend": 0.5, "volatility": 0.05, "spike_rate": 0.0},
            "mem": {"level": 0.6, "trend": 0.5, "volatility": 0.02, "spike_rate": 0.0},
        },
    }


class TestCanHandle:
    def test_accepts_biometrics_induction_kind(self, adapter):
        assert adapter.can_handle("biometrics.induction.v1", {"metrics": {}}) is True

    def test_accepts_biometrics_channel(self, adapter):
        assert adapter.can_handle("orion:biometrics:induction", {"metrics": {}}) is True

    def test_rejects_unrelated_channel(self, adapter):
        assert adapter.can_handle("orion:recall:result", {}) is False


class TestAdapt:
    def test_produces_valid_signal(self, adapter, norm_ctx):
        payload = make_induction_payload()
        signal = adapter.adapt("biometrics.induction.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
        assert signal is not None
        assert signal.organ_id == "biometrics"
        assert signal.organ_class == OrganClass.exogenous
        assert signal.signal_kind is not None
        assert len(signal.dimensions) > 0

    def test_dimensions_in_range(self, adapter, norm_ctx):
        payload = make_induction_payload()
        signal = adapter.adapt("biometrics.induction.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
        assert signal is not None
        for key, val in signal.dimensions.items():
            if "level" in key or "volatility" in key or "confidence" in key:
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_exogenous_has_empty_causal_parents(self, adapter, norm_ctx):
        payload = make_induction_payload()
        signal = adapter.adapt("biometrics.induction.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
        assert signal is not None
        assert signal.causal_parents == []

    def test_graceful_degradation_on_empty_payload(self, adapter, norm_ctx):
        signal = adapter.adapt("biometrics.induction.v1", {}, ORGAN_REGISTRY, {}, norm_ctx)
        assert signal is not None
        assert signal.dimensions.get("confidence", 1.0) <= 0.2
        assert len(signal.notes) > 0

    def test_repeated_identical_input_stable_level(self, adapter, norm_ctx):
        """Repeated same input should converge to a stable level."""
        payload = make_induction_payload({"gpu_util": {"level": 0.7, "trend": 0.5, "volatility": 0.0, "spike_rate": 0.0}})
        levels = []
        for _ in range(30):
            signal = adapter.adapt("biometrics.induction.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
            if signal:
                levels.append(signal.dimensions.get("gpu_util_level", signal.dimensions.get("level", None)))
        valid = [l for l in levels if l is not None]
        if len(valid) >= 5:
            # Last few should be close to each other — convergence
            last5 = valid[-5:]
            assert max(last5) - min(last5) < 0.15, f"Level did not converge: {last5}"
