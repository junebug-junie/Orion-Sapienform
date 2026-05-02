"""Biometrics adapter contract (spec §7.A) — lives under ``orion/signals/adapters/tests/``."""
from datetime import datetime, timezone

import pytest

from orion.signals.adapters.biometrics import BiometricsAdapter
from orion.signals.models import OrganClass
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


@pytest.fixture
def adapter() -> BiometricsAdapter:
    return BiometricsAdapter()


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


def test_adapt_leaves_otel_for_gateway(adapter: BiometricsAdapter, norm_ctx: NormalizationContext) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "athena",
        "correlation_id": "corr-1",
        "metrics": {
            "gpu_util": {"level": 0.7, "trend": 0.6, "volatility": 0.1, "spike_rate": 0.0},
        },
    }
    sig = adapter.adapt("biometrics.induction.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert sig is not None
    assert sig.organ_class == OrganClass.exogenous
    assert sig.otel_trace_id is None
    assert sig.otel_span_id is None
