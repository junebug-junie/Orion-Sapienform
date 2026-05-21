from datetime import datetime, timezone

import pytest

from orion.signals.adapters.spark import SparkAdapter
from orion.signals.models import OrganClass
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


@pytest.fixture
def adapter() -> SparkAdapter:
    return SparkAdapter()


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


def test_spark_adapter_signal_v1(adapter: SparkAdapter, norm_ctx: NormalizationContext) -> None:
    now = datetime.now(timezone.utc)
    payload = {
        "signal_type": "equilibrium",
        "intensity": 0.42,
        "valence_delta": -0.1,
        "coherence_delta": -0.05,
        "as_of_ts": now.isoformat(),
        "ttl_ms": 15000,
        "source_service": "orion-equilibrium-service",
    }
    signal = adapter.adapt("spark.signal.v1", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "spark_introspector"
    assert signal.organ_class == OrganClass.endogenous
    assert signal.signal_kind == "spark_signal"
    assert signal.dimensions["level"] == 0.42
    assert "stub adapter" not in " ".join(signal.notes)
