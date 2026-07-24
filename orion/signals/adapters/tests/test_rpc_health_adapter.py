from datetime import datetime, timezone

import pytest

from orion.signals.adapters.rpc_health import RpcHealthAdapter
from orion.signals.models import OrganClass
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


@pytest.fixture
def adapter() -> RpcHealthAdapter:
    return RpcHealthAdapter()


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


def _payload(**overrides) -> dict:
    now = datetime.now(timezone.utc)
    base = {
        "service": "orion-cortex-exec",
        "node": "athena",
        "instance": None,
        "window_start": now.isoformat(),
        "window_end": now.isoformat(),
        "success_count": 18,
        "timeout_count": 2,
        "success_latency_ms_p50": 12.0,
        "success_latency_ms_p95": 40.0,
        "success_latency_ms_max": 55.0,
        "timeout_elapsed_ms_max": 5000.0,
        "channel_counts": {"orion:cortex:exec:request": 20},
        "truncated": False,
    }
    base.update(overrides)
    return base


def test_can_handle_raw_channel(adapter: RpcHealthAdapter) -> None:
    assert adapter.can_handle("orion:rpc_health:snapshot", {}) is True


def test_can_handle_envelope_kind(adapter: RpcHealthAdapter) -> None:
    assert adapter.can_handle("rpc_health.snapshot.v1", {}) is True


def test_can_handle_rejects_unrelated_channel(adapter: RpcHealthAdapter) -> None:
    assert adapter.can_handle("orion:cortex:exec:request", {}) is False


def test_adapt_produces_signal(adapter: RpcHealthAdapter, norm_ctx: NormalizationContext) -> None:
    signal = adapter.adapt("orion:rpc_health:snapshot", _payload(), ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "rpc_health_cortex_exec"
    assert signal.organ_class == OrganClass.exogenous
    assert signal.signal_kind == "rpc_transport_health"
    assert 0.0 <= signal.dimensions["level"] <= 1.0
    assert 0.0 <= signal.dimensions["confidence"] <= 1.0
    assert 0.0 <= signal.dimensions["latency_level"] <= 1.0


def test_adapt_all_success_is_healthy(adapter: RpcHealthAdapter, norm_ctx: NormalizationContext) -> None:
    payload = _payload(success_count=25, timeout_count=0)
    signal = adapter.adapt("orion:rpc_health:snapshot", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["level"] == 1.0


def test_adapt_all_timeout_is_unhealthy(adapter: RpcHealthAdapter, norm_ctx: NormalizationContext) -> None:
    payload = _payload(success_count=0, timeout_count=10)
    signal = adapter.adapt("orion:rpc_health:snapshot", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["level"] == 0.0


def test_adapt_empty_window_is_low_confidence_not_unhealthy(
    adapter: RpcHealthAdapter, norm_ctx: NormalizationContext
) -> None:
    """An empty window (no real RPC calls observed) must not be reported as a failure --
    that's the exact degenerate 'measures nothing but looks fine/bad' shape this whole
    redesign exists to avoid. level stays neutral-healthy; confidence drops instead."""
    payload = _payload(success_count=0, timeout_count=0)
    signal = adapter.adapt("orion:rpc_health:snapshot", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.dimensions["level"] == 1.0
    assert signal.dimensions["confidence"] < 0.5


def test_adapt_falls_back_to_global_registry_when_passed_registry_lacks_entry(
    adapter: RpcHealthAdapter, norm_ctx: NormalizationContext
) -> None:
    """Matches the established fallback pattern already used by every sibling adapter
    (registry.get(...) or ORGAN_REGISTRY.get(...)) -- a caller-passed registry missing this
    organ still resolves via the module-level ORGAN_REGISTRY, not a hard failure."""
    signal = adapter.adapt("orion:rpc_health:snapshot", _payload(), {}, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "rpc_health_cortex_exec"


def test_adapt_uses_distinct_organ_id_per_service(
    adapter: RpcHealthAdapter, norm_ctx: NormalizationContext
) -> None:
    """The bug found in review: a shared organ_id across producers makes each publish
    overwrite the previous producer's SignalWindow entry. Two different services must
    resolve to two different organ_ids."""
    exec_signal = adapter.adapt(
        "orion:rpc_health:snapshot", _payload(service="orion-cortex-exec"), ORGAN_REGISTRY, {}, norm_ctx
    )
    orch_signal = adapter.adapt(
        "orion:rpc_health:snapshot", _payload(service="orion-cortex-orch"), ORGAN_REGISTRY, {}, norm_ctx
    )
    assert exec_signal is not None and orch_signal is not None
    assert exec_signal.organ_id == "rpc_health_cortex_exec"
    assert orch_signal.organ_id == "rpc_health_cortex_orch"
    assert exec_signal.organ_id != orch_signal.organ_id


def test_adapt_unknown_service_degrades_to_none(
    adapter: RpcHealthAdapter, norm_ctx: NormalizationContext
) -> None:
    signal = adapter.adapt(
        "orion:rpc_health:snapshot", _payload(service="orion-some-future-service"), ORGAN_REGISTRY, {}, norm_ctx
    )
    assert signal is None


def test_adapt_is_deterministic_for_same_source_event(
    adapter: RpcHealthAdapter, norm_ctx: NormalizationContext
) -> None:
    payload = _payload()
    s1 = adapter.adapt("orion:rpc_health:snapshot", payload, ORGAN_REGISTRY, {}, norm_ctx)
    s2 = adapter.adapt("orion:rpc_health:snapshot", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert s1 is not None and s2 is not None
    assert s1.signal_id == s2.signal_id


def test_registry_entry_shape() -> None:
    for organ_id in ("rpc_health_cortex_exec", "rpc_health_cortex_orch"):
        entry = ORGAN_REGISTRY[organ_id]
        assert entry.organ_class == OrganClass.exogenous
        assert "orion:rpc_health:snapshot" in entry.bus_channels
        assert "rpc_transport_health" in entry.signal_kinds
