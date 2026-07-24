"""RPC-health snapshots -> rpc_transport_health signals.

Step 3 of docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md.
Consumes RpcHealthSnapshotV1 envelopes published on orion:rpc_health:snapshot by each
producer service's own periodic drain of OrionBusAsync.get_rpc_health_snapshot().
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id


def _level(payload: dict) -> float:
    """Fraction of calls that succeeded in this window. An empty window (no calls at
    all) is healthy-by-absence, not a failure -- level=1.0, but confidence stays low
    to reflect the lack of real evidence (see _confidence below)."""
    success = int(payload.get("success_count") or 0)
    timeout = int(payload.get("timeout_count") or 0)
    total = success + timeout
    if total == 0:
        return 1.0
    return clamp01(success / total)


def _confidence(payload: dict) -> float:
    """More real calls observed this window = more confidence in `level`. Caps out at
    a modest sample size rather than requiring a huge window, since a healthy quiet
    period (fewer calls) is still meaningful evidence, not noise."""
    success = int(payload.get("success_count") or 0)
    timeout = int(payload.get("timeout_count") or 0)
    total = success + timeout
    if total == 0:
        return 0.1
    return clamp01(0.1 + 0.9 * min(total, 20) / 20.0)


def _latency_level(payload: dict) -> float:
    p95 = payload.get("success_latency_ms_p95")
    if p95 is None:
        return 0.5
    return clamp01(1.0 - min(float(p95), 30_000) / 30_000.0)


class RpcHealthAdapter(OrionSignalAdapter):
    organ_id = "rpc_health"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if channel == "rpc_health.snapshot.v1":
            return True
        return "rpc_health:snapshot" in channel

    def adapt(
        self,
        channel: str,
        payload: dict,
        registry: Dict[str, OrionOrganRegistryEntry],
        prior_signals: Dict[str, OrionSignalV1],
        norm_ctx: NormalizationContext,
    ) -> Optional[OrionSignalV1]:
        entry = registry.get(self.organ_id) or ORGAN_REGISTRY.get(self.organ_id)
        if entry is None:
            return None
        now = datetime.now(timezone.utc)
        service = str(payload.get("service") or "unknown")
        window_end = str(payload.get("window_end") or "")
        src_id = f"rpc_health:{service}:{window_end or int(now.timestamp())}"
        success = int(payload.get("success_count") or 0)
        timeout = int(payload.get("timeout_count") or 0)
        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind="rpc_transport_health",
            dimensions={
                "level": _level(payload),
                "confidence": _confidence(payload),
                "latency_level": _latency_level(payload),
            },
            causal_parents=[],
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary=(
                f"rpc_health {service}: success={success} timeout={timeout} "
                f"p95={payload.get('success_latency_ms_p95')}ms"
            ),
        )
