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


_KNOWN_SERVICE_ORGAN_IDS = {
    "cortex-exec": "rpc_health_cortex_exec",
    "cortex-orch": "rpc_health_cortex_orch",
}


def _organ_id_for_service(service: str) -> Optional[str]:
    """Per-service organ_id, e.g. 'cortex-exec' -> 'rpc_health_cortex_exec'.

    Deliberately NOT a single shared 'rpc_health' organ_id across every producer:
    orion-signal-gateway's SignalWindow keys its current-state view by organ_id alone
    (services/orion-signal-gateway/app/signal_window.py), so a shared id would make
    every producer's publish silently overwrite the previous producer's entry -- found
    in review of this step's first cut. Returns None for an unrecognized service so the
    caller can degrade to no signal rather than guess a wrong identity.

    Keys are the real `SERVICE_NAME` values (confirmed live: both services' settings.py
    default to "cortex-exec"/"cortex-orch", no "orion-" prefix -- NOT
    "orion-cortex-exec"/"orion-cortex-orch" as this function originally assumed, which
    made adapt() silently return None on every real call in production, caught only by
    live verification after deploy, not by review or the unit tests -- the test payloads
    used the same wrong assumption as the implementation). Normalizes an "orion-" prefix
    if present, defensively, in case that convention is ever used instead.
    """
    normalized = service.removeprefix("orion-") if service else service
    return _KNOWN_SERVICE_ORGAN_IDS.get(normalized)


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
        service = str(payload.get("service") or "")
        resolved_organ_id = _organ_id_for_service(service)
        if resolved_organ_id is None:
            return None
        entry = registry.get(resolved_organ_id) or ORGAN_REGISTRY.get(resolved_organ_id)
        if entry is None:
            return None
        now = datetime.now(timezone.utc)
        window_end = str(payload.get("window_end") or "")
        src_id = f"{resolved_organ_id}:{window_end or int(now.timestamp())}"
        success = int(payload.get("success_count") or 0)
        timeout = int(payload.get("timeout_count") or 0)
        return OrionSignalV1(
            signal_id=make_signal_id(resolved_organ_id, src_id),
            organ_id=resolved_organ_id,
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
