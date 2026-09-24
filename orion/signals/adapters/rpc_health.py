"""RPC-health snapshots -> rpc_transport_health signals.

Step 3 of docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md.
Consumes RpcHealthSnapshotV1 envelopes published on orion:rpc_health:snapshot by each
producer service's own periodic drain of OrionBusAsync.get_rpc_health_snapshot().
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, Optional

from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrganClass, OrionOrganRegistryEntry, OrionSignalV1
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


# A publisher's sole/primary instance. Its organ_id carries no instance suffix, so a
# single-container service keeps one stable organ_id (e.g. rpc_health_cortex_orch).
PRIMARY_INSTANCE = "main"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    return _SLUG_RE.sub("_", value.strip().lower()).strip("_")


def _organ_id_for(service: str, instance: Optional[str]) -> Optional[str]:
    """organ_id keyed by (service, instance), e.g. ('cortex-exec', 'chat') ->
    'rpc_health_cortex_exec_chat'; ('cortex-orch', 'main'|None) -> 'rpc_health_cortex_orch'.

    Pass-through, not a whitelist (2026-09-24, A0 of
    docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md).
    The old two-service whitelist dropped every other producer's snapshot, and all four
    cortex-exec lane containers (instance=None) overwrote one shared slot, so a
    :background backlog was indistinguishable from :chat.

    Still one organ_id per producer, never a shared id: orion-signal-gateway's
    SignalWindow keys its current-state view by organ_id alone
    (services/orion-signal-gateway/app/signal_window.py), so a shared id would make every
    producer silently overwrite the previous one's entry.

    Service keys are the real `SERVICE_NAME` values ("cortex-exec", not
    "orion-cortex-exec" -- the original whitelist assumed the prefix and silently
    returned None on every real call until live verification caught it). An "orion-"
    prefix is still normalized away defensively. Returns None only when the payload has
    no service at all.
    """
    normalized = _slug((service or "").removeprefix("orion-"))
    if not normalized:
        return None
    organ_id = f"rpc_health_{normalized}"
    inst = _slug(instance or "")
    if inst and inst != PRIMARY_INSTANCE:
        organ_id = f"{organ_id}_{inst}"
    return organ_id


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
        instance = payload.get("instance")
        instance = str(instance) if instance else None
        resolved_organ_id = _organ_id_for(service, instance)
        if resolved_organ_id is None:
            return None
        # A registered organ (cortex-exec lanes, cortex-orch) supplies its class; an
        # unregistered producer still passes through as exogenous rather than being
        # dropped -- every downstream consumer tolerates an organ_id with no registry
        # entry (causal_helpers / processor OTEL parent lookup both no-op on None).
        entry = registry.get(resolved_organ_id) or ORGAN_REGISTRY.get(resolved_organ_id)
        organ_class = entry.organ_class if entry is not None else OrganClass.exogenous
        now = datetime.now(timezone.utc)
        window_end = str(payload.get("window_end") or "")
        src_id = f"{resolved_organ_id}:{window_end or int(now.timestamp())}"
        success = int(payload.get("success_count") or 0)
        timeout = int(payload.get("timeout_count") or 0)
        return OrionSignalV1(
            signal_id=make_signal_id(resolved_organ_id, src_id),
            organ_id=resolved_organ_id,
            organ_class=organ_class,
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
                f"rpc_health {service}{'/' + instance if instance else ''}: "
                f"success={success} timeout={timeout} "
                f"p95={payload.get('success_latency_ms_p95')}ms"
            ),
        )
