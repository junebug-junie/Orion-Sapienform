"""Cortex gateway request → route_decision signal (Milestone B5)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id


class CortexGatewayAdapter(OrionSignalAdapter):
    organ_id = "cortex_gateway"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if "cortex:gateway:request" in channel:
            return True
        if payload.get("verb") and payload.get("context"):
            return True
        if payload.get("mode") and (payload.get("context") or payload.get("recall")):
            return True
        return False

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
        corr = payload.get("correlation_id") or payload.get("_envelope_correlation_id")
        src_id = str(corr) if corr else None
        mode = str(payload.get("mode") or "auto")
        verb = str(payload.get("verb") or payload.get("verb_name") or "")
        recall = payload.get("recall") if isinstance(payload.get("recall"), dict) else {}
        recall_on = 1.0 if recall.get("enabled", True) else 0.0

        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]

        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind="route_decision",
            dimensions={
                "confidence": 0.85 if verb else 0.55,
                "level": recall_on,
            },
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary=f"gateway route mode={mode} verb={verb or 'unset'}",
        )
