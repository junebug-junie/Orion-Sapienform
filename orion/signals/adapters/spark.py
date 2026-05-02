import hashlib
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


class SparkAdapter(OrionSignalAdapter):
    organ_id = "spark_introspector"

    def can_handle(self, channel: str, payload: dict) -> bool:
        return "spark" in channel and "biometrics" not in channel

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
        src_id = payload.get("correlation_id") or payload.get("id")
        now = datetime.now(timezone.utc)
        sig_id = (
            hashlib.sha256(f"{self.organ_id}:{src_id}".encode()).hexdigest()[:16]
            if src_id
            else uuid4().hex[:16]
        )
        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]
        return OrionSignalV1(
            signal_id=sig_id,
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind=entry.signal_kinds[0] if entry.signal_kinds else "unknown",
            dimensions={"level": 0.5, "confidence": 0.5},
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            notes=["stub adapter — not yet implemented"],
        )
