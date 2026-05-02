from datetime import datetime, timezone
from typing import Dict, Optional
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.signal_ids import make_signal_id
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


class SocialRoomBridgeAdapter(OrionSignalAdapter):
    organ_id = "social_room_bridge"

    def can_handle(self, channel: str, payload: dict) -> bool:
        return "social_room_bridge" in channel

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
        src_raw = payload.get("correlation_id") or payload.get("id")
        src_id = str(src_raw) if src_raw is not None else None
        now = datetime.now(timezone.utc)
        sig_id = make_signal_id(self.organ_id, src_id)
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
