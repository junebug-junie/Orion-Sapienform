"""Social relational updates → social_bond_state signal (Milestone B6)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id


class SocialMemoryAdapter(OrionSignalAdapter):
    organ_id = "social_memory"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if "social" in channel:
            return True
        if payload.get("platform") and payload.get("room_id"):
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
        continuity = payload.get("continuity_score") or payload.get("bond_score")
        repair = payload.get("repair_event") or payload.get("repair_needed")
        level = clamp01(float(continuity)) if continuity is not None else (0.7 if not repair else 0.4)
        valence = clamp01(float(payload.get("valence", 0.5)) if payload.get("valence") is not None else level)

        src_id = str(
            payload.get("update_id")
            or payload.get("correlation_id")
            or f"{payload.get('platform')}:{payload.get('room_id')}"
        )
        signal_kind = "social_repair_event" if repair else "social_bond_state"

        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]

        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind=signal_kind,
            dimensions={"level": level, "valence": valence, "trend": 0.5, "confidence": 0.8},
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary=f"social_memory {signal_kind} platform={payload.get('platform', 'unknown')}",
        )
