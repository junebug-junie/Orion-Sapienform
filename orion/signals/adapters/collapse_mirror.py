"""Collapse mirror intake → cognitive_collapse signal (Milestone B6)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.schemas.collapse_mirror import DEFAULT_CHANGE_TYPE_BY_ENTRY_TYPE
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id

_TYPE_LEVEL = {
    "turbulence": 0.85,
    "glitch": 0.9,
    "epiphany": 0.75,
    "flow": 0.35,
    "idle": 0.2,
}


class CollapseMirrorAdapter(OrionSignalAdapter):
    organ_id = "collapse_mirror"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if "collapse" in channel:
            return True
        if payload.get("type") and payload.get("observer"):
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
        entry_type = str(payload.get("type") or "flow").lower()
        change_type = str(payload.get("change_type") or DEFAULT_CHANGE_TYPE_BY_ENTRY_TYPE.get(entry_type, "stabilizing"))
        level = _TYPE_LEVEL.get(entry_type, 0.5)
        scores = payload.get("change_type_scores") or {}
        if isinstance(scores, dict) and scores:
            level = clamp01(max(float(v) for v in scores.values() if isinstance(v, (int, float))))

        src_id = str(payload.get("event_id") or payload.get("id") or payload.get("correlation_id") or "")
        if not src_id:
            src_id = f"collapse:{entry_type}:{int(now.timestamp())}"

        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]

        signal_kind = "metacog_event" if entry_type == "idle" else "cognitive_collapse"

        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind=signal_kind,
            dimensions={"level": level, "valence": clamp01(level - 0.5), "confidence": 0.85},
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary=f"collapse type={entry_type} change={change_type}",
            notes=[],
        )
