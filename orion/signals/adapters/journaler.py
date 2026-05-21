"""Chat history / collapse writes → journal_entry signal (Milestone B6)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id


class JournalerAdapter(OrionSignalAdapter):
    organ_id = "journaler"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if "chat:history" in channel or "chat.history" in channel:
            return True
        if "collapse:sql-write" in channel or "journal" in channel:
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
        src_id = str(
            payload.get("turn_id")
            or payload.get("entry_id")
            or payload.get("correlation_id")
            or payload.get("event_id")
            or ""
        )
        if not src_id:
            src_id = f"journal:{int(now.timestamp())}"

        has_turn = 1.0 if payload.get("turn_id") or payload.get("message_id") else 0.0
        signal_kind = "journal_entry" if has_turn else "recall_event"

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
            dimensions={"level": has_turn or 0.6, "novelty": 0.5, "valence": 0.5, "confidence": 0.8},
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary=f"journaler {signal_kind} channel={channel.split(':')[-1] if ':' in channel else channel}",
        )
