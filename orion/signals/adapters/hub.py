"""Hub chat turn → chat_turn exogenous signal (Milestone B5, optional root)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id


class HubAdapter(OrionSignalAdapter):
    organ_id = "hub"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if "chat.history.turn" in channel:
            return True
        if "chat:history:turn" in channel:
            return True
        if payload.get("turn_id") or payload.get("message_id"):
            return "prompt" in payload or "response" in payload or "session_id" in payload
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
        meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        corr = (
            payload.get("correlation_id")
            or payload.get("root_correlation_id")
            or meta.get("correlation_id")
        )
        src_id = str(corr or payload.get("turn_id") or payload.get("message_id") or "")
        if not src_id:
            src_id = f"hub-turn:{int(now.timestamp())}"

        has_response = 1.0 if (payload.get("response") or payload.get("assistant_text")) else 0.0
        meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        corr_prop = 1.0 if meta.get("correlation_id") or meta.get("root_correlation_id") else 0.5

        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind="chat_turn",
            dimensions={"confidence": clamp01(corr_prop), "level": has_response},
            causal_parents=[],
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary="hub chat_turn presence (no message text)",
        )
