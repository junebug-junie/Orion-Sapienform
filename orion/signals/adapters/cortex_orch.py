"""Cortex orch request → plan_resolution signal (Milestone B5)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id


class CortexOrchAdapter(OrionSignalAdapter):
    organ_id = "cortex_orch"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if channel in ("cortex.orch.request", "cortex.orch.plan"):
            return True
        if "cortex:request" in channel and payload.get("verb"):
            return True
        if payload.get("packs") is not None and payload.get("mode"):
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
        packs = payload.get("packs") or []
        pack_count = len(packs) if isinstance(packs, list) else 0
        verb = str(payload.get("verb") or payload.get("verb_name") or "")

        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]
        gw = prior_signals.get("cortex_gateway")
        if gw is not None and gw.signal_id not in causal_parents:
            causal_parents.append(gw.signal_id)

        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind="plan_resolution",
            dimensions={
                "confidence": 0.9 if verb else 0.5,
                "level": clamp01(min(pack_count, 5) / 5.0),
            },
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary=f"orch plan verb={verb or 'unset'} packs={pack_count}",
        )
