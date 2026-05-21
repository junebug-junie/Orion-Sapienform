"""Recall exec result → recall_result OrionSignalV1 (Milestone B2)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.core.contracts.recall import RecallReplyV1
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id

_REPLY_KINDS = frozenset({"recall.reply.v1", "recall.reply"})


def _recall_dimensions(payload: dict, reply: RecallReplyV1 | None) -> tuple[dict[str, float], str, list[str]]:
    notes: list[str] = []
    if reply is not None:
        bundle = reply.bundle
        items = bundle.items or []
        n = len(items)
        scores = [float(it.score) for it in items if it.score is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        latency_ms = int(bundle.stats.latency_ms or 0)
        level = clamp01(avg_score if n else 0.0)
        confidence = 0.9 if n else 0.35
        return (
            {
                "level": level,
                "trend": clamp01(min(n, 10) / 10.0),
                "confidence": confidence,
            },
            "recall_result",
            notes,
        )

    items = payload.get("items") or (payload.get("bundle") or {}).get("items") or []
    if isinstance(items, list) and items:
        scores = [float(x.get("score", 0.0)) for x in items if isinstance(x, dict)]
        level = clamp01(sum(scores) / len(scores) if scores else 0.0)
        return (
            {"level": level, "trend": clamp01(min(len(items), 10) / 10.0), "confidence": 0.5},
            "recall_result",
            ["partial recall payload; bundle schema validation skipped"],
        )

    return (
        {"level": 0.2, "trend": 0.0, "confidence": 0.25},
        "recall_gap",
        ["empty or partial recall reply; confidence degraded"],
    )


class RecallAdapter(OrionSignalAdapter):
    organ_id = "recall"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if channel in _REPLY_KINDS:
            return True
        if "recall.reply" in channel:
            return True
        if "RecallService" in channel and "result" in channel:
            return True
        if "bundle" in payload and isinstance(payload.get("bundle"), dict):
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
        reply: RecallReplyV1 | None = None
        try:
            reply = RecallReplyV1.model_validate(payload)
        except Exception:
            reply = None

        src_raw = payload.get("correlation_id") or (reply.correlation_id if reply else None) or payload.get("id")
        src_id = str(src_raw) if src_raw is not None else None
        dimensions, signal_kind, notes = _recall_dimensions(payload, reply)

        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]

        item_count = 0
        if reply is not None:
            item_count = len(reply.bundle.items or [])
        elif isinstance(payload.get("bundle"), dict):
            items = payload["bundle"].get("items") or []
            item_count = len(items) if isinstance(items, list) else 0

        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind=signal_kind,
            dimensions=dimensions,
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary=f"recall {signal_kind} hits={item_count} level={dimensions['level']:.2f}",
            notes=notes[:5],
        )
