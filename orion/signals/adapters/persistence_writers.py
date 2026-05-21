"""Persistence writer stored events → persist signals (Milestone B5)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id


def _latency_level(payload: dict) -> float:
    for key in ("latency_ms", "duration_ms", "write_latency_ms"):
        if payload.get(key) is not None:
            return clamp01(min(float(payload[key]), 30_000) / 30_000.0)
    stats = payload.get("stats")
    if isinstance(stats, dict) and stats.get("latency_ms") is not None:
        return clamp01(min(float(stats["latency_ms"]), 30_000) / 30_000.0)
    return 0.5


class SqlWriterAdapter(OrionSignalAdapter):
    organ_id = "sql_writer"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if channel in ("collapse.mirror.stored.v1", "journal.entry.created.v1"):
            return True
        if "collapse:stored" in channel or "journal:created" in channel:
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
        src_id = str(payload.get("entry_id") or payload.get("id") or payload.get("correlation_id") or "")
        if not src_id:
            src_id = f"sql:{channel}:{int(now.timestamp())}"
        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind="persist",
            dimensions={"level": 1.0, "confidence": 0.9, "latency_level": _latency_level(payload)},
            causal_parents=[],
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary=f"sql persist {channel}",
        )


class RdfWriterAdapter(OrionSignalAdapter):
    organ_id = "rdf_writer"

    def can_handle(self, channel: str, payload: dict) -> bool:
        return "rdf" in channel and ("stored" in channel or "written" in channel or "persist" in channel)

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
        src_id = str(payload.get("uri") or payload.get("id") or payload.get("correlation_id") or f"rdf:{int(now.timestamp())}")
        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind="persist",
            dimensions={"level": 1.0, "confidence": 0.75, "latency_level": _latency_level(payload)},
            causal_parents=[],
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary="rdf persist presence",
        )


class VectorWriterAdapter(OrionSignalAdapter):
    organ_id = "vector_writer"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if "vector" in channel and ("stored" in channel or "indexed" in channel or "upsert" in channel):
            return True
        return channel in ("evidence.index.upsert.v1",)

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
        src_id = str(payload.get("doc_id") or payload.get("id") or payload.get("correlation_id") or f"vector:{int(now.timestamp())}")
        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind="persist",
            dimensions={"level": 1.0, "confidence": 0.75, "latency_level": _latency_level(payload)},
            causal_parents=[],
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary="vector persist presence",
        )
