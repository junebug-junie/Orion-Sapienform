"""World Pulse bus events → OrionSignalV1 (run result, digest, situation, capsule, graph)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrganClass, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id

_COVERAGE_LEVEL: dict[str, float] = {
    "complete": 0.9,
    "partial": 0.65,
    "sparse": 0.4,
    "empty": 0.1,
}


def _coverage_level(payload: dict) -> float:
    digest = payload.get("digest")
    if isinstance(digest, dict):
        status = str(digest.get("coverage_status") or "empty")
        return _COVERAGE_LEVEL.get(status, 0.5)
    if payload.get("coverage_status") is not None:
        return _COVERAGE_LEVEL.get(str(payload.get("coverage_status")), 0.5)
    salient = payload.get("salient_topics")
    if isinstance(salient, list) and salient:
        return 0.55
    return 0.5


def _run_source_id(payload: dict) -> Optional[str]:
    run = payload.get("run")
    if isinstance(run, dict):
        raw = run.get("run_id")
        if raw is not None:
            return str(raw)
    for key in ("run_id", "capsule_id", "graph_delta_id", "brief_id", "correlation_id", "id"):
        if payload.get(key) is not None:
            return str(payload[key])
    return None


def _signal_kind_for_channel(channel: str, entry: OrionOrganRegistryEntry) -> str:
    if "run:result" in channel:
        return "time_context"
    if "situation:brief" in channel:
        return "situation_state"
    if "graph:upsert" in channel:
        return "time_context"
    if "world_context" in channel or "daily_capsule" in channel:
        return "environmental_context"
    if "digest" in channel:
        return "environmental_context"
    kinds = entry.signal_kinds or []
    return kinds[0] if kinds else "time_context"


class WorldPulseAdapter(OrionSignalAdapter):
    organ_id = "world_pulse"

    def can_handle(self, channel: str, payload: dict) -> bool:
        return "world_pulse" in channel or "world_context" in channel

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

        if not isinstance(payload, dict) or not payload:
            now = datetime.now(timezone.utc)
            sig_id = make_signal_id(self.organ_id, None)
            return OrionSignalV1(
                signal_id=sig_id,
                organ_id=self.organ_id,
                organ_class=OrganClass.hybrid,
                signal_kind=_signal_kind_for_channel(channel, entry),
                dimensions={"level": 0.5, "confidence": 0.1, "valence": 0.5},
                causal_parents=[],
                source_event_id=None,
                observed_at=now,
                emitted_at=now,
                notes=["malformed or empty payload; confidence degraded"],
            )

        src_id = _run_source_id(payload)
        now = datetime.now(timezone.utc)
        sig_id = make_signal_id(self.organ_id, src_id)

        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]

        level = _coverage_level(payload)
        confidence = level
        rollups = []
        digest = payload.get("digest")
        if isinstance(digest, dict):
            rollups = digest.get("section_rollups") or []
        elif isinstance(payload.get("section_rollups"), list):
            rollups = payload.get("section_rollups") or []
        if rollups and isinstance(rollups[0], dict):
            try:
                confidence = clamp01(float(rollups[0].get("confidence", confidence)))
            except (TypeError, ValueError):
                pass

        run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
        run_status = str(run.get("status") or payload.get("status") or "unknown")
        notes: list[str] = []
        if run.get("dry_run") is True or payload.get("dry_run") is True:
            notes.append("dry_run=true")
        if run_status not in {"completed", "partial", "unknown"}:
            notes.append(f"run_status={run_status}")
        if src_id is None:
            notes.append("missing source_event_id")

        return OrionSignalV1(
            signal_id=sig_id,
            organ_id=self.organ_id,
            organ_class=entry.organ_class or OrganClass.hybrid,
            signal_kind=_signal_kind_for_channel(channel, entry),
            dimensions={
                "level": clamp01(level),
                "confidence": clamp01(confidence),
                "valence": 0.5,
            },
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            notes=notes,
        )
