"""Equilibrium snapshot → mesh_health OrionSignalV1 (Milestone B1)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.schemas.telemetry.system_health import EquilibriumSnapshotV1
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id

_SNAPSHOT_KINDS = frozenset({"equilibrium.snapshot.v1"})


def _parse_observed_at(payload: dict, fallback: datetime) -> datetime:
    raw = payload.get("generated_at") or payload.get("timestamp")
    if raw is None:
        return fallback
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return fallback


def _mesh_level(payload: dict, snap: EquilibriumSnapshotV1 | None) -> tuple[float, str]:
    """Return (level, signal_kind) from snapshot fields."""
    if snap is not None:
        if snap.distress_score >= 0.35:
            return clamp01(snap.distress_score), "service_distress"
        return clamp01(snap.zen_score), "mesh_health"
    if payload.get("mesh_health") is not None:
        return clamp01(float(payload["mesh_health"])), "mesh_health"
    if payload.get("distress_score") is not None:
        return clamp01(float(payload["distress_score"])), "service_distress"
    if payload.get("zen_score") is not None:
        return clamp01(float(payload["zen_score"])), "mesh_health"
    return 0.5, "mesh_health"


class EquilibriumAdapter(OrionSignalAdapter):
    organ_id = "equilibrium"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if channel in _SNAPSHOT_KINDS:
            return True
        if "equilibrium.snapshot" in channel:
            return True
        if "equilibrium" in channel and ("snapshot" in channel or "zen_score" in payload or "distress_score" in payload):
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
        snap: EquilibriumSnapshotV1 | None = None
        try:
            snap = EquilibriumSnapshotV1.model_validate(payload)
        except Exception:
            snap = None

        src_raw = (
            payload.get("correlation_id")
            or (snap.correlation_id if snap else None)
            or payload.get("producer_boot_id")
            or payload.get("id")
        )
        src_id = str(src_raw) if src_raw is not None else None
        observed = _parse_observed_at(payload, now)
        level, signal_kind = _mesh_level(payload, snap)

        tracker = norm_ctx.get_tracker(self.organ_id, signal_kind)
        state = tracker.update("level", level)

        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]

        confidence = 0.85 if snap is not None else 0.55
        notes: list[str] = []
        if snap is None:
            notes.append("partial equilibrium payload; schema validation skipped")

        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind=signal_kind,
            dimensions={
                "level": state.level,
                "trend": clamp01(0.5 + 0.5 * state.trend),
                "confidence": confidence,
            },
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=observed,
            emitted_at=now,
            summary=f"equilibrium {signal_kind} level={state.level:.2f}",
            notes=notes[:5],
        )
