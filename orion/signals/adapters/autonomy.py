"""Autonomy summary/state → autonomy_state OrionSignalV1 (Milestone B4)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from orion.autonomy.summary import summarize_autonomy_state
from orion.autonomy.models import AutonomyStateV1
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id

_DRIVE_KEYS = (
    "coherence",
    "continuity",
    "relational",
    "autonomy",
    "capability",
    "predictive",
)


def _extract_state(payload: dict) -> AutonomyStateV1 | None:
    for key in ("chat_autonomy_state", "autonomy_state", "state"):
        raw = payload.get(key)
        if isinstance(raw, dict):
            try:
                return AutonomyStateV1.model_validate(raw)
            except Exception:
                continue
    return None


def _extract_summary_dict(payload: dict) -> dict[str, Any] | None:
    for key in ("chat_autonomy_summary", "autonomy_summary", "autonomy_state_preview"):
        raw = payload.get(key)
        if isinstance(raw, dict):
            return raw
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        raw = meta.get("autonomy_state_preview") or meta.get("autonomy_summary")
        if isinstance(raw, dict):
            return raw
    return None


def _pressure_dimensions(state: AutonomyStateV1 | None, summary_dict: dict[str, Any] | None) -> tuple[dict[str, float], float, list[str]]:
    """Derive pressure_<drive> dims from whatever real signal is actually available.

    ``AutonomyStateV1``/``AutonomyStateV2`` no longer carry ``dominant_drive``/
    ``active_drives``/``drive_pressures``/``tension_kinds`` (removed 2026-07-30,
    chore/delete-orion-drives Wave 2a -- DriveEngine, the only producer, was
    already deleted in Wave 1, so those fields had nothing real left to hold).
    A full ``state`` object therefore never carries per-drive pressure data
    now; only the raw ``summary_dict`` payload (an untyped dict scraped from
    ``chat_autonomy_summary``/``autonomy_state_preview``, independent of the
    ``AutonomyStateV1`` schema) can still carry a ``dominant_drive``/
    ``top_drives`` hint. Prefer that hint whenever present, even alongside a
    full state; fall back to a neutral, explicitly low-confidence reading
    rather than a fabricated high-confidence all-default one when only a bare
    state is available.
    """
    notes: list[str] = []
    if summary_dict is not None:
        pressures = {k: 0.5 for k in _DRIVE_KEYS}
        dom = (summary_dict.get("dominant_drive") or "").strip().lower()
        if dom:
            pressures[dom] = 0.75
        for d in summary_dict.get("top_drives") or []:
            if isinstance(d, str):
                pressures[d.strip().lower()] = max(pressures.get(d.strip().lower(), 0.0), 0.65)
        raw_state_present = bool(summary_dict.get("raw_state_present")) or state is not None
        confidence = 0.75 if raw_state_present else 0.45
        notes.append("autonomy summary-only payload; drive pressures approximated")
    elif state is not None:
        summarize_autonomy_state(state)  # confirms state validates; no per-drive data to extract from it anymore
        pressures = {k: 0.5 for k in _DRIVE_KEYS}
        confidence = 0.3
        notes.append("autonomy state present but carries no drive_pressures (field removed); dimensions are neutral defaults")
    else:
        return (
            {f"pressure_{k}": 0.5 for k in _DRIVE_KEYS} | {"confidence": 0.2},
            0.2,
            ["partial autonomy payload; confidence degraded"],
        )

    dims: dict[str, float] = {f"pressure_{k}": clamp01(float(pressures.get(k, 0.0))) for k in _DRIVE_KEYS}
    dims["confidence"] = confidence
    return dims, confidence, notes


class AutonomyAdapter(OrionSignalAdapter):
    organ_id = "autonomy"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if _extract_state(payload) is not None or _extract_summary_dict(payload) is not None:
            return True
        if "autonomy" in channel:
            return True
        if "cortex:exec" in channel and (
            payload.get("chat_autonomy_state") or payload.get("chat_autonomy_summary")
        ):
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
        state = _extract_state(payload)
        summary_dict = _extract_summary_dict(payload)
        if state is None and summary_dict is None:
            return None

        dimensions, confidence, notes = _pressure_dimensions(state, summary_dict)
        corr = payload.get("correlation_id") or payload.get("_envelope_correlation_id")
        src_id = str(corr) if corr else None
        if src_id is None:
            src_id = f"autonomy:{int(now.timestamp())}"
            notes.append("synthetic_correlation_id")

        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]

        # AutonomyStateV1/V2 no longer carry dominant_drive (removed 2026-07-30,
        # chore/delete-orion-drives Wave 2a); only the untyped summary_dict hint
        # (independent of that schema) can still carry one.
        dom = (summary_dict or {}).get("dominant_drive") or "unknown"
        signal_kind = "autonomy_state" if confidence >= 0.5 else "tension_state"

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
            summary=f"autonomy {signal_kind} dominant={dom}",
            notes=notes[:5],
        )
