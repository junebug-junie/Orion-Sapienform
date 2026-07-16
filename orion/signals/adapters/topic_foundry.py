"""Topic Foundry drift/run telemetry -> topic_foundry OrionSignalV1 (Phase 5).

Real signal source: ``services/orion-topic-foundry/app/services/drift.py``
computes a Jensen-Shannon divergence (``js_divergence``, bounded [0, 1]) plus
outlier/top-topic-share deltas between the baseline and current topic
distributions and publishes them as ``TopicFoundryDriftAlertV1`` on
``orion:topic:foundry:drift:alert.v1`` (envelope kind
``topic.foundry.drift.alert.v1``). That divergence is the strongest, most
direct measure of "how much has the topic landscape moved" that topic-foundry
emits, so it drives ``dimensions["level"]`` here. Run/enrich-completion
payloads (no drift fields) fall back to a success-ratio or status-based
reading. Anything unrecognized degrades to a neutral, low-confidence,
explicitly-noted reading -- this adapter never raises.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from orion.schemas.topic_foundry import TopicFoundryDriftAlertV1
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.signal_ids import make_signal_id
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _dimensions_from_payload(payload: dict) -> Tuple[Dict[str, float], str, List[str]]:
    """Return (dimensions, signal_kind, notes) derived from a real payload.

    Preference order:
    1. ``TopicFoundryDriftAlertV1``-shaped payload -> js_divergence drives
       ``level``; corroborating outlier/top-share deltas raise ``confidence``.
    2. Raw drift fields present but schema validation failed (partial
       payload) -> same derivation, lower confidence, note the fallback.
    3. Enrich-complete style payload (``enriched_count``/``failed_count``)
       -> success ratio drives ``level``.
    4. Run-complete style payload (``status`` string) -> binary level.
    5. Nothing recognized -> neutral 0.5/low-confidence reading, noted.
    """
    notes: List[str] = []

    drift: Optional[TopicFoundryDriftAlertV1] = None
    try:
        drift = TopicFoundryDriftAlertV1.model_validate(payload)
    except Exception:
        drift = None

    js_divergence = drift.js_divergence if drift is not None else _as_float(payload.get("js_divergence"))
    if js_divergence is not None:
        level = clamp01(js_divergence)
        outlier_delta = drift.outlier_pct_delta if drift is not None else _as_float(payload.get("outlier_pct_delta"))
        top_share_delta = (
            drift.top_topic_share_delta if drift is not None else _as_float(payload.get("top_topic_share_delta"))
        )
        corroborating = sum(1 for v in (outlier_delta, top_share_delta) if v is not None and abs(v) > 0.0)
        confidence = clamp01((0.85 if drift is not None else 0.6) + 0.05 * corroborating)
        if drift is None:
            notes.append("partial drift-alert payload; schema validation skipped")
        return {"level": level, "confidence": confidence}, "topic_drift", notes

    enriched = _as_float(payload.get("enriched_count"))
    failed = _as_float(payload.get("failed_count"))
    if enriched is not None and failed is not None and (enriched + failed) > 0:
        level = clamp01(enriched / (enriched + failed))
        return {"level": level, "confidence": 0.6}, "topic_state", notes

    status = payload.get("status")
    if isinstance(status, str) and status:
        level = 1.0 if status.lower() in ("complete", "completed", "success", "succeeded") else 0.0
        return {"level": level, "confidence": 0.5}, "topic_state", notes

    notes.append("no recognized topic-foundry drift/run/enrich fields; neutral reading")
    return {"level": 0.5, "confidence": 0.2}, "topic_state", notes


class TopicFoundryAdapter(OrionSignalAdapter):
    organ_id = "topic_foundry"

    def can_handle(self, channel: str, payload: dict) -> bool:
        # Real envelope kinds use dots (e.g. "topic.foundry.drift.alert.v1");
        # keep the underscore form too for callers/tests that pass the raw
        # organ id as the channel.
        return "topic_foundry" in channel or "topic.foundry" in channel

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
        src_raw = (
            payload.get("drift_id")
            or payload.get("run_id")
            or payload.get("correlation_id")
            or payload.get("id")
        )
        src_id = str(src_raw) if src_raw is not None else None
        now = datetime.now(timezone.utc)
        sig_id = make_signal_id(self.organ_id, src_id)
        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]
        dimensions, signal_kind, notes = _dimensions_from_payload(payload)
        return OrionSignalV1(
            signal_id=sig_id,
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind=signal_kind,
            dimensions=dimensions,
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            notes=notes[:5],
        )
