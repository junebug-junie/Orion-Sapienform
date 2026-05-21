"""Spark telemetry → spark_introspector OrionSignalV1 (Milestone B4)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from orion.schemas.telemetry.spark_signal import SparkSignalV1
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id

_SPARK_KINDS = frozenset({"spark.signal.v1", "spark.signal"})


def _delta_dim(value: float | None, default: float = 0.5) -> float:
    if value is None:
        return default
    return clamp01(0.5 + 0.5 * float(value))


class SparkAdapter(OrionSignalAdapter):
    organ_id = "spark_introspector"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if channel in _SPARK_KINDS:
            return True
        if "spark.signal" in channel:
            return True
        if "spark:signal" in channel and "biometrics" not in channel:
            return True
        if payload.get("signal_type") and payload.get("intensity") is not None:
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
        spark: SparkSignalV1 | None = None
        try:
            spark = SparkSignalV1.model_validate(payload)
        except Exception:
            spark = None

        notes: list[str] = []
        if spark is not None:
            level = clamp01(spark.intensity)
            valence = _delta_dim(spark.valence_delta)
            arousal = _delta_dim(spark.arousal_delta)
            coherence = _delta_dim(spark.coherence_delta)
            novelty = _delta_dim(spark.novelty_delta)
            signal_kind = "spark_signal"
            confidence = 0.9
            observed = spark.as_of_ts
            src_raw = f"{spark.source_service}:{int(observed.timestamp())}"
        else:
            level = clamp01(float(payload.get("intensity", 0.5)))
            valence = _delta_dim(payload.get("valence_delta"))
            arousal = _delta_dim(payload.get("arousal_delta"))
            coherence = _delta_dim(payload.get("coherence_delta"))
            novelty = _delta_dim(payload.get("novelty_delta"))
            signal_kind = "tissue_state"
            confidence = 0.5
            observed = now
            src_raw = payload.get("source_service") or payload.get("correlation_id")
            notes.append("partial spark payload; schema validation skipped")

        src_id = str(src_raw) if src_raw is not None else None
        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]

        stype = (spark.signal_type if spark else None) or payload.get("signal_type") or "unknown"

        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind=signal_kind,
            dimensions={
                "level": level,
                "valence": valence,
                "arousal": arousal,
                "coherence": coherence,
                "novelty": novelty,
                "confidence": confidence,
            },
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=observed if observed.tzinfo else observed.replace(tzinfo=timezone.utc),
            emitted_at=now,
            summary=f"spark {signal_kind} type={stype} intensity={level:.2f}",
            notes=notes[:5],
        )
