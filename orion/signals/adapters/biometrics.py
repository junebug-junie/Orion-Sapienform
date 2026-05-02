"""
Reference adapter: biometrics bus events → OrionSignalV1.
Handles kind ``biometrics.induction.v1`` from the biometrics pipeline.
OTEL trace ids are assigned by the gateway (spec §5); this adapter leaves them unset.
"""
import hashlib
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.models import OrionOrganRegistryEntry, OrganClass, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01, clamp11


class BiometricsAdapter(OrionSignalAdapter):
    organ_id = "biometrics"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if not channel and not payload:
            return False
        if "biometrics.induction" in channel or channel == "biometrics.induction.v1":
            return True
        if "biometrics" in channel and "induction" in channel:
            return True
        return bool(payload.get("metrics"))

    def adapt(
        self,
        channel: str,
        payload: dict,
        registry: Dict[str, OrionOrganRegistryEntry],
        prior_signals: Dict[str, OrionSignalV1],
        norm_ctx: NormalizationContext,
    ) -> Optional[OrionSignalV1]:
        now = datetime.now(timezone.utc)

        try:
            raw_metrics: dict = payload.get("metrics") or {}
        except Exception:
            raw_metrics = {}

        src_id: Optional[str] = (
            str(payload.get("correlation_id"))
            if payload.get("correlation_id") is not None
            else None
        )
        if src_id is None and payload.get("node") is not None:
            ts = payload.get("timestamp")
            src_id = f"{payload.get('node')}:{ts}" if ts is not None else str(payload.get("node"))

        if src_id:
            sig_id = hashlib.sha256(f"{self.organ_id}:{src_id}".encode()).hexdigest()[:16]
        else:
            sig_id = str(uuid4())

        if not raw_metrics:
            return OrionSignalV1(
                signal_id=sig_id,
                organ_id="biometrics",
                organ_class=OrganClass.exogenous,
                signal_kind="biometrics_state",
                dimensions={"level": 0.5, "confidence": 0.1},
                causal_parents=[],
                source_event_id=src_id,
                observed_at=now,
                emitted_at=now,
                notes=["malformed or empty metrics payload; confidence degraded"],
            )

        dimensions: Dict[str, float] = {}
        for metric_name, metric in raw_metrics.items():
            if isinstance(metric, dict):
                level = float(metric.get("level", 0.5))
            else:
                level = float(getattr(metric, "level", 0.5))

            level = clamp01(level)
            tracker = norm_ctx.get_tracker(self.organ_id, str(metric_name))
            state = tracker.update("v", level)

            dimensions[f"{metric_name}_level"] = state.level
            dimensions[f"{metric_name}_trend"] = clamp11(2.0 * (state.trend - 0.5))
            dimensions[f"{metric_name}_volatility"] = clamp01(state.volatility)

        dimensions["confidence"] = 0.9

        return OrionSignalV1(
            signal_id=sig_id,
            organ_id="biometrics",
            organ_class=OrganClass.exogenous,
            signal_kind="biometrics_state",
            dimensions=dimensions,
            causal_parents=[],
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
        )
