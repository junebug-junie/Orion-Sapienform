"""
Normalization primitives for signal adapters.

EwmaBand, InductionTracker, and clamp01 live here per the organ-signal-gateway
design; ``orion.telemetry.biometrics_pipeline`` re-exports them for backward
compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

__all__ = [
    "clamp01",
    "EwmaBand",
    "InductionMetricState",
    "InductionTracker",
    "clamp11",
    "NormalizationContext",
]


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def clamp11(v: float) -> float:
    if v < -1.0:
        return -1.0
    if v > 1.0:
        return 1.0
    return v


@dataclass
class EwmaBand:
    alpha: float = 0.1
    mean: Optional[float] = None
    dev: Optional[float] = None

    def update(self, value: float) -> None:
        if self.mean is None:
            self.mean = value
            self.dev = 0.0
            return
        delta = value - self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * value
        dev = abs(delta)
        self.dev = (1 - self.alpha) * (self.dev or 0.0) + self.alpha * dev

    def normalize(self, value: float) -> float:
        if self.mean is None or self.dev is None:
            return 0.0
        low = self.mean - 2 * self.dev
        high = self.mean + 2 * self.dev
        if high <= low:
            return 0.0
        return clamp01((value - low) / (high - low))


@dataclass
class InductionMetricState:
    level: float = 0.0
    trend: float = 0.5
    volatility: float = 0.0
    spike_rate: float = 0.0
    initialized: bool = False


class InductionTracker:
    def __init__(
        self,
        *,
        level_alpha: float = 0.3,
        trend_alpha: float = 0.2,
        volatility_alpha: float = 0.2,
        spike_alpha: float = 0.1,
        spike_threshold: float = 0.15,
    ) -> None:
        self.level_alpha = level_alpha
        self.trend_alpha = trend_alpha
        self.volatility_alpha = volatility_alpha
        self.spike_alpha = spike_alpha
        self.spike_threshold = spike_threshold
        self._metrics: Dict[str, InductionMetricState] = {}

    def update(self, name: str, value: float) -> InductionMetricState:
        value = clamp01(value)
        metric = self._metrics.get(name)
        if metric is None:
            metric = InductionMetricState(level=value, trend=0.5, volatility=0.0, spike_rate=0.0, initialized=True)
            self._metrics[name] = metric
            return metric

        prev_level = metric.level
        metric.level = (1 - self.level_alpha) * metric.level + self.level_alpha * value
        delta = value - prev_level

        trend_signed = (1 - self.trend_alpha) * (metric.trend - 0.5) + self.trend_alpha * delta
        metric.trend = clamp01(trend_signed + 0.5)

        metric.volatility = (1 - self.volatility_alpha) * metric.volatility + self.volatility_alpha * abs(delta)
        spike = 1.0 if abs(delta) >= self.spike_threshold else 0.0
        metric.spike_rate = (1 - self.spike_alpha) * metric.spike_rate + self.spike_alpha * spike
        return metric

    def snapshot(self) -> Dict[str, InductionMetricState]:
        return dict(self._metrics)


class NormalizationContext:
    """Per-organ EWMA and tracker state, owned by the gateway and passed to adapters."""

    def __init__(self) -> None:
        self._bands: Dict[str, Dict[str, EwmaBand]] = {}
        self._trackers: Dict[str, Dict[str, InductionTracker]] = {}

    def get_band(self, organ_id: str, metric_key: str) -> EwmaBand:
        organ_bands = self._bands.setdefault(organ_id, {})
        if metric_key not in organ_bands:
            organ_bands[metric_key] = EwmaBand()
        return organ_bands[metric_key]

    def get_tracker(self, organ_id: str, metric_key: str) -> InductionTracker:
        """Return the InductionTracker for (organ_id, metric_key)."""
        organ_trackers = self._trackers.setdefault(organ_id, {})
        if metric_key not in organ_trackers:
            organ_trackers[metric_key] = InductionTracker()
        return organ_trackers[metric_key]
