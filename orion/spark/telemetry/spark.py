from __future__ import annotations

import warnings

from orion.schemas.telemetry.spark import SparkStateSnapshotV1, SparkTelemetryPayload

warnings.warn(
    "orion.spark.telemetry.spark is deprecated; use orion.schemas.telemetry.spark instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SparkStateSnapshotV1", "SparkTelemetryPayload"]
