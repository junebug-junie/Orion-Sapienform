from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from pydantic import ValidationError

from orion.schemas.telemetry.spark import SparkStateSnapshotV1, SparkTelemetryPayload


_SNAPSHOT_REQUIRED_FIELDS = {
    "source_service",
    "producer_boot_id",
    "seq",
    "snapshot_ts",
}


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except ValueError:
            pass
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def _coerce_telemetry_timestamp(value: Any) -> Optional[datetime | str]:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except ValueError:
            return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return None


def _as_mapping(obj: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return None


def normalize_spark_state_snapshot(
    obj: Any,
    *,
    now: datetime | None = None,
) -> SparkStateSnapshotV1 | None:
    """Normalize a Spark state snapshot payload.

    Returns None for incomplete payloads instead of inventing semantics.
    """
    if isinstance(obj, SparkStateSnapshotV1):
        return obj

    data = _as_mapping(obj)
    if data is None:
        return None

    allowed_fields = set(SparkStateSnapshotV1.model_fields.keys())
    filtered: dict[str, Any] = {k: data[k] for k in data.keys() if k in allowed_fields}

    if not _SNAPSHOT_REQUIRED_FIELDS.issubset(filtered.keys()):
        return None

    snapshot_ts = _coerce_datetime(filtered.get("snapshot_ts"))
    if snapshot_ts is None:
        return None
    filtered["snapshot_ts"] = snapshot_ts

    if "phi" in filtered and not isinstance(filtered.get("phi"), Mapping):
        filtered.pop("phi")

    if "metadata" in filtered and not isinstance(filtered.get("metadata"), Mapping):
        filtered.pop("metadata")

    try:
        return SparkStateSnapshotV1.model_validate(filtered)
    except ValidationError:
        if now is not None:
            return None
        return None


def normalize_spark_telemetry(
    obj: Any,
    *,
    now: datetime | None = None,
) -> SparkTelemetryPayload | None:
    if isinstance(obj, SparkTelemetryPayload):
        return obj

    data = _as_mapping(obj)
    if data is None:
        return None

    normalized: dict[str, Any] = dict(data)

    if "timestamp" in normalized:
        normalized_ts = _coerce_telemetry_timestamp(normalized.get("timestamp"))
        if normalized_ts is None:
            return None
        normalized["timestamp"] = normalized_ts

    state_snapshot = normalized.get("state_snapshot")
    if state_snapshot is not None:
        normalized_snapshot = normalize_spark_state_snapshot(state_snapshot, now=now)
        normalized["state_snapshot"] = normalized_snapshot

    metadata = normalized.get("metadata")
    if isinstance(metadata, Mapping) and "spark_state_snapshot" in metadata:
        if normalized.get("state_snapshot") is None:
            nested = normalize_spark_state_snapshot(metadata.get("spark_state_snapshot"), now=now)
            normalized["state_snapshot"] = nested

    try:
        return SparkTelemetryPayload.model_validate(normalized)
    except ValidationError:
        return None


def normalize_spark(
    kind: str | None,
    payload: Any,
    *,
    now: datetime | None = None,
) -> SparkTelemetryPayload | SparkStateSnapshotV1 | None:
    if isinstance(payload, (SparkTelemetryPayload, SparkStateSnapshotV1)):
        return payload

    if kind:
        lowered = kind.lower()
        if "spark.state.snapshot" in lowered:
            return normalize_spark_state_snapshot(payload, now=now)
        if "spark.telemetry" in lowered or "spark.introspection" in lowered:
            return normalize_spark_telemetry(payload, now=now)

    telemetry = normalize_spark_telemetry(payload, now=now)
    if telemetry is not None:
        return telemetry
    return normalize_spark_state_snapshot(payload, now=now)
