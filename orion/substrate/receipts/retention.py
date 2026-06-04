from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from orion.schemas.reduction_receipt import ReductionReceiptV1


@dataclass(frozen=True)
class ReceiptRetentionSettings:
    success_minutes: int = 30
    error_hours: int = 6
    full_payload_success: bool = False
    full_payload_sample_rate: float = 0.0

    @classmethod
    def from_env(cls) -> ReceiptRetentionSettings:
        minutes_env = os.getenv("ORION_RECEIPT_RETENTION_SUCCESS_MINUTES")
        hours_env = os.getenv("ORION_RECEIPT_RETENTION_SUCCESS_HOURS")
        if minutes_env is not None:
            success_minutes = int(minutes_env)
        elif hours_env is not None:
            success_minutes = max(1, int(float(hours_env) * 60))
        else:
            success_minutes = 30

        error_hours_env = os.getenv("ORION_RECEIPT_RETENTION_ERROR_HOURS")
        error_days_env = os.getenv("ORION_RECEIPT_RETENTION_ERROR_DAYS")
        if error_hours_env is not None:
            error_hours = int(error_hours_env)
        elif error_days_env is not None:
            error_hours = max(1, int(float(error_days_env) * 24))
        else:
            error_hours = 6

        return cls(
            success_minutes=success_minutes,
            error_hours=error_hours,
            full_payload_success=os.getenv("ORION_RECEIPT_FULL_PAYLOAD_SUCCESS", "false").lower()
            in ("1", "true", "yes"),
            full_payload_sample_rate=float(
                os.getenv("ORION_RECEIPT_FULL_PAYLOAD_SAMPLE_RATE", "0")
            ),
        )


@dataclass(frozen=True)
class ReceiptClassification:
    receipt_kind: str
    receipt_status: str
    is_full_payload: bool


def _utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def classify_receipt(
    receipt: ReductionReceiptV1,
    *,
    settings: ReceiptRetentionSettings,
    rng_value: float,
) -> ReceiptClassification:
    if receipt.warnings:
        return ReceiptClassification("error", "error", True)
    if receipt.rejected_event_ids and not receipt.state_deltas:
        return ReceiptClassification("error", "error", True)
    if settings.full_payload_success:
        return ReceiptClassification("success", "ok", True)
    if settings.full_payload_sample_rate > 0 and rng_value < settings.full_payload_sample_rate:
        return ReceiptClassification("debug_sample", "ok", True)
    return ReceiptClassification("success", "ok", False)


def compact_receipt_json(receipt: ReductionReceiptV1, *, is_full_payload: bool) -> dict[str, Any]:
    if is_full_payload:
        return receipt.model_dump(mode="json")
    payload: dict[str, Any] = {
        "schema_version": receipt.schema_version,
        "receipt_id": receipt.receipt_id,
        "organ_id": receipt.organ_id,
        "emission_id": receipt.emission_id,
        "state_deltas": [d.model_dump(mode="json") for d in receipt.state_deltas],
        "warnings": list(receipt.warnings),
        "created_at": receipt.created_at.isoformat(),
    }
    if receipt.noop_event_ids:
        payload["noop_event_ids"] = list(receipt.noop_event_ids)
    return payload


def payload_fingerprint(receipt: ReductionReceiptV1) -> str:
    raw = json.dumps(receipt.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def payload_byte_length(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def primary_delta_id(receipt: ReductionReceiptV1) -> str | None:
    if not receipt.state_deltas:
        return None
    return receipt.state_deltas[0].delta_id


def primary_event_id(receipt: ReductionReceiptV1) -> str | None:
    if receipt.accepted_event_ids:
        return receipt.accepted_event_ids[0]
    if receipt.noop_event_ids:
        return receipt.noop_event_ids[0]
    if receipt.rejected_event_ids:
        return receipt.rejected_event_ids[0]
    for delta in receipt.state_deltas:
        if delta.caused_by_event_ids:
            return delta.caused_by_event_ids[0]
    return None


def primary_reducer_name(receipt: ReductionReceiptV1) -> str | None:
    if receipt.state_deltas:
        return receipt.state_deltas[0].reducer_id
    return None


def retention_expires_at(
    classification: ReceiptClassification,
    *,
    settings: ReceiptRetentionSettings,
    now: datetime,
) -> datetime | None:
    clock = _utc(now)
    if classification.receipt_kind == "error":
        return clock + timedelta(hours=settings.error_hours)
    if classification.receipt_kind == "debug_sample":
        return clock + timedelta(minutes=settings.success_minutes)
    if classification.receipt_kind == "success":
        return clock + timedelta(minutes=settings.success_minutes)
    return None
