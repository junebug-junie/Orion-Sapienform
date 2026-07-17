"""Metadata size/key caps to prevent property-graph cathedral bloat."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

METADATA_MAX_KEYS = 16
METADATA_MAX_BYTES = 4096


class PropertyCathedralError(ValueError):
    """Metadata violates property cathedral caps and fail_closed is enabled."""


def _metadata_byte_size(metadata: dict[str, Any]) -> int:
    payload = json.dumps(metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return len(payload.encode("utf-8"))


def _log_rejected(rejected: list[str], *, reason: str) -> None:
    if not rejected:
        return
    logger.warning(
        "property_cathedral_rejected reason=%s keys=%s",
        reason,
        ",".join(rejected),
    )


def sanitize_metadata(
    metadata: Any,
    *,
    fail_closed: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Return sanitized metadata and rejected key names.

    Key retention uses **insertion order** (Python dict order): the first
    ``METADATA_MAX_KEYS`` keys present in the input are kept; later keys are
    dropped. When byte size still exceeds ``METADATA_MAX_BYTES`` after key
    trimming, keys are dropped from the end (most recently inserted among
    survivors) until the serialized JSON fits or no keys remain.

    Non-dict input is treated as empty metadata when ``fail_closed=False``,
    or raises ``PropertyCathedralError`` when ``fail_closed=True``.
    """
    rejected: list[str] = []
    reject_reason: str | None = None

    if not isinstance(metadata, dict):
        if fail_closed:
            raise PropertyCathedralError("metadata must be a dict")
        rejected = ["<non-dict>"]
        _log_rejected(rejected, reason="type")
        return {}, rejected

    if not metadata:
        return {}, []

    items = list(metadata.items())
    if len(items) > METADATA_MAX_KEYS:
        for key, _ in items[METADATA_MAX_KEYS:]:
            rejected.append(str(key))
        reject_reason = "key_count"
        if fail_closed:
            _log_rejected(rejected, reason=reject_reason)
            raise PropertyCathedralError(
                f"metadata exceeds max keys ({METADATA_MAX_KEYS}): {rejected}"
            )
        items = items[:METADATA_MAX_KEYS]

    sanitized = dict(items)

    if _metadata_byte_size(sanitized) > METADATA_MAX_BYTES:
        if fail_closed:
            reject_reason = "byte_size"
            _log_rejected(list(sanitized.keys()), reason=reject_reason)
            raise PropertyCathedralError(
                f"metadata exceeds max bytes ({METADATA_MAX_BYTES})"
            )

        reject_reason = "byte_size"
        while sanitized and _metadata_byte_size(sanitized) > METADATA_MAX_BYTES:
            drop_key = next(reversed(sanitized))
            del sanitized[drop_key]
            if str(drop_key) not in rejected:
                rejected.append(str(drop_key))

    if rejected and reject_reason:
        _log_rejected(rejected, reason=reject_reason)

    return sanitized, rejected
