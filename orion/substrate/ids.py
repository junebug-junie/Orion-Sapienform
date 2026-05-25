"""Deterministic substrate receipt and state-delta identifiers."""

from __future__ import annotations

import hashlib


def _sorted_join(values: list[str]) -> str:
    return ",".join(sorted(v.strip() for v in values if v and str(v).strip()))


def stable_hash_id(prefix: str, parts: list[str]) -> str:
    """Return ``{prefix}_{sha256(preimage)[:24]}`` from ordered semantic parts."""
    normalized = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    preimage = "|".join(normalized)
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def stable_delta_id(
    *,
    reducer_id: str,
    target_projection: str,
    target_kind: str,
    target_id: str,
    operation: str,
    caused_by_event_ids: list[str],
) -> str:
    return stable_hash_id(
        "delta",
        [
            reducer_id,
            target_projection,
            target_kind,
            target_id.strip().lower(),
            operation,
            _sorted_join(caused_by_event_ids),
        ],
    )


def stable_receipt_id(
    *,
    reducer_id: str,
    accepted_event_ids: list[str],
    rejected_event_ids: list[str],
    merged_event_ids: list[str],
    noop_event_ids: list[str],
    emission_id: str | None = None,
) -> str:
    parts = [
        reducer_id,
        _sorted_join(accepted_event_ids),
        _sorted_join(rejected_event_ids),
        _sorted_join(merged_event_ids),
        _sorted_join(noop_event_ids),
    ]
    if emission_id:
        parts.append(emission_id.strip())
    return stable_hash_id("rcpt", parts)
