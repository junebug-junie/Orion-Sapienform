"""Single place to normalize OpenTelemetry trace ids for Hub (spec §2.1)."""
from __future__ import annotations


def normalize_otel_trace_id(raw: str) -> str:
    """Return lowercase hex without 0x; does not validate length (callers may)."""
    t = (raw or "").strip().lower()
    if t.startswith("0x"):
        t = t[2:]
    return t


def is_valid_otel_trace_id(normalized: str) -> bool:
    return len(normalized) == 32 and all(c in "0123456789abcdef" for c in normalized)
