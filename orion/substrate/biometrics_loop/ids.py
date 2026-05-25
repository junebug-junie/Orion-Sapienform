from __future__ import annotations


def parse_biometrics_trace_id(trace_id: str) -> str | None:
    if not trace_id.startswith("biometrics.node:"):
        return None
    parts = trace_id.split(":", 2)
    if len(parts) < 3:
        return None
    return parts[1].strip().lower()


def parse_pressure_trace_id(trace_id: str) -> str | None:
    if not trace_id.startswith("substrate.pressure:"):
        return None
    parts = trace_id.split(":", 2)
    if len(parts) < 3:
        return None
    return parts[1].strip().lower()
