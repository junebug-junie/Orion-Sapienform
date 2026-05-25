from __future__ import annotations


def parse_execution_trace_id(trace_id: str) -> tuple[str, str] | None:
    if not trace_id.startswith("cortex.exec:"):
        return None
    parts = trace_id.split(":", 2)
    if len(parts) < 3 or not parts[1].strip() or not parts[2].strip():
        return None
    return parts[1].strip().lower(), parts[2]
