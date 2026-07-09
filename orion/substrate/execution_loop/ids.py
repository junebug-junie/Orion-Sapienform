from __future__ import annotations


def cortex_exec_trace_id(
    node_name: str,
    correlation_id: str,
    *,
    lane: str | None = None,
) -> str:
    """Build a cortex.exec trace id.

    ``lane`` isolates auxiliary cortex-exec runs (e.g. harness finalize reflect)
    from the primary unified-turn motor trace that shares the same correlation_id.
    """
    base = f"cortex.exec:{node_name}:{correlation_id}"
    if lane:
        return f"{base}:{lane}"
    return base


def parse_execution_trace_id(trace_id: str) -> tuple[str, str] | None:
    if not trace_id.startswith("cortex.exec:"):
        return None
    parts = trace_id.split(":", 2)
    if len(parts) < 3 or not parts[1].strip() or not parts[2].strip():
        return None
    return parts[1].strip().lower(), parts[2]
