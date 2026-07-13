from __future__ import annotations


def route_trace_id(node_name: str, correlation_id: str) -> str:
    """Build an orch.route trace id.

    Mirrors ``cortex_exec_trace_id`` in ``execution_loop/ids.py``: same
    two-part structure after the prefix (node_name, correlation_id).
    """
    return f"orch.route:{node_name}:{correlation_id}"


def parse_route_trace_id(trace_id: str) -> tuple[str, str] | None:
    if not trace_id.startswith("orch.route:"):
        return None
    parts = trace_id.split(":", 2)
    if len(parts) < 3 or not parts[1].strip() or not parts[2].strip():
        return None
    return parts[1].strip().lower(), parts[2]
