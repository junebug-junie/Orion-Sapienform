"""Standard payload for finalize_response tool (shared, testable)."""

from __future__ import annotations

import json
from typing import Any


def build_finalize_tool_input(
    *,
    user_text: str,
    trace_snapshot: list,
    output_mode: str | None,
    response_profile: str | None,
) -> dict[str, Any]:
    return {
        "original_request": user_text,
        "request": user_text,
        "text": user_text,
        "trace": json.dumps([dict(s) for s in trace_snapshot], default=str)[:12000],
        "prior_trace": str(trace_snapshot),
        "output_mode": output_mode or "direct_answer",
        "response_profile": response_profile or "direct_answer",
    }
