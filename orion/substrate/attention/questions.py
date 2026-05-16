from __future__ import annotations

from orion.schemas.attention_frame import OpenLoopV1
from orion.substrate.attention.common import compact


def question_for(loop: OpenLoopV1) -> str:
    desc = compact(loop.description, 90)
    if loop.target_type == "plan":
        return f"What is the unresolved constraint around {desc}?"
    if loop.target_type == "activity":
        return f"What part of {desc} is still open?"
    if loop.target_type == "anomaly":
        return f"What changed right before {desc} showed up?"
    return f"What is the sharp unknown around {desc}?"
