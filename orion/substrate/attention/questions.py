from __future__ import annotations

from orion.schemas.attention_frame import OpenLoopV1
from orion.substrate.attention.common import compact

# Provenance key (AttentionSignalV1 -> OpenLoopV1, merged in scoring.build_open_loops)
# carrying the same-turn LLM read's follow-up question for something the user shared.
# Lives in provenance rather than a typed field: OpenLoopV1 is extra="forbid" and is
# persisted/read across services, so a new field would break any not-yet-redeployed
# reader of rows written by an updated producer.
NATURAL_QUESTION_KEY = "natural_question"
_MAX_NATURAL_QUESTION_LEN = 160


def question_for(loop: OpenLoopV1) -> str:
    natural = compact(str(loop.provenance.get(NATURAL_QUESTION_KEY) or ""), _MAX_NATURAL_QUESTION_LEN)
    if natural:
        return natural
    desc = compact(loop.description, 90)
    if loop.target_type == "plan":
        return f"What is the unresolved constraint around {desc}?"
    if loop.target_type == "activity":
        return f"What part of {desc} is still open?"
    if loop.target_type == "anomaly":
        return f"What changed right before {desc} showed up?"
    return f"What is the sharp unknown around {desc}?"
