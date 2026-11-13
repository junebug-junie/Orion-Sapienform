"""The curiosity focus a run took, carried to the governor so it survives CLI compaction.

Live 2026-10-17: run 480ee593ead6 took focus a21f96c9, compacted about 12 minutes in, and the
summary lost the task. The focus and the "write focus_disposition" instruction lived only in the
kickoff prompt, so no disposition was written and the focus reads unknown.

This model rides the run end to end as a typed field (never scraped from the prompt):

    Hub kickoff -> CuriosityRunBriefV1.focus -> (durable runner) CuriosityTurnRequestV1.focus
      -> Hub turn -> HarnessRunRequestV1.curiosity_focus -> governor task anchor
         (orion/harness/task_anchor.py::build_focus_anchor, --append-system-prompt)

Set only when the run took a focus (orion/curiosity/focus.py::take_focus). Every other run has
None, and None is omitted from every dump, so older readers of the ``extra="forbid"`` models see
byte-identical payloads.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Same cap the queue tool applies (orion/schemas/introspect.py FOCUS_TEXT_CAP).
FOCUS_QUESTION_MAX_CHARS = 500


class CuriosityFocusAnchorV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus_id: str = Field(pattern=r"^[0-9a-f-]{8,64}$")
    # The curiosity run that took it: the :TurnOutcome's run_id (orion/curiosity/worldview.py).
    run_id: str = Field(pattern=r"^[0-9a-f]{6,32}$")
    question: str = Field(min_length=1, max_length=FOCUS_QUESTION_MAX_CHARS)
    origin: Literal["self", "juniper"]
    # Orion's own graph name (Hub config), so the anchor can say where the node goes.
    graph: str = Field(default="", pattern=r"^[A-Za-z0-9_.:-]{0,64}$")
