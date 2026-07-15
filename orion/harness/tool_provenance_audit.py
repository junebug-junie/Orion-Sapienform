"""Post-hoc audit: did this turn's draft claim live computation for content
that was actually a tool fetch?

The FCC motor (fcc_motor.py) spawns a single-shot ``claude -p`` subprocess
with no mid-run injection point, so nothing here can prevent a confabulated
claim before it's generated -- that's what compile_harness_prefix's CONTEXT
PROVENANCE block is for. What this module can do is compare, after the fact,
in the same turn: did the tool trace (grammar_receipts, already collected by
HarnessRunner.run()) include a fetch-shaped tool call, and does draft_text
use immediacy language ("this turn", "right now", ...) that would only be
true of live substrate signal? See
project_orion_substrate_bridge_confabulation for the incident this audits
against: a GitHub file fetch narrated as live computation "this turn".

Pure and synchronous by design -- no bus/schema wiring yet (that's the
second, separate patch once this detection heuristic is validated).
"""

from __future__ import annotations

import re
from typing import Sequence

from orion.schemas.harness_finalize import GrammarReceiptV1

# Tool names shaped like "go read something and bring back its contents" --
# matched against GrammarReceiptV1.tool_name. Deliberately narrow (false
# negatives are safer than false positives for an audit signal): a generic
# code-execution or search tool isn't itself evidence of a static-content
# fetch the way a file-read/get-contents/fetch call is.
_FETCH_SHAPED_TOOL_PATTERN = re.compile(
    r"get_file_contents|read_file|web[_-]?fetch|fetch_url|read_content",
    re.IGNORECASE,
)

_IMMEDIACY_PATTERN = re.compile(
    # Deliberately no bare "computing" -- that fires on any generic use
    # ("cloud computing costs") with no temporal/proximity claim attached.
    # The incident's actual phrase ("computing in the background this
    # turn") is still caught by "in the background" and "this turn" alone.
    r"\bthis turn\b|\bright now\b|\bhappening now\b|\bin the background\b",
    re.IGNORECASE,
)


def fetch_shaped_tool_names(receipts: Sequence[GrammarReceiptV1]) -> list[str]:
    return sorted(
        {
            receipt.tool_name
            for receipt in receipts
            if receipt.tool_name and _FETCH_SHAPED_TOOL_PATTERN.search(receipt.tool_name)
        }
    )


def detect_tool_provenance_mismatch(
    draft_text: str,
    receipts: Sequence[GrammarReceiptV1],
) -> str | None:
    """None unless this turn both fetched content via a tool AND the draft
    uses live-immediacy language -- the pairing the incident this audits was
    an instance of. Either alone is legitimate: a fetch with no immediacy
    claim, or immediacy language backed by a real live substrate cue with no
    tool fetch involved."""
    if not draft_text or not receipts:
        return None
    fetch_tools = fetch_shaped_tool_names(receipts)
    if not fetch_tools:
        return None
    if not _IMMEDIACY_PATTERN.search(draft_text):
        return None
    return (
        "tool_provenance_mismatch: draft uses live-immediacy language but this "
        "turn's tool trace shows content fetched via " + ", ".join(fetch_tools)
    )
