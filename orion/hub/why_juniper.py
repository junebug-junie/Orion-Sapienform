"""The ``Why Juniper:`` line every composed outreach message ends with (door policy, 2026-11-10).

Juniper: a message from Orion must say why it is for her, not only that Orion found something.
The composer (tick, curiosity, readings) ends its reply with one line ``Why Juniper: <reason>``.
``split_why_juniper`` takes that line off the delivered text and returns it for the decision row.
Missing or empty means the caller does not send (``no_why_juniper``). This is a presence check on a
structured field, not a judgement of the reason: no second model reads it.

Shared by Hub (``services/orion-hub/scripts/outreach_door_policy.py``) and the prompt builders in
``orion/curiosity/outreach_prompt.py`` and ``orion/world_pulse_read/finding_prompt.py``.
"""
from __future__ import annotations

import re
from typing import Tuple

WHY_JUNIPER_MAX_CHARS = 300

WHY_JUNIPER_INSTRUCTION = (
    "End your reply with one last line, on its own, in exactly this form:\n"
    "Why Juniper: <one short sentence: why this is for her, not only something you found>\n"
    "That line is taken off before she sees the message and kept in your records. If you cannot "
    "say why it is for her, reply PASS instead. A message without that line is not sent."
)

# The separator is a colon, or a dash with space on both sides ("Why Juniper - ..."): a bare hyphen
# ("why juniper-loving people matter") is prose, not the trailer (review finding).
_WHY_LINE = re.compile(
    r"^[\s>*_`#]*why[ _]juniper[*_`]*(?:\s*:|\s+[-–—]\s)[\s*_`]*(?P<why>.*?)[\s*_`]*$",
    re.IGNORECASE,
)
_TRAILING_RULE = re.compile(r"\n\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def split_why_juniper(text: str) -> Tuple[str, str]:
    """(message, why) from a composed reply. Pure.

    The LAST non-empty line must be ``Why Juniper: ...`` (markdown emphasis tolerated). It is
    removed from the message, with a rule line (``---``) the composer put just before it. Anything
    else -- no such line, or the line not last -- gives ``(text, "")``: the caller refuses it."""
    body = str(text or "").rstrip()
    if not body:
        return "", ""
    lines = body.split("\n")
    idx = len(lines) - 1
    m = _WHY_LINE.match(lines[idx])
    if not m:
        return body.strip(), ""
    why = " ".join(m.group("why").split())
    if len(why) > WHY_JUNIPER_MAX_CHARS:
        why = why[: WHY_JUNIPER_MAX_CHARS - 1].rstrip() + "…"
    message = _TRAILING_RULE.sub("", "\n".join(lines[:idx]).rstrip()).strip()
    return message, why
