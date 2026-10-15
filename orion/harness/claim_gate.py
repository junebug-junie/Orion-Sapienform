"""A reply may only name a `correction_id` that a `record_correction` result in the same turn returned.

Live 2026-10-15 (turn 5fdf2faf): Orion was asked to record a correction. The tool was never
called, yet the reply listed a `record_correction(...)` call and `correction_id:
corr_9_218_reproduce` -- an id no tool returned (real ids are UUIDs). The repair step that wrote
the reply has no tool access; it wrote what a success would look like. The brief already tells
Orion never to claim a recording without a returned id. Nothing enforced it.

This is the enforcement, kept narrow on purpose: it checks one structured claim (an id token
after the words `correction_id`) against the turn's real tool results. It does not try to read
intent from prose. Pure and synchronous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# A delimiter (`:` or `=`, optionally quoted) must sit between the label and the id, and the id
# must contain a digit, `_` or `-`: "each correction id should be recorded" is prose, not a claim.
_CLAIM = re.compile(
    r"(?P<lead>correction[_ ]id[\"'`]?\s*[:=]\s*[\"'`]?)"
    r"(?P<id>(?=[A-Za-z0-9_\-]*[0-9_\-])[A-Za-z0-9][A-Za-z0-9_\-]{5,63})",
    re.IGNORECASE,
)

UNVERIFIED_MARK = "[unverified]"


@dataclass(frozen=True)
class ClaimGateResult:
    text: str
    unverified_ids: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.unverified_ids)


def gate_correction_claims(final_text: str | None, correction_results: Sequence[str]) -> ClaimGateResult:
    """Replace every claimed correction id no tool result contains, and say so at the end."""
    text = final_text or ""
    if not text:
        return ClaimGateResult(text=text, unverified_ids=())
    haystack = "\n".join(correction_results)
    unverified: list[str] = []

    def _returned(claimed: str) -> bool:
        # Whole-token match: "123456789" must not pass because a result holds "1234567890".
        return re.search(r"(?<![\w-])" + re.escape(claimed) + r"(?![\w-])", haystack) is not None

    def _check(match: re.Match[str]) -> str:
        claimed = match.group("id")
        if _returned(claimed):
            return match.group(0)
        if claimed not in unverified:
            unverified.append(claimed)
        return match.group("lead") + UNVERIFIED_MARK

    gated = _CLAIM.sub(_check, text)
    if not unverified:
        return ClaimGateResult(text=text, unverified_ids=())
    notice = (
        "\n\n[Unverified: this reply named a correction_id ("
        + ", ".join(unverified)
        + ") that no record_correction result in this turn returned. Nothing was recorded.]"
    )
    return ClaimGateResult(text=gated + notice, unverified_ids=tuple(unverified))
