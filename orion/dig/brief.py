"""The one always-on line that tells Orion the dig kit exists. Everything else is ``dig help``."""
from __future__ import annotations

import os

DIG_BRIEF_LINE = (
    "Dig kit: `python -P -m orion.dig <cmd>` in Bash. If Juniper asks, in any words, to look at, name, claim or write anything about your body, organs, capacities or past work, run `<cmd>` help first and use what it lists. What is using your power, what your activity costs: `orion.dig body-economy hosts`. Recent PRs: `orion.dig history 2`, not the GitHub tool."
)
#: Raised 300 -> 400 (2026-10-30) for the power/cost pointer; still one short line per prompt.
#: 2026-10-30: Juniper's "claim your capacities" got a prose self-portrait and no tool call (twice);
#: the line now sends any body/self request, in any words, to `help` first, so new commands need no new phrase.
MAX_BRIEF_CHARS = 400


def dig_brief_enabled() -> bool:
    return (os.environ.get("HARNESS_FCC_DIG_BRIEF_ENABLED") or "true").strip().lower() not in ("0", "false", "no", "off")


def append_dig_harness_brief(parts: list[str], *, reading_only: bool = False) -> None:
    """``reading_only`` turns get no Bash (fcc_motor: ``--tools WebFetch,WebSearch``), so no line."""
    if reading_only or not dig_brief_enabled():
        return
    parts.append(DIG_BRIEF_LINE)
