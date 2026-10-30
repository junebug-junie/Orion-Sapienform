"""The one always-on line that tells Orion the dig kit exists. Everything else is ``dig help``."""
from __future__ import annotations

import os

DIG_BRIEF_LINE = (
    "Your dig kit is `python -P -m orion.dig <cmd>` in Bash; `help` lists your history and body tools. "
    "What is using your power, what your activity costs: `python -P -m orion.dig body-economy hosts`. "
    "Recent PRs: `python -P -m orion.dig history 2`, not the GitHub tool. "
    "Asked to name your organs, run `name-organs`; to claim or write your capacities, run `capacities`, then act on its last line."
)
#: Raised 300 -> 400 (2026-10-30) for the power/cost pointer; still one short line per prompt.
#: 2026-10-30: Juniper's "claim your capacities" got a prose self-portrait and no tool call (twice);
#: the line now maps her two rung phrases to their commands.
MAX_BRIEF_CHARS = 400


def dig_brief_enabled() -> bool:
    return (os.environ.get("HARNESS_FCC_DIG_BRIEF_ENABLED") or "true").strip().lower() not in ("0", "false", "no", "off")


def append_dig_harness_brief(parts: list[str], *, reading_only: bool = False) -> None:
    """``reading_only`` turns get no Bash (fcc_motor: ``--tools WebFetch,WebSearch``), so no line."""
    if reading_only or not dig_brief_enabled():
        return
    parts.append(DIG_BRIEF_LINE)
