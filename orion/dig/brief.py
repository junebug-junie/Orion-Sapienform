"""The one always-on line that tells Orion the dig kit exists. Everything else is ``dig help``."""
from __future__ import annotations

import os

DIG_BRIEF_LINE = (
    "To look at your own history or body (the repo as it was on any day since 2025-06-24, what "
    "changed since, merged PRs, your organs and body economy now), run "
    "`python -P -m orion.dig help` in Bash. "
    "For recent PRs run `python -P -m orion.dig history 2` (40 lines max), not the GitHub tool."
)
MAX_BRIEF_CHARS = 300


def dig_brief_enabled() -> bool:
    return (os.environ.get("HARNESS_FCC_DIG_BRIEF_ENABLED") or "true").strip().lower() not in ("0", "false", "no", "off")


def append_dig_harness_brief(parts: list[str], *, reading_only: bool = False) -> None:
    """``reading_only`` turns get no Bash (fcc_motor: ``--tools WebFetch,WebSearch``), so no line."""
    if reading_only or not dig_brief_enabled():
        return
    parts.append(DIG_BRIEF_LINE)
