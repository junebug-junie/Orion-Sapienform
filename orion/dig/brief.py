"""The one always-on line that tells Orion the dig kit exists. Everything else is ``dig help``."""
from __future__ import annotations

import os

DIG_BRIEF_LINE = (
    "Dig kit: `python -P -m orion.dig <cmd>` in Bash. If Juniper asks, in any words, to look at, name, claim or write anything about your body, organs, capacities or past work, run `<cmd>` help first, use what it lists, and write the claims `capacities` asks for: Juniper can veto any. What is using your power, what your activity costs: `orion.dig body-economy hosts`. Recent PRs: `orion.dig history 2`, not the GitHub tool. To see a camera now: `orion.dig look office` (or walkway), then Read the path it prints. A Hub 'ref consult <id>': `orion.dig consult <id>`. Before telling Juniper you don't remember or have no record of something, run `orion.dig know <words>` and use every hit that bears on it, weak ones too. If it could not search properly (skipped, degraded, timed out, error), say you could not look, never that you have no memory."
)
#: Raised 300 -> 400 (2026-10-30) for the power/cost pointer; still one short line per prompt.
#: 2026-10-30: Juniper's "claim your capacities" got a prose self-portrait and no tool call (twice);
#: the line now sends any body/self request, in any words, to `help` first, so new commands need no new phrase.
#: 2026-11-04: raised 400 -> 420. Orion ran `capacities` on "what can you do" turns, then offered
#: claims instead of writing them (ledger still empty). Since 2026-11-04 a claim holds its job while its
#: evidence checks on each read; Juniper can veto any (vsm_confirmed.yaml `rejected`).
#: 2026-11-13: raised 420 -> 520 for the camera pointer (`look`, design 2026-11-13-camera-look-design.md):
#: without it a "look at the walkway" turn has no way to learn the command exists.
#: 2026-11-17: raised 520 -> 640 for "look before you deny" (#3535, Recall rev 2): on 2026-11-14 Orion said
#: it had no memory of something episode_memory held; `know` searches the same lived memories recall injects.
#: 2026-11-17: raised 640 -> 740 (#3663): a search that was skipped, degraded, timed out or errored is not a
#: basis for "I have no memory"; Orion says it could not look instead.
#: 2026-11-25: raised 740 -> 800 for the consult pointer: Juniper names a consult in chat by the
#: "ref consult <id>" on its Hub card, and consults are never in chat history (static/js/consult.js).
#: 2026-11-25: raised 800 -> 850 for "use every hit, weak ones too" (corr 8297d0c3): a weak hit that
#: answered the question was left out of the reply, which then said the rest was "not in my records".
#: The line is 841 chars: the next addition needs a raise here and in the same breath a dated reason.
MAX_BRIEF_CHARS = 850


def dig_brief_enabled() -> bool:
    return (os.environ.get("HARNESS_FCC_DIG_BRIEF_ENABLED") or "true").strip().lower() not in ("0", "false", "no", "off")


def append_dig_harness_brief(parts: list[str], *, reading_only: bool = False) -> None:
    """``reading_only`` turns get no Bash (fcc_motor: ``--tools WebFetch,WebSearch``), so no line."""
    if reading_only or not dig_brief_enabled():
        return
    parts.append(DIG_BRIEF_LINE)
