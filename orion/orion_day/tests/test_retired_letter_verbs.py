"""Gate: the bare letter verbs and Hub's letter timer stay retired (living Prime design #3224, Part 3).

The day letter is Prime's end-of-day turn: one durable unified turn ordered by Prime's funded ``make`` act. The
two cortex verbs it replaced, Hub's letter timer and the curiosity carry-forward offer are gone, kill-means-kill
(no fallback). This test fails if any of them comes back anywhere in the code, config, CI or verb registry.
Prose (markdown: docs and READMEs that record the retirement) is exempt; so are the two files that name it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RETIRED = (
    "orion_day_note_v1",
    "orion_day_carry_forward_v1",
    "ORION_DAY_NOTE_VERB",
    "ORION_DAY_CARRY_FORWARD_VERB",
    "LLM_ORION_DAY_",
    "OrionDayLetterLoop",
    "take_carry_forward",
    "release_carry_forward",
    "HUB_CURIOSITY_CARRY_FORWARD",
    "HUB_ORION_DAY_CARRY_FORWARD_TTL_HOURS",
    "HUB_ORION_DAY_HOUR_LOCAL",
    "scripts.orion_day_letter",
    # Deleted test files a CI workflow must not still try to run.
    "test_orion_day_verbs.py",
    "tests/test_orion_day_letter.py",
)
# Files that name what was retired, in prose, on purpose.
ALLOWED = {
    "orion/orion_day/tests/test_retired_letter_verbs.py",
    "orion/orion_day/letter_turn.py",
}
SCANNED_TOP = ("orion/", "services/", "scripts/", "config/", ".github/")
SCANNED_SUFFIX = (".py", ".yaml", ".yml", ".j2", ".json", ".toml", ".cfg", ".ini", ".sh", ".env_example",
                  ".html", ".js", "Dockerfile", "Makefile")


def _tracked_and_new() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if p.startswith(SCANNED_TOP) and p.endswith(SCANNED_SUFFIX)]


def test_no_retired_letter_verb_timer_or_offer_is_referenced_anywhere():
    hits = []
    for rel in _tracked_and_new():
        if rel in ALLOWED:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue  # deleted in the working tree
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits += [f"{rel}: {name}" for name in RETIRED if name in text]
    assert hits == [], "retired letter verbs / timer / carry-forward offer referenced:\n" + "\n".join(hits)


def test_the_verb_and_prompt_files_are_gone():
    for rel in ("orion/cognition/verbs/orion_day_note_v1.yaml", "orion/cognition/verbs/orion_day_carry_forward_v1.yaml",
                "orion/cognition/prompts/orion_day_note_v1.j2", "orion/cognition/prompts/orion_day_carry_forward_v1.j2",
                "services/orion-hub/scripts/orion_day_letter.py", "orion/orion_day/carry_forward.py"):
        assert not (ROOT / rel).exists(), rel
