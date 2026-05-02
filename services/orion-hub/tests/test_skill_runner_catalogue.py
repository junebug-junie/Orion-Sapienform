"""Guards Hub Skill Runner catalogue against drift from templates/index.html."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.skill_runner_catalogue import SKILL_RUNNER_CATALOGUE_VERBS  # noqa: E402


def test_skill_runner_catalogue_keys_are_verbatim_in_index_html() -> None:
    html = (REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html").read_text(encoding="utf-8")
    for prompt in SKILL_RUNNER_CATALOGUE_VERBS:
        assert prompt in html, f"catalogue prompt not found verbatim in index.html: {prompt!r}"


def test_skill_runner_catalogue_entry_count_matches_non_workflow_options() -> None:
    """21 operator catalogue skills (excludes placeholder, workflows, and free-text rows)."""
    assert len(SKILL_RUNNER_CATALOGUE_VERBS) == 21
