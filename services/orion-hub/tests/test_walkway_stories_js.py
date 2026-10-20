"""walkway-stories.js pure helpers for the story people boxes (2026-10-18), run in node."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parents[1] / "static" / "js" / "walkway-stories.js"


def _node(expr: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    script = f"const api = require({json.dumps(str(JS))}); process.stdout.write(JSON.stringify({expr}));"
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_person_title_names_or_says_unnamed_with_a_short_id() -> None:
    got = _node("""[
      api.personTitle({label: "Maya", role: "walker"}),
      api.personTitle({label: null, kind: "person", short_id: "abcdef1234", role: "walker"}),
      api.personTitle({label: null, kind: "dog", short_id: "d0g", role: "companion"}),
      api.personTitle({label: "<b>x</b>", role: "walker"})
    ]""")
    assert got == ["Maya", "unnamed person (individual abcdef12)", "With them: unnamed dog (individual d0g)", "<b>x</b>"]


def test_action_line_says_what_juniper_said_on_this_story() -> None:
    got = _node("""[
      api.actionLine({action: "name", name: "Theo"}),
      api.actionLine({action: "someone_else", name: "Cora"}),
      api.actionLine({action: "someone_else", name: null}),
      api.actionLine({action: "unsure"}),
      api.actionLine(null),
      api.actionLine({action: "other"})
    ]""")
    assert got == ["You named them Theo on this story.", "You said this is someone else: Cora.",
                   "You said this is someone else.", "You said you are not sure.", "", ""]
