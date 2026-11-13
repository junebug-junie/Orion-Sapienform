"""Replay eval over real Strata text-tool-call drafts (fixture copy of 4 of the 23 live turns)."""
from __future__ import annotations

from pathlib import Path

from orion.harness.evals.text_tool_call_replay import main, replay
import json

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "strata_text_tool_call_drafts.json"


def test_every_fixture_turn_is_resumed_and_kept_from_repair():
    report = replay(json.loads(FIXTURE.read_text()))
    assert report["turns"] == 4
    assert report["detected_for_resume"] == report["note_names_toolsearch_select"] == 4
    # A turn whose only step is the call has no findings: the motor fails, nothing is repaired.
    assert report["runner_if_resume_fails"] == {"motor_failed": 4}
    assert report["raw_call_or_repairable_draft_reaches_finalize"] == 0
    assert report["not_detected"] == []


def test_cli_exit_code(capsys):
    assert main(["--json", str(FIXTURE)]) == 0
