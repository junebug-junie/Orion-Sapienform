from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from scripts import api_routes  # noqa: E402


def test_goal_archive_api_dry_run() -> None:
    with patch(
        "orion.autonomy.goal_archive.archive_subjects",
        return_value=[{"subject": "orion", "candidates": 2, "applied": 0, "dry_run": True, "rows_scanned": 5}],
    ):
        payload = api_routes.api_debug_autonomy_goal_archive(
            api_routes.AutonomyGoalArchiveRequest(dry_run=True),
        )

    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["summaries"][0]["subject"] == "orion"
    assert "orion-actions" in payload["note"]
