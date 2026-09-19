"""Hub turn splice of Mind work-shape into role-teach (motor prompt only)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for key in list(sys.modules):
    if key == "scripts" or key.startswith("scripts."):
        del sys.modules[key]
    if key == "app" or key.startswith("app."):
        del sys.modules[key]
for candidate in (REPO_ROOT, HUB_ROOT):
    try:
        sys.path.remove(str(candidate))
    except ValueError:
        pass
for candidate in (REPO_ROOT, HUB_ROOT):
    sys.path.insert(0, str(candidate))

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

from orion.hub.turn_orchestrator import _maybe_splice_role_teach_disclosure

_KICKOFF = (
    "YOUR ROLE FOR THIS SITTING.\n"
    "ASKING FOR CONTRACTOR HELP (orion_worldview).\n"
    '  MERGE (h:HelpRequest {help_id: "<unique help id>"})\n'
    "Pick something.\n"
)

_SHAPE = {
    "expected_depth": "deep",
    "cross_cutting": "yes",
    "foresight_note": "multi-hop archaeology likely",
}


def test_maybe_splice_orion_shape_flag_on_inserts_foresight() -> None:
    out = _maybe_splice_role_teach_disclosure(
        _KICKOFF,
        utterance_origin="orion",
        mind_work_shape=_SHAPE,
        enabled=True,
    )
    assert "Mind work-shape for this sitting (advisory):" in out
    assert "multi-hop archaeology likely" in out
    assert "expected depth: deep" in out
    # Prefer insert before contractor-help marker.
    assert out.index("Mind work-shape") < out.index("ASKING FOR CONTRACTOR HELP")


def test_maybe_splice_juniper_origin_unchanged() -> None:
    out = _maybe_splice_role_teach_disclosure(
        _KICKOFF,
        utterance_origin="juniper",
        mind_work_shape=_SHAPE,
        enabled=True,
    )
    assert out == _KICKOFF


def test_maybe_splice_flag_off_unchanged() -> None:
    out = _maybe_splice_role_teach_disclosure(
        _KICKOFF,
        utterance_origin="orion",
        mind_work_shape=_SHAPE,
        enabled=False,
    )
    assert out == _KICKOFF


def test_maybe_splice_missing_shape_unchanged() -> None:
    out = _maybe_splice_role_teach_disclosure(
        _KICKOFF,
        utterance_origin="orion",
        mind_work_shape=None,
        enabled=True,
    )
    assert out == _KICKOFF


def test_maybe_splice_all_unknown_shape_unchanged() -> None:
    out = _maybe_splice_role_teach_disclosure(
        _KICKOFF,
        utterance_origin="orion",
        mind_work_shape={
            "expected_depth": "unknown",
            "cross_cutting": "unknown",
            "foresight_note": "unknown",
        },
        enabled=True,
    )
    assert out == _KICKOFF


def test_role_teach_disclosure_flag_defaults_true() -> None:
    from app.settings import Settings

    s = Settings()
    assert s.HUB_CURIOSITY_ROLE_TEACH_DISCLOSURE is True


def test_execute_unified_turn_wires_maybe_splice() -> None:
    """Call site must exist after proceed / before harness user_message use."""
    source = (REPO_ROOT / "orion" / "hub" / "turn_orchestrator.py").read_text()
    assert "async def execute_unified_turn(" in source
    # Prefer the assignment call site, not the helper definition.
    marker = "user_message = _maybe_splice_role_teach_disclosure("
    assert marker in source
    idx = source.index(marker)
    window = source[idx : idx + 400]
    assert "utterance_origin=utterance_origin" in window
    assert "mind_work_shape=thought.mind_work_shape" in window
    assert "HUB_CURIOSITY_ROLE_TEACH_DISCLOSURE" in window
