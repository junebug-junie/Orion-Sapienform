"""Hub turn splice of Mind work-shape into role-teach (motor prompt only)."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

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

from orion.field.queue_contention import SOURCE_DURABLE
from orion.hub.turn_orchestrator import (
    _gather_role_teach_progress_lines,
    _maybe_splice_role_teach_disclosure,
)
from orion.schemas.field_state import FieldStateV1

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


def test_splice_progress_only_without_mind_work_shape() -> None:
    prompt = "intro\nASKING FOR CONTRACTOR HELP. rest"
    out = _maybe_splice_role_teach_disclosure(
        prompt,
        utterance_origin="orion",
        mind_work_shape=None,
        enabled=True,
        progress_lines=["Access refused at least twice this sitting. Hand off to Cursor now."],
    )
    assert "Hand off to Cursor" in out
    assert out.index("Hand off") < out.index("ASKING FOR CONTRACTOR HELP")


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
    window = source[idx : idx + 500]
    assert "utterance_origin=utterance_origin" in window
    assert "mind_work_shape=thought.mind_work_shape" in window
    assert "HUB_CURIOSITY_ROLE_TEACH_DISCLOSURE" in window
    assert "progress_lines=progress_lines" in window
    assert "_gather_role_teach_progress_lines" in source


def test_gather_composes_refusal_budget_and_fieldstate_queue() -> None:
    state = FieldStateV1(
        generated_at=datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc),
        tick_id="tick_qc_hub",
        queue_contention_score=8.0,
        queue_contention_driver=SOURCE_DURABLE,
    )
    with patch(
        "orion.hub.queue_contention_field_read.read_latest_queue_contention",
        return_value=(state.queue_contention_score, state.queue_contention_driver),
    ):
        lines = _gather_role_teach_progress_lines(
            {
                "role_teach_hop_notes": [
                    "permission denied on durable table",
                    "ACL blocked atlas read",
                ],
                "role_teach_peer_brief": {
                    "status": "refused_budget",
                    "next_hop_n": 3,
                },
            }
        )
    text = "\n".join(lines)
    assert "Access refused" in text
    assert "Cursor budget is spent" in text
    assert "hop 3" in text
    assert "8/10" in text
    assert "durable" in text.lower()
    assert "121" not in text


def test_gather_queue_fail_open_still_emits_refusal() -> None:
    with patch(
        "orion.hub.queue_contention_field_read.read_latest_queue_contention",
        side_effect=RuntimeError("postgres down"),
    ):
        lines = _gather_role_teach_progress_lines(
            {
                "role_teach_hop_notes": [
                    "permission denied",
                    "permission denied again",
                ],
            }
        )
    text = "\n".join(lines)
    assert "Access refused" in text
    assert "Queue pressure" not in text


def test_gather_omits_queue_when_score_zero() -> None:
    with patch(
        "orion.hub.queue_contention_field_read.read_latest_queue_contention",
        return_value=(0.0, SOURCE_DURABLE),
    ):
        lines = _gather_role_teach_progress_lines({})
    assert lines == []


def test_gather_prefers_prebuilt_progress_lines() -> None:
    with patch(
        "orion.hub.queue_contention_field_read.read_latest_queue_contention",
        return_value=(9.0, SOURCE_DURABLE),
    ):
        lines = _gather_role_teach_progress_lines(
            {"role_teach_progress_lines": ["prebuilt only"]}
        )
    assert lines == ["prebuilt only"]


def test_reading_from_field_state_uses_score_attributes() -> None:
    from orion.hub.queue_contention_field_read import reading_from_field_state

    state = FieldStateV1(
        generated_at=datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc),
        tick_id="tick_qc_hub_attr",
        queue_contention_score=4.0,
        queue_contention_driver=SOURCE_DURABLE,
    )
    score, driver = reading_from_field_state(state)
    assert score == 4.0
    assert driver == SOURCE_DURABLE


def test_no_hub_queue_pressure_ewma_keys() -> None:
    """Acceptance: Hub must not own Redis EWMA for this score."""
    forbidden = ("orion:hire:queue_pressure", "queue_pressure:ewma")
    roots = (
        REPO_ROOT / "services" / "orion-hub",
        REPO_ROOT / "orion" / "hub",
    )
    skip_parts = {"node_modules", "__pycache__", ".venv", "tests"}
    for root in roots:
        for path in root.rglob("*.py"):
            if any(part in skip_parts for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for needle in forbidden:
                assert needle not in text, f"{path} contains {needle}"
