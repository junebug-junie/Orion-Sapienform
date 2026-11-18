"""Boundary Rule 3 (Juniper, 2026-10-01) and the named legacy close reasons."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _load(rel_path: str, name: str):
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    sys.path.insert(0, str(SERVICE_ROOT))
    spec = importlib.util.spec_from_file_location(name, SERVICE_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


boundary = _load("app/boundary.py", "mc_boundary_rule3")
episode_shadow = importlib.import_module("app.episode_shadow")

from orion.schemas.memory_consolidation import MemoryTurnPersistedV1  # noqa: E402


class _S:
    MEMORY_BOUNDARY_SCORE_THRESHOLD = 0.70
    MEMORY_BOUNDARY_LLM_ONLY_THRESHOLD = 0.85
    MEMORY_BOUNDARY_OVERRIDE_THRESHOLD = 0.92
    MEMORY_WINDOW_FALLBACK_GAP_SEC = 5400


@pytest.mark.parametrize(
    "phase, score, gap, expected, reason",
    [
        ("long_gap", 0.0, 19000, True, "v2:phase_long_gap"),
        ("next_day", None, 70000, True, "v2:phase_next_day"),
        ("stale_thread", 0.1, 200000, True, "v2:phase_stale_thread"),
        ("resumed_thread", 0.92, 6000, True, "v2:resumed_thread+llm"),
        ("resumed_thread", 0.919, 6000, False, "v2:resumed_thread_below_llm_threshold"),
        ("resumed_thread", None, 6000, False, "v2:resumed_thread_below_llm_threshold"),
        ("same_breath", 1.0, 30, False, "v2:phase_same_breath"),
        ("short_pause", 1.0, 900, False, "v2:phase_short_pause"),
        (None, 1.0, 5400, True, "v2:no_phase_time_gap"),
        (None, 1.0, 5399, False, "v2:no_phase_no_gap"),
        ("unknown", 1.0, 100, False, "v2:no_phase_no_gap"),
    ],
)
def test_rule3_matrix(phase, score, gap, expected, reason):
    got, why = boundary.rule3_boundary(phase=phase, boundary_score=score, gap_sec=gap, settings=_S())
    assert (got, why) == (expected, reason)


def test_rule3_needs_no_llm_score_for_reorient_phases():
    """The spec drops the >=0.70 LLM condition for long_gap/next_day/stale_thread."""
    assert boundary.rule3_boundary(phase="long_gap", boundary_score=0.0, gap_sec=None, settings=_S())[0]


def _turn(phase: str | None, *, response: str = "ok") -> MemoryTurnPersistedV1:
    meta = {"conversation_phase": {"phase_change": phase}} if phase else {}
    return MemoryTurnPersistedV1(correlation_id="c1", prompt="p", response=response, spark_meta=meta)


@pytest.mark.parametrize(
    "phase, score, reason",
    [
        ("long_gap", 0.71, "legacy:phase_long_gap+llm"),
        ("long_gap", 0.69, None),
        (None, 0.86, "legacy:unknown_phase+llm"),
        ("short_pause", 0.99, None),
    ],
)
def test_legacy_reason_names_the_branch_without_changing_it(phase, score, reason):
    turn = _turn(phase)
    scores = {"conversation_boundary_score": score}
    assert boundary.legacy_close_reason(turn, scores, _S()) == reason
    assert boundary.should_close_window(turn, scores, _S()) is (reason is not None)


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Workflow: Journal Pass\nStatus: ok", True),
        ("Workflow 'github_compactor_pass' started", True),
        ("  Workflow: Chat History Compact", True),
        ("I think the workflow you built is great", False),
        ("", False),
    ],
)
def test_command_turn_is_the_workflow_runtime_reply_header(response, expected):
    assert episode_shadow.is_workflow_command_turn(response) is expected


def _shadow(corr: str, at: datetime, *, phase=None, score=None, command=False):
    return episode_shadow.ShadowTurn(
        correlation_id=corr,
        at=at,
        phase_change=phase,
        delta_user_seconds=None,
        phase_source=None,
        boundary_score=score,
        is_command=command,
        legacy_close_reason=None,
    )


T0 = datetime(2026, 9, 28, 12, 26, tzinfo=timezone.utc)


def test_closed_event_payload_and_close_lag():
    turns = [
        _shadow("a", T0).entry(v2_boundary=False, v2_reason="v2:first_turn"),
        _shadow("b", T0 + timedelta(minutes=5)).entry(v2_boundary=False, v2_reason="v2:phase_short_pause"),
        _shadow("cmd", T0 + timedelta(hours=3, minutes=30), command=True).entry(
            v2_boundary=False, v2_reason="v2:phase_short_pause"
        ),
    ]
    closing = _shadow("next", T0 + timedelta(hours=9), phase="long_gap", score=0.4)
    ev = episode_shadow.build_closed_event(
        episode_id="ep1",
        source_platform=None,
        turns=turns,
        started_at=T0,
        closing=closing,
        close_reason="v2:phase_long_gap",
    )
    assert ev.turn_ids == ["a", "b", "cmd"]
    assert ev.juniper_turn_count == 2 and ev.command_turn_count == 1
    assert ev.ended_at == T0 + timedelta(hours=3, minutes=30)
    assert ev.close_lag_sec == pytest.approx(5.5 * 3600)
    assert ev.phase_at_close == "long_gap" and ev.boundary_score_at_close == 0.4
    assert ev.episode_status == "closed" and ev.skip_reason is None
    assert "next" not in ev.turn_ids  # the boundary turn opens the next episode


def test_command_only_episode_is_skipped():
    turns = [_shadow("cmd", T0, command=True).entry(v2_boundary=False, v2_reason="v2:first_turn")]
    ev = episode_shadow.build_closed_event(
        episode_id="ep2",
        source_platform=None,
        turns=turns,
        started_at=T0,
        closing=_shadow("x", T0 + timedelta(hours=6), phase="long_gap"),
        close_reason="v2:phase_long_gap",
    )
    assert ev.episode_status == "skipped" and ev.skip_reason == "command_only"


def test_closed_event_resolves_and_is_registered():
    from orion.schemas.registry import SCHEMA_REGISTRY, resolve
    from orion.schemas.memory_episode import MEMORY_EPISODE_CLOSED_KIND, MemoryEpisodeClosedV1

    assert resolve("MemoryEpisodeClosedV1") is MemoryEpisodeClosedV1
    assert SCHEMA_REGISTRY["MemoryEpisodeClosedV1"].kind == MEMORY_EPISODE_CLOSED_KIND


class _FlagOff(_S):
    MEMORY_LEGACY_BOUNDARY_USE_PHASE = False


class _FlagOn(_S):
    MEMORY_LEGACY_BOUNDARY_USE_PHASE = True


def test_live_legacy_path_does_not_see_the_new_stamp_by_default():
    """Fix 1 must not change live window closing unless Juniper opts in.

    A stamped long_gap turn with judge 0.80: legacy-with-phase would close
    (long_gap and >= 0.70); the pre-Fix-1 legacy path saw "unknown" and needed
    >= 0.85, so it did not. Default keeps the old answer.
    """
    turn = _turn("long_gap")
    scores = {"conversation_boundary_score": 0.80}
    hidden = boundary.legacy_view(turn, _FlagOff())
    assert "conversation_phase" not in hidden.spark_meta
    assert boundary.should_close_window(hidden, scores, _FlagOff()) is False
    assert "conversation_phase" in turn.spark_meta  # the shadow still reads the real stamp
    shown = boundary.legacy_view(turn, _FlagOn())
    assert boundary.should_close_window(shown, scores, _FlagOn()) is True


def test_classify_prompt_phase_line_unchanged_by_default():
    from orion.memory.consolidation_classify import build_classify_prompt

    turn = _turn("short_pause")
    prompt = build_classify_prompt(
        prompt=turn.prompt, response=turn.response, spark_meta=boundary.legacy_view(turn, _FlagOff()).spark_meta
    )
    assert "phase=unknown" in prompt
