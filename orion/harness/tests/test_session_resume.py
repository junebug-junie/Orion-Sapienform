"""Start-vs-resume decisions for a killed FCC turn (orion/harness/session_resume.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from orion.harness import session_resume as sr


def _transcript(home: Path, session_id: str, size: int = 100) -> Path:
    path = home / "projects" / "-mnt-orion-fcc-repo" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * size, encoding="utf-8")
    return path


def test_first_attempt_starts_a_session_under_the_stable_id(tmp_path):
    plan = sr.plan_attempt("run-1", home=tmp_path)
    assert (plan.resume, plan.reason) == (False, "first_attempt")
    assert plan.session_id == sr.session_id_for("run-1")


def test_a_killed_attempt_resumes_the_same_session(tmp_path):
    first = sr.plan_attempt("run-1", home=tmp_path)
    _transcript(tmp_path, first.session_id)
    # No mark_end: the process was killed, so the record stays `running`.
    second = sr.plan_attempt("run-1", home=tmp_path)
    assert second.resume is True
    assert second.session_id == first.session_id
    assert second.resumes == 1


def test_key_is_independent_of_the_lease_fenced_correlation_id(tmp_path):
    assert sr.session_id_for("run-1") == sr.session_id_for("run-1")
    assert sr.session_id_for("run-1") != sr.session_id_for("run-2")


def test_a_completed_attempt_does_not_resume_into_the_next_turn(tmp_path):
    first = sr.plan_attempt("run-1", home=tmp_path)
    _transcript(tmp_path, first.session_id)
    sr.mark_end("run-1", sr.STATE_COMPLETED, plan=first, home=tmp_path)
    second = sr.plan_attempt("run-1", home=tmp_path)
    assert second.resume is False
    assert second.session_id != first.session_id
    assert second.reason == "previous_attempt_completed"


def test_a_context_ceiling_death_is_not_resumed(tmp_path):
    first = sr.plan_attempt("run-1", home=tmp_path)
    _transcript(tmp_path, first.session_id)
    sr.mark_end("run-1", sr.STATE_CEILING, plan=first, home=tmp_path)
    second = sr.plan_attempt("run-1", home=tmp_path)
    assert second.resume is False
    assert second.reason == "previous_attempt_hit_context_ceiling"


def test_missing_transcript_starts_fresh_under_a_new_session_id(tmp_path):
    first = sr.plan_attempt("run-1", home=tmp_path)
    second = sr.plan_attempt("run-1", home=tmp_path)
    assert second.resume is False
    assert second.reason == "transcript_missing"
    assert second.session_id != first.session_id


def test_transcript_too_big_for_the_new_lane_window_starts_fresh(tmp_path):
    first = sr.plan_attempt("run-1", home=tmp_path)
    _transcript(tmp_path, first.session_id, size=5000)  # ~2500 chars live after the envelope discount
    assert sr.plan_attempt("run-1", home=tmp_path, window_chars=4000).resume is True
    second = sr.plan_attempt("run-1", home=tmp_path, window_chars=2000)
    assert second.resume is False
    assert second.reason == "transcript_exceeds_lane_window"


def test_resume_count_is_capped(tmp_path):
    plan = sr.plan_attempt("run-1", home=tmp_path, limit=2)
    _transcript(tmp_path, plan.session_id)
    assert sr.plan_attempt("run-1", home=tmp_path, limit=2).resume is True
    assert sr.plan_attempt("run-1", home=tmp_path, limit=2).resume is True
    third = sr.plan_attempt("run-1", home=tmp_path, limit=2)
    assert third.resume is False
    assert third.reason == "resume_limit_reached"


def test_interrupted_error_end_is_still_resumable(tmp_path):
    first = sr.plan_attempt("run-1", home=tmp_path)
    _transcript(tmp_path, first.session_id)
    sr.mark_end("run-1", sr.STATE_INTERRUPTED, plan=first, home=tmp_path)
    assert sr.plan_attempt("run-1", home=tmp_path).resume is True


def test_continuation_prompt_does_not_restate_the_task(tmp_path):
    plan = sr.ResumePlan(session_id="s", resume=True, resumes=1, reason="r")
    text = sr.continuation_prompt(plan)
    assert "earlier work" in text and "do not redo" in text


# --- motor wiring -----------------------------------------------------------

from orion.harness.fcc_motor import _record_session_end, build_claude_argv  # noqa: E402


def _argv(plan):
    return build_claude_argv(
        claude_bin="claude", model_id="m", prompt="THE TASK", correlation_id="c", session_plan=plan
    )


def test_first_attempt_argv_pins_the_session_id_and_keeps_the_task_prompt():
    plan = sr.ResumePlan(session_id="sid-1", resume=False, resumes=0, reason="first_attempt")
    argv = _argv(plan)
    assert argv[:3] == ["claude", "-p", "THE TASK"]
    assert argv[argv.index("--session-id") + 1] == "sid-1"
    assert "--resume" not in argv


def test_resume_argv_replaces_the_task_with_the_continuation_prompt():
    plan = sr.ResumePlan(session_id="sid-1", resume=True, resumes=1, reason="r")
    argv = _argv(plan)
    assert "THE TASK" not in argv
    assert argv[argv.index("--resume") + 1] == "sid-1"
    assert "--session-id" not in argv


def test_no_plan_keeps_legacy_argv():
    argv = _argv(None)
    assert argv[:3] == ["claude", "-p", "THE TASK"]
    assert "--resume" not in argv and "--session-id" not in argv


def _end(frame, tmp_path, monkeypatch, key="run-9"):
    monkeypatch.setenv("HARNESS_FCC_CLAUDE_CONFIG_DIR", str(tmp_path))
    plan = sr.plan_attempt(key)
    out = _record_session_end(frame, resume_key=key, plan=plan)
    return plan, out["metadata"]["session_resume"]["ended"]


def test_a_clean_final_closes_the_turn(tmp_path, monkeypatch):
    assert _end({"type": "final", "metadata": {"a": 1}}, tmp_path, monkeypatch)[1] == sr.STATE_COMPLETED


def test_only_an_outside_kill_leaves_the_turn_resumable(tmp_path, monkeypatch):
    killed = {"type": "error", "error_code": "fcc_nonzero_exit", "metadata": {"exit_code": -9}}
    assert _end(killed, tmp_path, monkeypatch)[1] == sr.STATE_INTERRUPTED


@pytest.mark.parametrize("code", ["fcc_timeout", "fcc_stream_stalled", "fcc_stream_line_limit"])
def test_motor_verdict_endings_close_the_turn_so_the_next_request_is_a_new_turn(tmp_path, monkeypatch, code):
    # The governor finalizes these; resuming them would hijack the next turn of the run.
    plan, ended = _end({"type": "error", "error_code": code}, tmp_path, monkeypatch)
    assert ended == sr.STATE_COMPLETED
    assert sr.plan_attempt("run-9").resume is False


def test_a_cli_error_exit_closes_the_turn(tmp_path, monkeypatch):
    frame = {"type": "error", "error_code": "fcc_nonzero_exit", "metadata": {"exit_code": 1}}
    assert _end(frame, tmp_path, monkeypatch)[1] == sr.STATE_COMPLETED


def test_context_ceiling_is_recorded(tmp_path, monkeypatch):
    frame = {"type": "error", "error_code": "fcc_context_ceiling_exceeded"}
    assert _end(frame, tmp_path, monkeypatch)[1] == sr.STATE_CEILING


def test_a_late_end_from_a_cancelled_attempt_cannot_overwrite_a_newer_attempt(tmp_path):
    a = sr.plan_attempt("run-1", home=tmp_path)
    _transcript(tmp_path, a.session_id)
    b = sr.plan_attempt("run-1", home=tmp_path)  # B took the record while A was dying
    assert b.resume is True
    assert sr.mark_end("run-1", sr.STATE_INTERRUPTED, plan=a, home=tmp_path) is False
    assert sr.mark_end("run-1", sr.STATE_COMPLETED, plan=b, home=tmp_path) is True
    # A's trailing frame arriving after B completed must not reopen the turn.
    assert sr.mark_end("run-1", sr.STATE_INTERRUPTED, plan=a, home=tmp_path) is False
    assert sr.plan_attempt("run-1", home=tmp_path).resume is False


def test_a_corrupt_registry_value_degrades_to_a_fresh_start_not_a_crash(tmp_path):
    first = sr.plan_attempt("run-1", home=tmp_path)
    _transcript(tmp_path, first.session_id)
    path = sr._registry_path(tmp_path, "run-1")
    path.write_text('{"generation": "x", "resumes": [], "state": "running", "token": "t"}')
    plan = sr.plan_attempt("run-1", home=tmp_path)
    assert plan.session_id and plan.reason


def test_plan_gets_a_distinct_token_per_attempt(tmp_path):
    a = sr.plan_attempt("run-1", home=tmp_path)
    b = sr.plan_attempt("run-1", home=tmp_path)
    assert a.token and b.token and a.token != b.token
