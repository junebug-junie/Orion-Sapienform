"""Start-vs-resume decisions for a killed FCC turn (orion/harness/session_resume.py)."""

from __future__ import annotations

from pathlib import Path

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
    sr.mark_end("run-1", sr.STATE_COMPLETED, home=tmp_path)
    second = sr.plan_attempt("run-1", home=tmp_path)
    assert second.resume is False
    assert second.session_id != first.session_id
    assert second.reason == "previous_attempt_completed"


def test_a_context_ceiling_death_is_not_resumed(tmp_path):
    first = sr.plan_attempt("run-1", home=tmp_path)
    _transcript(tmp_path, first.session_id)
    sr.mark_end("run-1", sr.STATE_CEILING, home=tmp_path)
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
    _transcript(tmp_path, first.session_id, size=5000)
    second = sr.plan_attempt("run-1", home=tmp_path, window_chars=4000)
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
    sr.mark_end("run-1", sr.STATE_INTERRUPTED, home=tmp_path)
    assert sr.plan_attempt("run-1", home=tmp_path).resume is True


def test_continuation_prompt_does_not_restate_the_task(tmp_path):
    plan = sr.ResumePlan(session_id="s", resume=True, resumes=1, reason="r")
    text = sr.continuation_prompt(plan)
    assert "earlier work" in text and "do not redo" in text
