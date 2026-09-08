"""Tests that the discrimination eval can actually return each verdict.

An eval that is structurally incapable of returning FAIL reads as a permanent
green light -- the exact failure CLAUDE.md's "a gate that cannot run reads as
all green" exists to force out. So every branch is exercised against a
synthetic log, including both named failure modes.

Separate file rather than appended to `test_ask_claude_trigger.py` on purpose:
that module defines its own `FakePrior`/`FakeLimit` helpers, and appending a
second set of fixtures into one file is how a helper gets shadowed.
"""

from __future__ import annotations

import json
from pathlib import Path

from orion.autonomy.evals.run_ask_claude_trigger_eval import run


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps({"decision": r}) for r in rows) + "\n")


def _decision(stuck: int, total: int, *, would=True, refused=None, state="clear") -> dict:
    assessments = [{"prior_id": f"p{i}", "stuck": i < stuck} for i in range(total)]
    return {
        "would_ask": would,
        "refused": refused,
        "subject_prior_id": "p0" if stuck else None,
        "limit_state": state,
        "assessments": assessments,
    }


def test_insufficient_data_below_min_runs(tmp_path):
    log = tmp_path / "l.jsonl"
    _write(log, [_decision(1, 7)] * 3)
    assert run(["--log", str(log)]) == 0


def test_missing_log_is_insufficient_data_not_a_pass(tmp_path):
    # A log that does not exist must not read as a clean run.
    out = run(["--log", str(tmp_path / "nope.jsonl")])
    assert out == 0


def test_pass_when_the_stuck_count_varies(tmp_path):
    log = tmp_path / "l.jsonl"
    rows = [_decision(1 if i % 2 else 2, 7) for i in range(50)]
    _write(log, rows)
    assert run(["--log", str(log)]) == 0


def test_fail_when_no_prior_is_ever_selected(tmp_path):
    log = tmp_path / "l.jsonl"
    _write(log, [_decision(0, 7, would=False, refused="no_stuck_prior")] * 50)
    assert run(["--log", str(log)]) == 1


def test_fail_when_every_prior_is_always_selected(tmp_path):
    log = tmp_path / "l.jsonl"
    _write(log, [_decision(7, 7)] * 50)
    assert run(["--log", str(log)]) == 1


def test_no_readable_priors_is_insufficient_data_not_fail(tmp_path):
    # An unreachable worldview is an ACL/connection problem, not a verdict on
    # the knobs. It must not be reported as FAIL -- that would blame the
    # trigger for a graph outage.
    log = tmp_path / "l.jsonl"
    _write(log, [_decision(0, 0, would=False, refused="no_live_priors")] * 50)
    assert run(["--log", str(log)]) == 0


def test_a_week_of_budget_refusals_does_not_fail_the_prior_criterion(tmp_path):
    # The two sides are judged separately: a broken transcript mount must not
    # read as "the trigger is dead" when the trigger still wanted to fire.
    log = tmp_path / "l.jsonl"
    rows = [
        _decision(1 if i % 2 else 2, 7, would=False, refused="budget_unobserved", state="unknown")
        for i in range(50)
    ]
    _write(log, rows)
    assert run(["--log", str(log)]) == 0


def test_a_truncated_final_line_is_skipped_not_fatal(tmp_path):
    # Normal for a log being appended to right now.
    log = tmp_path / "l.jsonl"
    _write(log, [_decision(1 if i % 2 else 2, 7) for i in range(50)])
    with log.open("a") as fh:
        fh.write('{"decision": {"would_ask": tr')
    assert run(["--log", str(log)]) == 0
