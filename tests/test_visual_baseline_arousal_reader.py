"""R3 reader 4: Orion's arousal defers a due visual-reverie baseline image while strained.

Must-not: arousal never turns any non-issue reason into an issued baseline, is never read
before the other checks pass, and never touches the thermal gate (which lives downstream in
orion-thought and has no arousal input).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.reverie import baseline as b
from orion.schemas.reverie_visual import VisualActivityV1

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)
POLICY = b.VisualBaselinePolicy(enabled=True)
LEVELS = ["engaged", "idle", "strained", None]


class Probe:
    def __init__(self, level):
        self.level, self.calls = level, 0

    def __call__(self):
        self.calls += 1
        return self.level


def activity(**kw):
    return VisualActivityV1(observed_at=kw.pop("observed_at", NOW), history_status=kw.pop("history_status", "ok"), **kw)


def test_strained_defers_by_retry_sec_and_keeps_the_need():
    probe = Probe("strained")
    need, state, reason = b.schedule(activity(), {}, now=NOW, policy=POLICY, arousal_level=probe)
    assert need is None and reason == "arousal_strained" == b.AROUSAL_STRAINED_REASON
    assert probe.calls == 1
    assert datetime.fromisoformat(state["next_attempt_not_before"]) == NOW + timedelta(seconds=POLICY.retry_sec)
    assert "last_attempt_id" not in state
    # Within retry_sec it is an ordinary cooldown; arousal is not read again.
    again = Probe("strained")
    none, state2, reason2 = b.schedule(activity(observed_at=NOW + timedelta(seconds=60)), json.loads(json.dumps(state)),
                                       now=NOW + timedelta(seconds=60), policy=POLICY, arousal_level=again)
    assert none is None and reason2 == "visual_baseline_cooldown" and again.calls == 0
    # After retry_sec, once strain has cleared, the SAME need is issued.
    later = NOW + timedelta(seconds=POLICY.retry_sec + 1)
    issued, _, reason3 = b.schedule(activity(observed_at=later), state2, now=later, policy=POLICY,
                                    arousal_level=Probe("idle"))
    assert issued is not None and reason3 == "visual_baseline_due"
    assert issued.need_id == state["need_id"]


@pytest.mark.parametrize("level", ["engaged", "idle", None])
def test_other_levels_and_unknown_issue_exactly_like_no_reader(level):
    with_reader = b.schedule(activity(), {}, now=NOW, policy=POLICY, arousal_level=Probe(level))
    without = b.schedule(activity(), {}, now=NOW, policy=POLICY)
    assert with_reader[0] == without[0] and with_reader[1] == without[1] and with_reader[2] == without[2] == "visual_baseline_due"


def _non_due_cases():
    """(activity, checkpoint, policy, now) inputs that today never issue a baseline."""
    stale = activity(observed_at=NOW - timedelta(seconds=POLICY.freshness_sec + 5))
    yield "disabled", activity(), {}, b.VisualBaselinePolicy(enabled=False)
    yield "stale", stale, {}, POLICY
    yield "history_error", activity(history_status="unavailable"), {}, POLICY
    yield "not_due", activity(last_success_at=NOW - timedelta(seconds=60), last_success_chain_id="c",
                              last_success_sha256="a" * 64), {}, POLICY
    yield "active", activity(active_attempt_id="att-1"), {}, POLICY
    yield "cooldown", activity(), {"activation_at": NOW.isoformat(),
                                    "next_attempt_not_before": (NOW + timedelta(seconds=30)).isoformat()}, POLICY


@pytest.mark.parametrize("case", list(_non_due_cases()), ids=lambda c: c[0])
@pytest.mark.parametrize("level", LEVELS)
def test_must_not_arousal_never_turns_a_hold_into_an_issue(case, level):
    _, act, checkpoint, policy = case
    probe = Probe(level)
    need, _, reason = b.schedule(act, dict(checkpoint), now=NOW, policy=policy, arousal_level=probe)
    baseline_need, _, baseline_reason = b.schedule(act, dict(checkpoint), now=NOW, policy=policy)
    assert need is None and baseline_need is None
    assert reason == baseline_reason  # the original reason, never arousal's
    assert probe.calls == 0  # arousal is consulted only at the issue point


def test_must_not_thermal_gate_has_no_arousal_input():
    src = (REPO / "orion/autonomy/thermal_gate.py").read_text()
    assert "arousal" not in src and "regulation" not in src


def test_worker_reads_arousal_only_through_the_flagged_reader():
    src = (REPO / "services/orion-proposal-runtime/app/worker.py").read_text()
    assert "arousal_level=self._baseline_arousal_level" in src
    assert "enabled=bool(self._settings.visual_baseline_arousal_enabled)" in src


def test_deferrals_are_recorded_with_their_first_time_and_cleared_on_issue():
    """2026-10-20: the checkpoint names the deferral run so the painting-gap alert can cite it.
    Live 10-18 12:07 -> 10-20: every due baseline deferred, and nothing recorded why or since when."""
    _, state, _ = b.schedule(activity(), {}, now=NOW, policy=POLICY, arousal_level=Probe("strained"))
    assert state[b.DEFERRAL_KEY] == {"reason": "arousal_strained", "since": NOW.isoformat(), "last_at": NOW.isoformat()}
    later = NOW + timedelta(seconds=POLICY.retry_sec + 1)
    _, state, reason = b.schedule(activity(observed_at=later), json.loads(json.dumps(state)), now=later,
                                  policy=POLICY, arousal_level=Probe("strained"))
    assert reason == "arousal_strained"
    assert state[b.DEFERRAL_KEY] == {"reason": "arousal_strained", "since": NOW.isoformat(), "last_at": later.isoformat()}
    issue_at = later + timedelta(seconds=POLICY.retry_sec + 1)
    issued, state, _ = b.schedule(activity(observed_at=issue_at), state, now=issue_at, policy=POLICY,
                                  arousal_level=Probe("idle"))
    assert issued is not None and b.DEFERRAL_KEY not in state


def test_a_new_painting_clears_an_old_deferral():
    _, state, _ = b.schedule(activity(), {}, now=NOW, policy=POLICY, arousal_level=Probe("strained"))
    painted = activity(last_success_at=NOW, last_success_chain_id="new", last_success_sha256="a" * 64)
    _, state, reason = b.schedule(painted, state, now=NOW + timedelta(seconds=60), policy=POLICY)
    assert reason == "visual_baseline_not_due" and b.DEFERRAL_KEY not in state
