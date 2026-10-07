"""Gate over the checked-in 30-day boundary replay (run_episode_boundary_replay_eval.py).

Pins what the real data says, including the result the spec did not expect:
Rule 3 keeps the Austin morning as one episode only with the self-comparison
artifact scores. With the judge's real scores it splits it in three, because
the classify prompt never defines BOUNDARY and the judge says YES (>=0.97) to
both resumed_thread turns that morning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_episode_boundary_replay_eval as ev  # noqa: E402


def _report():
    return ev.replay(json.loads(ev.FIXTURE.read_text(encoding="utf-8")))


def test_fixture_holds_no_text():
    raw = json.loads(ev.FIXTURE.read_text(encoding="utf-8"))
    allowed = {"correlation_id", "at", "chat_log_score", "first_pass_score", "is_command"}
    assert all(set(t) <= allowed for t in raw["turns"])


def test_fix2_the_closing_score_was_never_the_persisted_one():
    fix2 = _report()["fix2"]
    assert fix2["closing_turns_with_both_scores"] > 50
    assert fix2["closing_turns_score_mismatch"] >= 0.95 * fix2["closing_turns_with_both_scores"]
    assert fix2["first_pass_mean"] > 0.85 > fix2["chat_log_mean_same_turns"]


def test_rule3_makes_fewer_longer_episodes_than_the_live_windows():
    r = _report()
    assert r["v2_first_pass"]["episodes"] < r["legacy_actual"]["windows"]
    assert r["v2_chat_log"]["episodes"] < r["legacy_actual"]["windows"]


def test_austin_morning_one_episode_only_with_artifact_scores():
    austin = _report()["austin"]
    assert austin["legacy_actual_windows"] >= 9
    artifact = austin["v2_chat_log"]
    assert len(artifact) == 1 and artifact[0]["start_utc"] == "09-28 06:26" and artifact[0]["end_utc"] == "09-28 09:54"
    real = austin["v2_first_pass"]
    assert len(real) == 3
    assert [e["closed_by_phase"] for e in real[:2]] == ["resumed_thread", "resumed_thread"]
