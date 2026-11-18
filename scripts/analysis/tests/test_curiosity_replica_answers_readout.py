"""The replica's answers readout (scripts/analysis/curiosity_replica_answers_readout.py): stance from the act note,
arm from test_before, the 14-day refusal-rate alarm (design 2026-11-04 section 9)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import curiosity_replica_answers_readout as ro  # noqa: E402
from orion.curiosity.replica.answers import AnswerRecordV1, answer_note  # noqa: E402

TODAY = datetime(2026, 11, 30, 12, 0, tzinfo=timezone.utc)


def row(i, stance, *, arm="treated", delivered=True, days_ago=0.0):
    at = TODAY - timedelta(days=days_ago)
    rec = AnswerRecordV1(answer_act_id=f"a{i}", prime_act_id=f"p{i}", kind="focus", target_id="f", stance=stance,
                         reason="because", arm=arm, delivered=delivered, at=at)
    return {"act_id": f"a{i}", "parent_act_id": f"p{i}", "test_before": f"holdback:{arm}", "opened_at": at,
            "outcome_value": 1.0, "outcome_note": answer_note(rec)}


def test_the_readout_reads_the_notes_the_replica_writes():
    rows = [row(1, "refuse"), row(2, "challenge", arm="control", delivered=False), row(3, "support"),
            row(4, "refuse", delivered=False)]
    out = ro.summarize(ro.parse_rows(rows), today=TODAY)
    assert out["by_stance_arm"] == {"challenge/control": 1, "refuse/treated": 2, "support/treated": 1}
    assert out["treated_challenge_or_refuse"] == 2 and out["delivered"] == 1 and not out["alarm"]


def test_fourteen_days_over_half_raise_the_alarm_and_a_gap_does_not():
    rows = [row(i, "refuse" if i % 4 else "support", days_ago=20 - i * 0.5) for i in range(40)]
    out = ro.summarize(ro.parse_rows(rows), today=TODAY)
    assert out["alarm"] and out["verdict"].startswith("ALARM") and len(out["daily_refusal_rate"]) == 14
    young = ro.summarize(ro.parse_rows([row(1, "refuse", days_ago=3)]), today=TODAY)
    assert not young["alarm"] and young["daily_refusal_rate"][0][1] is None
    assert ro.summarize([], today=TODAY)["verdict"].startswith("NO DATA")
