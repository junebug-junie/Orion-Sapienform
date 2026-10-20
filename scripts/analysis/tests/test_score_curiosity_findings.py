"""Fast checks for the curiosity-finding scorer and the replay's yield rule. No DB, no network."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import replay_attention_world_candidates as replay  # noqa: E402
import score_curiosity_findings as score  # noqa: E402

T0 = datetime(2026, 10, 10, tzinfo=timezone.utc)


def _row(i: int, register: str, novelty: float) -> dict:
    return {"run_id": f"r{i:03d}", "t": (T0 + timedelta(hours=i)).isoformat(), "register": register,
            "novelty": novelty}


def test_operational_and_no_finding_never_compete_and_never_enter_history():
    rows = [_row(i, "philosophical", 0.1) for i in range(25)]
    rows += [_row(30, "operational", 0.9), _row(31, "no_finding", 0.9)]
    out = {x["run_id"]: x for x in score.assign_verdicts(rows)}
    assert out["r030"]["verdict"] == "no: operational"
    assert out["r031"]["verdict"] == "no: no_finding"
    # a calm philosophical week: identical novelty reads quiet, not competing
    assert out["r024"]["verdict"] == "no: not novel"
    assert out["r024"]["phil_hist_7d"] == 25


def test_short_history_is_insufficient_not_calm():
    rows = [_row(i, "philosophical", 0.1 + i / 100) for i in range(5)]
    assert {x["verdict"] for x in score.assign_verdicts(rows)} == {"no: history<20"}


def test_top_tenth_novelty_competes():
    rows = [_row(i, "philosophical", 0.05 + i / 1000) for i in range(30)]
    rows.append(_row(40, "philosophical", 0.30))
    out = {x["run_id"]: x for x in score.assign_verdicts(rows)}
    assert out["r040"]["verdict"] == "COMPETES"
    assert out["r040"]["nov_pct"] >= 0.9


def test_kappa_perfect_and_chance():
    a = ["philosophical", "operational"] * 10
    assert score.kappa(a, a) == (1.0, 1.0)


def _frame(chat_eligible: bool) -> dict:
    return {"winner": "world:chat" if chat_eligible else None, "candidates": [
        {"source_id": "world:chat", "eligible": chat_eligible, "rank_score": 0.95, "score": 0.95}]}


def test_curiosity_yields_to_eligible_chat():
    events = [(T0 + timedelta(hours=i), 0.05 + i / 1000, f"r{i}", "") for i in range(30)]
    events.append((T0 + timedelta(hours=40), 0.30, "rwin", "a new thought"))
    t = T0 + timedelta(hours=40, seconds=30)
    series = {"curiosity_finding": events}
    with_chat = replay.replay([(t, _frame(True))], series, "curiosity_phil_novel", min_readings=20, examples=1)
    assert with_chat["yielded_to_chat"] == 1 and with_chat["new_candidate_won"] == 0
    without = replay.replay([(t, _frame(False))], series, "curiosity_phil_novel", min_readings=20, examples=1)
    assert without["new_candidate_won"] == 1


def test_table_redactor_leaves_ordinary_words_alone():
    redact = score._load_privacy_redactor()
    assert redact("What am I, and what am I made of? 0.51 node:substrate.chat") == (
        "What am I, and what am I made of? 0.51 node:substrate.chat")
