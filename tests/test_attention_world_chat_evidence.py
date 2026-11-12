"""world:chat: a stable label (stable loop id) and Juniper's latest message as evidence.

Spec: docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md,
"Recommended next patch" 3. Live bug 2026-11-05..12: the label carried the
message count, so one conversation got a new open-loop id each time the count
moved (3 ids, 24 id switches between chat wins less than 15 min apart).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.attention.world_first import (
    CHAT_EVIDENCE_EXCERPT_CHARS,
    CHAT_EVIDENCE_KIND,
    CHAT_LATEST_TURN_SQL,
    CHAT_TALKING_LABEL,
    CHAT_TURN_TIMES_SQL,
    chat_candidate,
    chat_message_excerpt,
    judge_candidate,
)
from orion.substrate.attention.scoring import open_loop_id

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
WEEK = [NOW - timedelta(days=d, minutes=30) for d in range(1, 7)]


def _cand(*minutes_ago: float, latest=None):
    turns = WEEK + [NOW - timedelta(minutes=m) for m in minutes_ago]
    return chat_candidate(turns, now=NOW, latest_turn=latest)


def _loop_id(label: str) -> str:
    # The same formula orion/substrate/attention/scoring.py uses for loop ids.
    return open_loop_id(label)


def test_label_and_loop_id_do_not_move_with_the_message_count() -> None:
    one, two, three = _cand(1), _cand(1, 3), _cand(1, 3, 5)
    assert [int(c.unusualness.value) for c in (one, two, three)] == [1, 2, 3]
    assert one.label == two.label == three.label == CHAT_TALKING_LABEL
    assert len({_loop_id(c.label) for c in (one, two, three)}) == 1
    assert not any(ch.isdigit() for ch in one.label)


def test_latest_message_rides_as_evidence_with_the_count() -> None:
    latest = ("turn-42", NOW - timedelta(minutes=1), "  are you\n still   there?  ")
    cand = _cand(1, 3, latest=latest)
    assert judge_candidate(cand).eligible
    ev = cand.evidence
    assert ev == {
        "kind": CHAT_EVIDENCE_KIND,
        "chat_turn_id": "turn-42",
        "said_at": (NOW - timedelta(minutes=1)).isoformat(),
        "message_excerpt": "are you still there?",
        "truncated": False,
        "messages_in_window": 2,
        "window_min": 15,
    }
    # Evidence never changes the reading or the refs (ranking is untouched).
    assert cand.evidence_refs == ["world:chat", "chat_history_log"]
    assert cand.unusualness == _cand(1, 3).unusualness


def test_no_evidence_without_a_turn_in_the_window() -> None:
    old = ("turn-1", NOW - timedelta(minutes=15), "hello")
    assert _cand(1, latest=old).evidence is None  # exactly at the window edge: not counted
    assert _cand(1, latest=("t", NOW + timedelta(minutes=1), "future")).evidence is None
    quiet = chat_candidate(WEEK, now=NOW, latest_turn=("t", NOW - timedelta(minutes=1), "hi"))
    assert quiet.evidence is None and quiet.label == "chat is quiet"
    absent = chat_candidate(None, now=NOW, latest_turn=("t", NOW - timedelta(minutes=1), "hi"))
    assert absent.absent and absent.evidence is None


def test_malformed_or_empty_latest_turn_is_no_evidence_not_an_error() -> None:
    for bad in (None, object(), ("t", "not-a-time", "hi"), ("t", NOW, None), ("t", NOW - timedelta(minutes=1), "   ")):
        assert _cand(1, latest=bad).evidence is None


def test_excerpt_is_capped_and_marked() -> None:
    text = "word " * 100
    excerpt, truncated = chat_message_excerpt(text)
    assert truncated and len(excerpt) == CHAT_EVIDENCE_EXCERPT_CHARS and excerpt.endswith("…")
    cand = _cand(1, latest=("t", NOW - timedelta(minutes=1), text))
    assert cand.evidence["truncated"] is True
    assert len(cand.evidence["message_excerpt"]) <= CHAT_EVIDENCE_EXCERPT_CHARS


def test_latest_turn_query_counts_the_same_turns_as_the_rate() -> None:
    where = CHAT_TURN_TIMES_SQL.split("chat_history_log ", 1)[1].rsplit("ORDER BY", 1)[0]
    assert where.strip().startswith("WHERE") and where in CHAT_LATEST_TURN_SQL
    assert "ORDER BY created_at DESC LIMIT 1" in CHAT_LATEST_TURN_SQL


def test_evidence_is_not_in_the_debug_trace() -> None:
    cand = _cand(1, latest=("t", NOW - timedelta(minutes=1), "private words"))
    assert "private words" not in str(judge_candidate(cand).trace())


def test_loop_id_replay_eval_flags_churn_and_passes_the_fix() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "services/orion-substrate-runtime/evals/run_world_chat_loop_id_replay.py"
    spec = importlib.util.spec_from_file_location("chat_loop_replay", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = [f"Juniper is talking ({n} message{'s' if n != 1 else ''} in the last 15 min)" for n in (1, 2, 3, 2, 1)]
    rows = [(NOW + timedelta(minutes=2 * i), _loop_id(d), d) for i, d in enumerate(old)]
    report = mod.evaluate(rows)
    assert report["old_distinct_ids"] == 3 and report["old_switches_within_window"] == 4
    assert report["new_distinct_ids"] == 1 and report["new_switches_within_window"] == 0
    assert report["pass"]
    tampered = rows[:-1] + [(rows[-1][0], "open-loop-wrong", rows[-1][2])]
    assert not mod.evaluate(tampered)["checks"]["formula_reproduces"]
    assert not mod.evaluate([])["pass"]
