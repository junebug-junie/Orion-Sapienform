"""cooccurrence on in-memory evidence rows (no DB)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_personality_lane_independence import cooccurrence, render, stream  # noqa: E402

T0 = datetime(2026, 10, 17, tzinfo=timezone.utc)
MIN10, SHIFTS = timedelta(minutes=10), [timedelta(hours=6), timedelta(hours=-3)]


def _row(node, lane, minutes, kind=None):
    return {"disposition_id": node, "lane": lane, "occurred_at": T0 + timedelta(minutes=minutes),
            "meta": {"source_kind": kind} if kind else {}}


def _pair(pairs, a, b):
    return next(p for p in pairs if (p["a"], p["b"]) == (a, b))


def test_stream_splits_journal_by_source_kind_only():
    assert stream("journal", {"source_kind": "metacog"}) == "journal:metacog"
    assert stream("journal", {}) == "journal"
    assert stream("reverie", {"source_kind": "metacog"}) == "reverie"


def test_paired_streams_have_high_lift_and_independent_ones_do_not():
    rows = []
    for i in range(6):  # every 2 days: reverie, then a metacog journal one minute later
        rows += [_row("d1", "reverie", i * 2880), _row("d1", "journal", i * 2880 + 1, "metacog"),
                 _row("d1", "dream", i * 2880 + 700)]  # dream ~12 h later: never within 10 min
    pairs = cooccurrence(rows, window=MIN10, shifts=SHIFTS)
    tied = _pair(pairs, "journal:metacog", "reverie")
    assert (tied["n"], tied["near"], tied["shifted"]) == (6, 6, [0, 0])
    assert tied["lift"] == 6.0 and tied["same_source"] is True
    loose = _pair(pairs, "dream", "reverie")
    assert loose["near"] == 0 and loose["lift"] == 0.0 and loose["same_source"] is False


def test_pairs_count_only_within_one_node():
    rows = [_row("d1", "reverie", 0), _row("d2", "journal", 1, "metacog")]
    assert cooccurrence(rows, window=MIN10, shifts=SHIFTS) == []


def test_render_drops_pairs_under_min_hits():
    rows = [_row("d1", "reverie", 0), _row("d1", "dream", 1)]
    text = render(cooccurrence(rows, window=MIN10, shifts=SHIFTS), min_hits=2, window_min=10, shift_hours=[6, -3])
    assert "reverie" not in text.splitlines()[-1]
