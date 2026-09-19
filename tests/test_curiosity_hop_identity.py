"""`Hop` identity: the `written_at` clock and the resumed-sitting preamble.

The bug (design doc "Two data defects", re-caught live 2026-09-19 on run
`58b638778228`): a retried curiosity turn under the same run_id restarted its
hop numbering at 1, so `n` could not order a run and the retry redid the
first attempt's work blind. Three pieces fix it -- the kickoff template
stamps `written_at: timestamp()`, every reader orders by that clock (legacy
hops first, then `written_at`, then `n`), and Hub prepends what the earlier
attempt wrote when it re-runs the frozen prompt.
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.curiosity.kickoff_prompt import build_kickoff_prompt, build_resume_preamble
from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.supervisor import group_hops_by_run, parse_reading_batch
from orion.curiosity.worldview import (
    ALL_HOPS_CYPHER,
    HopRecord,
    WorldviewReader,
    WorldviewSnapshot,
    WorldviewUnavailable,
    hop_order_key,
    hops_for_run_cypher,
    next_hop_n,
    read_all_hops,
    read_hop_notes,
)


class _FakeReader(WorldviewReader):
    def __init__(self, *, answers=None, raises=False) -> None:
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.answers = answers or {}
        self.raises = raises

    def query(self, cypher: str):
        if self.raises:
            raise WorldviewUnavailable("ConnectionError: nope")
        for needle, rows in self.answers.items():
            if needle in cypher:
                return rows
        return []


# --- the read side -----------------------------------------------------------


def test_hop_queries_return_written_at_and_do_not_sort_on_n_in_cypher():
    per_run = hops_for_run_cypher("abc123")
    assert "h.written_at AS written_at" in per_run
    assert "ORDER BY" not in per_run
    assert "h.written_at AS written_at" in ALL_HOPS_CYPHER


def test_hop_order_key_legacy_first_then_clock_then_n():
    legacy_3 = HopRecord(run_id="r", n=3, note="c")
    legacy_1 = HopRecord(run_id="r", n=1, note="a")
    clocked_1 = HopRecord(run_id="r", n=1, note="resumed", written_at=2_000)
    clocked_2 = HopRecord(run_id="r", n=2, note="resumed", written_at=1_000)
    ordered = sorted([clocked_1, clocked_2, legacy_3, legacy_1], key=hop_order_key)
    # legacy by n, then the timestamped ones by their clock -- NOT by n,
    # which would put the resumed n=1 before the first attempt's n=3.
    assert ordered == [legacy_1, legacy_3, clocked_2, clocked_1]
    assert hop_order_key((5, None)) == hop_order_key(HopRecord(run_id="r", n=5, note="x"))


def test_read_hop_notes_orders_resumed_attempt_after_first_attempt():
    # The collision shape from run 58b638778228, plus a resumed attempt that
    # continued the count. FalkorDB returns rows in whatever order; the
    # reader must not depend on it.
    rows = [
        {"n": 4, "note": "resumed, fourth", "written_at": 1_789_000_000_004},
        {"n": 1, "note": "first attempt, first"},
        {"n": 5, "note": "resumed, fifth", "written_at": 1_789_000_000_005},
        {"n": 2, "note": "first attempt, second", "written_at": None},
        {"n": 3, "note": "   "},  # blank note dropped, as before
    ]
    reader = _FakeReader(answers={"WHERE h.run_id = 'abc123'": rows})
    assert read_hop_notes(reader, "abc123") == [
        (1, "first attempt, first"),
        (2, "first attempt, second"),
        (4, "resumed, fourth"),
        (5, "resumed, fifth"),
    ]


def test_read_hop_notes_written_at_tolerates_strings_and_junk():
    rows = [
        {"n": 2, "note": "b", "written_at": "1789000000002"},
        {"n": 1, "note": "a", "written_at": "not a clock"},
    ]
    reader = _FakeReader(answers={"WHERE h.run_id = 'abc123'": rows})
    # junk clock -> treated as legacy -> sorts first
    assert read_hop_notes(reader, "abc123") == [(1, "a"), (2, "b")]


def test_next_hop_n_continues_past_the_highest_written():
    assert next_hop_n([]) == 1
    assert next_hop_n([(1, "a"), (1, "a again"), (3, "c")]) == 4


def test_read_all_hops_carries_written_at_and_none_for_legacy():
    rows = [
        {"run_id": "abc123", "n": 1, "note": "old"},
        {"run_id": "abc123", "n": 2, "note": "new", "written_at": 1_789_000_000_000},
    ]
    hops = read_all_hops(_FakeReader(answers={"MATCH (h:Hop)": rows}))
    assert hops == [
        HopRecord(run_id="abc123", n=1, note="old", written_at=None),
        HopRecord(run_id="abc123", n=2, note="new", written_at=1_789_000_000_000),
    ]


def test_group_hops_by_run_orders_within_run_by_clock_not_n():
    hops = [
        HopRecord(run_id="r1", n=1, note="resumed first", written_at=200),
        HopRecord(run_id="r1", n=2, note="first attempt second"),
        HopRecord(run_id="r1", n=1, note="first attempt first"),
    ]
    [(run_id, ordered)] = group_hops_by_run(hops, run_order={"r1": 1})
    assert run_id == "r1"
    assert [h.note for h in ordered] == [
        "first attempt first",
        "first attempt second",
        "resumed first",
    ]


# --- the reading contract ----------------------------------------------------


def _reading(n, **extra):
    return {
        "hop_n": n,
        "about_prior_id": None,
        "kind": "test",
        "moved_the_claim": None,
        "reading_confidence": 0.5,
        "reasoning": "r",
        **extra,
    }


def test_parse_reading_batch_stamps_hop_written_at_from_caller_not_model():
    payload = {
        "readings": [
            _reading(1, hop_written_at=999),  # the model's number is ignored
            _reading("2"),  # string hop_n still maps to the clock
            _reading(3),
        ]
    }
    out = parse_reading_batch(
        payload,
        run_id="r1",
        hop_ns=[1, 2, 3],
        written_at_by_n={1: 1_000, 2: 2_000, 3: None},
    )
    assert [(r.hop_n, r.hop_written_at) for r in out] == [(1, 1_000), (2, 2_000), (3, None)]


def test_parse_reading_batch_without_map_stamps_none():
    out = parse_reading_batch({"readings": [_reading(1)]}, run_id="r1", hop_ns=[1])
    assert out[0].hop_written_at is None


# --- the write side (what Orion is told) -------------------------------------


def _material() -> StudyMaterial:
    return StudyMaterial(generated_at=datetime(2026, 9, 19, tzinfo=timezone.utc))


def test_kickoff_template_stamps_written_at_with_the_graph_clock():
    text = build_kickoff_prompt(_material(), view=WorldviewSnapshot(), run_id="abc123")
    assert 'CREATE (:Hop {run_id: "abc123", n: 1, note:' in text
    assert "written_at: timestamp()})" in text
    assert "`n` counts up from 1 within this sitting" in text


def test_kickoff_without_a_writable_graph_says_nothing_about_written_at():
    text = build_kickoff_prompt(_material(), run_id="abc123", graph_enabled=False)
    assert "written_at" not in text
    assert "CREATE (:Hop" not in text


def test_resume_preamble_is_empty_when_nothing_to_resume():
    assert build_resume_preamble([], run_id="abc123") == ""
    assert build_resume_preamble([(1, "   ")], run_id="abc123") == ""


def test_resume_preamble_lists_prior_hops_and_names_the_next_n():
    text = build_resume_preamble(
        [(1, "looked at the ledger"), (1, "same n, earlier collision"), (3, "third")],
        run_id="abc123",
    )
    assert text.startswith('RESUMED SITTING. This run (run_id "abc123")')
    assert "  n=1: looked at the ledger" in text
    assert "  n=3: third" in text
    assert "Number your next hop n=4" in text
    assert "do not start at 1 again" in text
    # ends with a separator so the frozen prompt reads as a distinct block
    assert text.endswith("----\n\n")
