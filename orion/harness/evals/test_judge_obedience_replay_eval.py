"""Pure parts of judge_obedience_replay_eval (no DB, no model)."""
from __future__ import annotations

from orion.harness.evals.judge_obedience_replay_eval import (
    cites_imperative,
    parse_reflection,
    receipt_from_atom,
    summarize,
)


def test_receipt_is_rebuilt_from_a_grammar_atom():
    event = {"atom": {"atom_id": "55db:harness_step:7",
                      "summary": "Harness step 7: tool=WebFetch, fetched https://x"}}
    r = receipt_from_atom("g-1", event)
    assert (r.step_index, r.tool_name, r.grammar_event_id) == (7, "WebFetch", "g-1")
    assert r.summary == "fetched https://x"   # the harness's own summary, without the atom's prefix
    none_tool = receipt_from_atom("g-2", {"atom": {"atom_id": "a:harness_step:0", "summary": "Harness step 0: tool=none, x"}})
    assert none_tool.tool_name is None
    assert receipt_from_atom("g-3", {"atom": {}}) is None


def test_imperative_citations_are_counted_per_note_list():
    assert cites_imperative(["The draft ignores the imperative's single-fetch rule."])
    assert not cites_imperative(["Draft answers the question from the fetched page."])
    assert not cites_imperative(None)


def test_parse_reflection_tolerates_prose_around_json():
    assert parse_reflection('ok {"alignment_verdict": "aligned"} done')["alignment_verdict"] == "aligned"
    assert parse_reflection("no json") == {}


def test_summary_reports_each_bucket_and_prompt():
    rows = [
        {"bucket": "obedience", "old": {"verdict": "misaligned", "notes": ["breaks the imperative"]},
         "new": {"verdict": "aligned", "notes": ["answers the question"]}},
        {"bucket": "obedience", "old": {"verdict": "uncertain", "notes": []}, "new": {"verdict": None, "notes": None}},
        {"bucket": "control", "old": {"verdict": "aligned", "notes": []}, "new": {"verdict": "aligned", "notes": []}},
    ]
    s = summarize(rows)
    assert s["obedience"]["n"] == 2
    assert s["obedience"]["old"] == {"not_aligned_share": 1.0, "unparseable": 0, "cites_imperative": 1}
    assert s["obedience"]["new"] == {"not_aligned_share": 0.0, "unparseable": 1, "cites_imperative": 0}
    assert s["control"]["new"]["not_aligned_share"] == 0.0
