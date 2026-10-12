"""Truth eval: a sentence citing an arc is checked against that arc's stored facts, deterministically.

Arc rows are the live temporal_self_arc rows the 2026-10-12 00:59:41Z stance (fbac3775) cited,
plus the carried-in concern arc 684d45c2010540ed.
"""
from __future__ import annotations

from datetime import datetime, timezone

from orion.temporal_self.citation_truth import ArcFacts, check_decision, check_sentence, summarize

AT = datetime(2026, 10, 12, 0, 59, 41, tzinfo=timezone.utc)
ROWS = {
    "571bce17764b0a9e": dict(kind="conversation", subject_ref="orion_journal", subject_label="", status="suspended",
                             began_at="2026-10-11T22:03:26Z", ended_at=None, last_seen_at="2026-10-11T23:15:43Z",
                             attention_returns=1),
    "381981f663496a47": dict(kind="attention", subject_ref="node:substrate.codebase",
                             subject_label="Codebase prediction error", status="suspended",
                             began_at="2026-10-12T00:27:08Z", ended_at=None, last_seen_at="2026-10-12T00:44:26Z",
                             attention_returns=1),
    "460c1b680514c3e8": dict(kind="sleep", subject_ref="dc-ae05c9ae329f", subject_label="pressure", status="closed",
                             began_at="2026-10-11T09:05:37Z", ended_at="2026-10-11T09:05:44Z",
                             last_seen_at="2026-10-11T09:05:44Z", attention_returns=0),
    "fd63c7ab0ad3ecbe": dict(kind="concern", subject_ref="open-loop-496dd47e676b", subject_label="", status="open",
                             began_at="2026-10-11T22:46:57Z", ended_at=None, last_seen_at="2026-10-11T22:46:57Z",
                             attention_returns=0),
    "684d45c2010540ed": dict(kind="concern", subject_ref="open-loop-5cb6cd032923", subject_label="", status="open",
                             began_at="2026-10-11T06:00:00Z", ended_at=None, last_seen_at="2026-10-11T23:15:00Z",
                             attention_returns=5),
}
ARCS = {k: ArcFacts.from_row({"arc_id": k, **v}) for k, v in ROWS.items()}


def _inline(sentence: str):
    import re

    ids = re.findall(r"\[arc:([0-9a-f]{16})\]", sentence)
    return check_sentence(sentence, source="imperative", at=AT, cited=ids, attribution="inline", arcs=ARCS)


def test_match_kind_subject_time_and_count():
    c = _inline("On now: attention on Codebase prediction error, set aside 15 min ago, back to it 1 time today "
                "[arc:381981f663496a47].")
    assert c.outcome == "match", c.reasons
    assert any(r.startswith("subject:") for r in c.reasons)
    assert any(r.startswith("time:") and "last seen" in r for r in c.reasons)


def test_wrong_kind_is_a_mismatch():
    c = _inline("I slept through most of it [arc:381981f663496a47].")
    assert c.outcome == "mismatch"
    assert any("names a sleep arc" in r and "attention" in r for r in c.reasons)


def test_wrong_time_is_a_mismatch():
    c = _inline("Last slept 2 h ago [arc:460c1b680514c3e8].")  # it ended 15 h 54 min before
    assert c.outcome == "mismatch"
    assert any(r.startswith("time:") and "off by" in r for r in c.reasons)


def test_right_time_on_a_closed_sleep_matches():
    assert _inline("Last slept 15 h 53 min ago [arc:460c1b680514c3e8].").outcome == "match"


def test_wrong_subject_is_a_mismatch():
    c = _inline("Attention on Bus synaptic prediction error for 30 min [arc:381981f663496a47].")
    assert c.outcome == "mismatch" and any(r.startswith("subject:") for r in c.reasons)


def test_no_readable_fact_is_undecidable():
    c = _inline("That one still matters to me [arc:fd63c7ab0ad3ecbe].")
    assert c.outcome == "undecidable"
    assert c.reasons == ["no fact the deterministic check can read (no template phrase, time, clock or count)"]


def test_missing_arc_row_is_undecidable():
    c = _inline("Juniper last spoke 3 min ago [arc:00000000000000ff].")
    assert c.outcome == "undecidable" and c.reasons[0] == "arc row missing: [arc:00000000000000ff]"


def test_row_that_moved_after_the_sentence_is_undecidable_not_mismatch():
    moved = dict(ARCS)
    moved["571bce17764b0a9e"] = ArcFacts.from_row(
        {"arc_id": "571bce17764b0a9e", **ROWS["571bce17764b0a9e"], "last_seen_at": "2026-10-12T01:30:00Z"})
    c = check_sentence("On now: talking with Juniper, last turn 40 min ago [arc:571bce17764b0a9e].", source="tone",
                       at=AT, cited=["571bce17764b0a9e"], attribution="inline", arcs=moved)
    assert c.outcome == "match"  # kind still matches
    assert any(r.startswith("time:") and "moved after the sentence" in r for r in c.reasons)


def test_return_count_above_the_row_is_a_mismatch():
    assert _inline("Attention on Codebase prediction error, back to it 4 times [arc:381981f663496a47].").outcome == "mismatch"


def test_open_thread_count_matches_the_cited_concern_arcs():
    c = _inline("2 open threads from conversation [arc:684d45c2010540ed] [arc:fd63c7ab0ad3ecbe].")
    assert c.outcome == "match", c.reasons
    c = _inline("1 open thread from conversation [arc:684d45c2010540ed] [arc:fd63c7ab0ad3ecbe].")
    assert c.outcome == "mismatch"


def test_reply_sentence_is_checked_against_evidence_refs_and_instructions_are_skipped():
    checks, notes = check_decision(
        at=AT,
        cited_valid=["arc:571bce17764b0a9e", "arc:381981f663496a47"],
        texts={
            "imperative": "Run a self-audit; cite arc ids where available.",  # instruction: skipped
            "tone": "Curious, unbothered.",
            "reply": "Juniper last spoke 1 h 44 min ago. I like the quiet. I slept 3 h ago.",
        },
        arcs=ARCS,
    )
    assert notes == []
    assert [(c.source, c.attribution, c.outcome) for c in checks] == [
        ("reply", "evidence_refs", "match"),
        ("reply", "evidence_refs", "undecidable"),  # no sleep arc cited: may rest on an uncited record
    ]


def test_nothing_to_check_is_named():
    checks, notes = check_decision(at=AT, cited_valid=["arc:571bce17764b0a9e"],
                                   texts={"imperative": "Answer warmly.", "tone": "warm", "reply": None}, arcs=ARCS)
    assert checks == []
    assert notes == ["no reply text stored",
                     "no sentence to check: no inline arc citation and no reply sentence stating a checkable day fact"]


def test_weekly_summary_truth_rate():
    rows = [
        {"created_at": AT.isoformat(), "sentences": [{"outcome": "match"}, {"outcome": "mismatch"},
                                                     {"outcome": "undecidable"}, {"outcome": "match"}]},
        {"created_at": AT.isoformat(), "sentences": []},
    ]
    week = summarize(rows)["2026-W42"]
    assert week == {"rows": 2, "rows_with_sentences": 1, "sentences": 4, "match": 2, "mismatch": 1,
                    "undecidable": 1, "truth_rate": 0.667, "decided_share": 0.75}


def test_eval_entrypoint_evaluates_db_shaped_rows():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "evals" / "run_day_citation_truth_eval.py"
    spec = importlib.util.spec_from_file_location("run_day_citation_truth_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    decisions = [{
        "correlation_id": "c-1", "created_at": AT.replace(tzinfo=None),  # thought_decision: naive UTC
        "imperative": "Mention the sleep: last slept 15 h 53 min ago [arc:460c1b680514c3e8].", "tone": "warm",
        "day_citations": {"label": "verified", "cited_valid": ["arc:460c1b680514c3e8"]},
    }]
    replies = {"c-1": {"src": "chat_history_log", "text": "I slept a while ago [arc:381981f663496a47]."}}
    report = mod.evaluate(decisions, replies, {k: {"arc_id": k, **v} for k, v in ROWS.items()})
    (row,) = report["rows"]
    assert row["reply_source"] == "chat_history_log"
    assert [s["outcome"] for s in row["sentences"]] == ["match", "mismatch"]
    assert report["weeks"]["2026-W42"]["truth_rate"] == 0.5
    text = mod.render_text(report, AT, ["verified"])
    assert "[MISMATCH] (reply, inline) I slept a while ago [arc:381981f663496a47]." in text


# ---------------------------------------------------------------- calibration on the renderers' own output


def test_the_stance_cue_is_true_to_its_own_arcs():
    """Every sentence the cue template writes from a real frame must read as a match: if the checker
    calls the renderer's own output wrong, the checker is wrong."""
    from orion.temporal_self.stance_cue import project_stance_cue
    from orion.temporal_self.tests.patch4_fixtures import BUSY_NOW, TALKING_NOW, frame
    from orion.temporal_self.tests.test_thread_arcs import frame_with_thread_arcs

    for f, now in ((frame_with_thread_arcs(), TALKING_NOW), (frame("busy"), BUSY_NOW)):
        arcs = {a["arc_id"]: ArcFacts.from_row(a) for a in f["arcs_today"] + [f["active_arc"], f["previous_arc"]]}
        for t in f["open_threads"]:
            if t.get("arc_id"):
                arcs[t["arc_id"]] = ArcFacts.from_row(dict(
                    arc_id=t["arc_id"], kind="concern", subject_ref=t["subject_ref"], status="open",
                    began_at=t["first_seen_today"], last_seen_at=t["last_returned"]))
        checks, _ = check_decision(at=now, cited_valid=[], texts={"cue": project_stance_cue(f, now).text}, arcs=arcs)
        assert len(checks) >= 3
        assert [c.outcome for c in checks] == ["match"] * len(checks), [c.reasons for c in checks]


def test_the_closed_day_block_group_lines_cite_one_arc_for_a_group_total():
    """Real finding on the 10-09 block: an attention line states the GROUP's returns ("11 returns",
    summed over 11 arcs) but cites only the longest arc, which has 3. The check flags exactly those
    lines; everything else the block says is a match or undecidable."""
    from orion.temporal_self.day_block import render_closed_day_block
    from orion.temporal_self.tests.patch4_fixtures import day_1009

    day = day_1009()
    arcs = {a.arc_id: ArcFacts.from_row(a.model_dump(mode="json")) for a in day.arcs}
    checks, _ = check_decision(at=day.closed_at, cited_valid=[], texts={"block": render_closed_day_block(day).text},
                               arcs=arcs)
    mismatched = [c.sentence for c in checks if c.outcome == "mismatch"]
    assert mismatched == [
        "- Harness closure prediction error: 2 h 15 min over 11 arcs, 11 returns [arc:2ce0b486a356bbd2].",
        "- Execution prediction error: 2 h 14 min over 16 arcs, 13 returns [arc:7bf5c22449afc56d].",
    ]
    assert sum(c.outcome == "match" for c in checks) == 14
