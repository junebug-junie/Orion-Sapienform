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
    # Another real attention arc (10-11 morning), cited by nothing here: a known label that is not cited.
    "ae0a3cb57ae73ee4": dict(kind="attention", subject_ref="node:substrate.bus_synaptic",
                             subject_label="Bus synaptic prediction error", status="closed",
                             began_at="2026-10-11T16:51:00Z", ended_at="2026-10-11T17:30:00Z",
                             last_seen_at="2026-10-11T17:30:00Z", attention_returns=1),
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
    # A timed claim about a kind no cited arc is: the citation points at the wrong record.
    c = _inline("Last slept 15 h 53 min ago [arc:381981f663496a47].")
    assert c.outcome == "mismatch"
    assert any(r.startswith("kind:") and "['sleep']" in r and "['attention']" in r for r in c.reasons)


def test_kind_words_without_a_timed_claim_are_not_decisive():
    # Ordinary English: a concern raised while talking with Juniper is still a true sentence.
    assert _inline("I slept through most of it [arc:381981f663496a47].").outcome == "undecidable"
    assert _inline("The thread from talking with Juniper is still open [arc:fd63c7ab0ad3ecbe].").outcome == "undecidable"


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
    assert any(r.startswith("time:") and "moved after the frame" in r for r in c.reasons)


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
    replies = {"c-1": {"src": "chat_history_log", "text": "Last slept 15 h 53 min ago [arc:381981f663496a47]."}}
    report = mod.evaluate(decisions, replies, {k: {"arc_id": k, **v} for k, v in ROWS.items()})
    (row,) = report["rows"]
    assert row["reply_source"] == "chat_history_log"
    assert [s["outcome"] for s in row["sentences"]] == ["match", "mismatch"]
    assert report["weeks"]["2026-W42"]["truth_rate"] == 0.5
    text = mod.render_text(report, AT, ["verified"])
    assert "[MISMATCH] (reply, inline) Last slept 15 h 53 min ago [arc:381981f663496a47]." in text


# ---------------------------------------------------------------- calibration on the renderers' own output


def test_the_stance_cue_is_true_to_its_own_arcs():
    """Every sentence the cue template writes from a real frame must read as a match: if the checker
    calls the renderer's own output wrong, the checker is wrong."""
    from orion.temporal_self.stance_cue import project_stance_cue
    from orion.temporal_self.tests.patch4_fixtures import BUSY_NOW, TALKING_NOW, frame
    from orion.temporal_self.tests.test_thread_arcs import frame_with_thread_arcs

    from datetime import timedelta

    # Also 5 min later, with the frame ~10 min old at the turn (the cue accepts up to 900 s).
    cases = [(frame_with_thread_arcs(), TALKING_NOW), (frame("busy"), BUSY_NOW)]
    cases += [(f, now + timedelta(minutes=5)) for f, now in cases]
    for f, now in cases:
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


# ---------------------------------------------------------------- review: never call a true sentence wrong


def test_a_lagging_frame_does_not_turn_a_true_count_into_a_mismatch():
    # The frame (<= 900 s old) said 2 returns; the row ticked to 3 five minutes before the turn.
    lag = {"381981f663496a47": ArcFacts.from_row({"arc_id": "381981f663496a47", **ROWS["381981f663496a47"],
                                                  "attention_returns": 3, "last_seen_at": "2026-10-12T00:54:41Z"})}
    c = check_sentence("Attention on Codebase prediction error, back to it 2 times [arc:381981f663496a47].",
                       source="tone", at=AT, cited=["381981f663496a47"], attribution="inline", arcs=lag)
    assert c.outcome == "match"  # subject; the count is undecidable, not a mismatch
    assert any(r.startswith("count:") and "moved since the frame" in r for r in c.reasons)


def test_a_paraphrased_subject_is_undecidable():
    c = _inline("Keep attention on the codebase errors for now [arc:381981f663496a47].")
    assert not any(r.startswith("subject:") and "another arc" in r for r in c.reasons)
    assert c.outcome != "mismatch"


def test_a_label_holding_a_separator_or_a_kind_phrase_is_read_whole():
    rows = {"aaaaaaaaaaaaaaaa": ArcFacts.from_row(dict(
        arc_id="aaaaaaaaaaaaaaaa", kind="attention", subject_ref="node:x", subject_label="curiosity runs, for Juniper",
        status="suspended", began_at="2026-10-12T00:30:00Z", last_seen_at="2026-10-12T00:50:00Z", attention_returns=1))}
    c = check_sentence("On now: attention on curiosity runs, for Juniper for 29 min, back to it 1 time today "
                       "[arc:aaaaaaaaaaaaaaaa].", source="tone", at=AT, cited=["aaaaaaaaaaaaaaaa"],
                       attribution="inline", arcs=rows)
    assert c.outcome == "match", c.reasons
    assert not any(r.startswith("kind:") and "curiosity" in r for r in c.reasons)


def test_hedged_times_are_bounds_not_values():
    # Juniper's last turn was 23:15:43, 1 h 44 min before AT.
    assert _inline("Talking with Juniper was over 1 h ago [arc:571bce17764b0a9e].").outcome == "match"
    assert _inline("Talking with Juniper was more than 5 h ago [arc:571bce17764b0a9e].").outcome == "match"  # kind
    c = _inline("Talking with Juniper was more than 5 h ago [arc:571bce17764b0a9e].")
    assert not any(o for o in c.reasons if "off by" in o)
    assert _inline("I talked with Juniper about 2 h ago [arc:571bce17764b0a9e].").outcome == "match"
    assert _inline("Juniper last spoke under 3 h ago [arc:571bce17764b0a9e].").outcome == "match"


def test_for_outside_the_template_is_not_a_start_time():
    c = _inline("I have not been talking with Juniper for 1 h 44 min [arc:571bce17764b0a9e].")
    assert c.outcome == "match", c.reasons  # matched against last seen, not began


def test_an_uncited_kind_makes_a_time_miss_undecidable():
    c = _inline("2 open threads, and she last spoke 2 h 50 min ago [arc:684d45c2010540ed] [arc:fd63c7ab0ad3ecbe] "
                "[arc:571bce17764b0a9e].")
    assert c.outcome != "mismatch", c.reasons


def test_a_clock_naming_the_time_of_writing_is_undecidable():
    c = _inline("It is 18:59 local and the thread remains [arc:fd63c7ab0ad3ecbe].")  # AT is 18:59 MDT
    assert c.outcome == "undecidable"
    assert any("time of writing" in r for r in c.reasons)


def test_open_thread_count_in_a_reply_never_mismatches():
    c = check_sentence("There is 1 open thread I keep coming back to.", source="reply", at=AT,
                       cited=["684d45c2010540ed", "fd63c7ab0ad3ecbe"], attribution="evidence_refs", arcs=ARCS)
    assert c.outcome != "mismatch"


def test_reasons_quote_without_trailing_punctuation():
    c = _inline("Last slept 2 h ago [arc:460c1b680514c3e8].")
    assert any("'2 h ago'" in r for r in c.reasons)
