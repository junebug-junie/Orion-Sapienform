"""Patch 4: the citation gate. Unknown arc ids removed and counted; uncited day sentences labelled."""
from __future__ import annotations

from orion.temporal_self.citations import check_arc_citations, gate_arc_refs

SHOWN = ["2ce0b486a356bbd2", "2545de72655edb66"]
LABELS = {"Harness closure prediction error": ["2ce0b486a356bbd2"], "15:00": ["2545de72655edb66"]}


def test_known_ref_is_kept_and_verified():
    text = "Most of my attention went to Harness closure prediction error [arc:2ce0b486a356bbd2]."
    out, check = check_arc_citations(text, SHOWN, labels=LABELS)
    assert out == text
    assert check.cited_valid == ["arc:2ce0b486a356bbd2"] and check.rejected_unknown == []
    assert check.label == "verified" and check.action == "kept" and check.shown == 2


def test_unknown_ref_is_removed_and_counted():
    text = "I slept well [arc:deadbeefdeadbeef]. Juniper spoke at 15:00 [arc:2545de72655edb66]."
    out, check = check_arc_citations(text, SHOWN, labels=LABELS)
    assert out == "I slept well. Juniper spoke at 15:00 [arc:2545de72655edb66]."
    assert check.rejected_unknown == ["arc:deadbeefdeadbeef"] and check.action == "stripped"
    assert check.cited_valid == ["arc:2545de72655edb66"]


def test_garbled_arc_ref_is_removed_too():
    out, check = check_arc_citations("x [arc:NOT-HEX]", SHOWN)
    assert out == "x" and check.rejected_unknown == ["arc:NOT-HEX"]


def test_uncited_day_sentence_is_unverified_not_blocked():
    text = "Harness closure prediction error held me most of the day. Then I rested."
    out, check = check_arc_citations(text, SHOWN, labels=LABELS)
    assert out == text  # never blocked, never rewritten
    assert check.uncited_day_sentences == 1 and check.label == "UNVERIFIED"
    assert check.uncited_examples == ["Harness closure prediction error held me most of the day."]


def test_clock_label_matches_local_times_not_utc_times():
    _, local = check_arc_citations("Juniper spoke at 15:00.", SHOWN, labels=LABELS)
    _, utc = check_arc_citations("The run finished at 15:00 UTC.", SHOWN, labels=LABELS)
    assert local.uncited_day_sentences == 1 and utc.uncited_day_sentences == 0


def test_label_cited_elsewhere_counts_as_cited():
    _, check = check_arc_citations("Stay with the Harness closure prediction error thread.", SHOWN,
                                   labels=LABELS, cited_elsewhere=["2ce0b486a356bbd2"])
    assert check.label == "verified" and check.uncited_day_sentences == 0


def test_text_about_plumbing_without_day_labels_is_no_claim():
    _, check = check_arc_citations("The bus and the harness are my plumbing; I can talk about them.", SHOWN,
                                   labels=LABELS)
    assert check.label == "no_day_claims" and check.uncited_day_sentences == 0


def test_gate_arc_refs_keeps_non_arc_and_shown_arc_refs():
    kept, rejected = gate_arc_refs(["hub:turn:abc", "arc:2ce0b486a356bbd2", "arc:deadbeefdeadbeef", "arc:junk"], SHOWN)
    assert kept == ["hub:turn:abc", "arc:2ce0b486a356bbd2"]
    assert rejected == ["arc:deadbeefdeadbeef", "arc:junk"]


def test_bracketed_arc_ref_in_a_ref_list_is_the_same_citation():
    """Live 2026-10-12: the stance model wrote evidence_refs ['hub:turn', '[arc:<id>]']."""
    kept, rejected = gate_arc_refs(["hub:turn", "[arc:2ce0b486a356bbd2]", "arc:2ce0b486a356bbd2",
                                    "[arc:deadbeefdeadbeef]"], SHOWN)
    assert kept == ["hub:turn", "arc:2ce0b486a356bbd2"] and rejected == ["arc:deadbeefdeadbeef"]


def test_bare_unknown_arc_in_text_is_removed_and_counted():
    out, check = check_arc_citations("Look at arc:0123456789abcdff here.", SHOWN)
    assert out == "Look at here." and check.rejected_unknown == ["arc:0123456789abcdff"]


def test_bare_shown_arc_in_text_cites_its_sentence():
    _, check = check_arc_citations("Harness closure prediction error held me (arc:2ce0b486a356bbd2).", SHOWN,
                                   labels=LABELS)
    assert check.cited_valid == ["arc:2ce0b486a356bbd2"] and check.label == "verified"


def test_uppercase_shown_id_is_the_same_id():
    out, check = check_arc_citations("x [arc:2CE0B486A356BBD2]", SHOWN)
    assert check.rejected_unknown == [] and check.cited_valid == ["arc:2ce0b486a356bbd2"]
    kept, rejected = gate_arc_refs(["ARC:2CE0B486A356BBD2"], SHOWN)
    assert kept == ["arc:2ce0b486a356bbd2"] and rejected == []


def test_labels_match_whole_words_only():
    labels = {"memory": ["2ce0b486a356bbd2"]}
    _, inside = check_arc_citations("I am memoryless today.", SHOWN, labels=labels)
    _, whole = check_arc_citations("My memory was busy.", SHOWN, labels=labels)
    assert inside.uncited_day_sentences == 0 and whole.uncited_day_sentences == 1
