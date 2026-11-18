"""The one story naming rule, shared by orion-sql-writer (write time) and the Hub (read time)."""

from __future__ import annotations

from orion.vision.story_naming import evidence_crops, name_from_crops, rename_subject, subject


def test_rule_names_only_one_named_id_covering_two_thirds_with_no_other_id() -> None:
    labels = {"a": "Maya", "a2": "maya", "b": "Theo"}
    assert name_from_crops([("a", False)] * 3, labels) == ("a", "Maya", None)
    assert name_from_crops([("a", False), ("a", False), (None, False)], labels) == ("a", "Maya", None)
    assert name_from_crops([("a", False), (None, False), (None, False)], labels)[2] == "low_share"
    assert name_from_crops([("a", False), ("a", False), ("x", False)], labels)[2] == "other_id"
    assert name_from_crops([("a", False), ("b", False)], labels)[2] == "two_named"
    assert name_from_crops([("a", False), ("a", True), ("a", False)], labels)[2] == "edge_clipped"
    # Several ids of one name (any case) are one person; the id on most crops is returned.
    assert name_from_crops([("a2", False), ("a", False), ("a2", False)], labels) == ("a2", "maya", None)
    assert name_from_crops([(None, False)], labels) == (None, None, None)
    assert name_from_crops([("a", False)], {"a": "  "}) == (None, None, None)


def test_evidence_crops_reads_stored_story_evidence() -> None:
    ev = [{"individual_id": "a", "clipped": False}, {"individual_id": None, "clipped": True}, "junk"]
    assert evidence_crops(ev) == [("a", False), (None, True)]


def test_rename_subject_swaps_only_the_leading_subject() -> None:
    s = "A person with a dog walked right to left, over 5 s (3 detections)."
    assert rename_subject(s, "person", None, "maya") == "Maya with a dog walked right to left, over 5 s (3 detections)."
    assert rename_subject("Maya walked.", "person", "Maya", None) == "A person walked."
    assert rename_subject("Maya walked.", "person", "Maya", "Maya") == "Maya walked."
    # Never guess: a sentence that does not start with the old subject is left alone.
    assert rename_subject("Someone walked.", "person", None, "Theo") == "Someone walked."
    assert rename_subject("Alex walked.", "person", "Al", "Theo") == "Alex walked."
    assert subject("dog", None) == "A dog"
