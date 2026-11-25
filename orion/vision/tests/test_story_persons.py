"""One person per linked group on the walkway story card (who-card redesign Patch A, 2026-11-24)."""

from __future__ import annotations

import pytest

from orion.vision import story_persons as sp


def _p(iid, *, label=None, crops=(), pics=(), boxes=(), role="walker", copied=False, action=None, thumb=None):
    pictures = [{"story_id": "s" * 32, "file": f"crop_{n}.jpg", "t": t, "crop_id": c} for n, (c, t) in enumerate(pics)]
    shown = [c for c, _ in pics] or ([crops[0]] if thumb else [])
    return {"individual_id": iid, "short_id": iid[:8], "role": role, "kind": "person", "label": label,
            "label_copied": copied, "crop_ids": list(crops), "pictures": pictures, "boxes": list(boxes),
            "picture_crop_ids": shown, "pictured": bool(shown), "thumb": thumb,
            "thumb_crop_id": crops[0] if thumb else None, "story_action": action}


def test_linked_camera_groups_are_one_person_asked_once() -> None:
    people = [_p("sasha1", label="Sasha", crops=["k1", "k2"], pics=[("k1", "t1")], boxes=[{"n": 2}, {"n": 3}]),
              _p("sasha2", label="Sasha", crops=["k3"], pics=[("k3", "t3")], boxes=[{"n": 4}], copied=True),
              _p("anon", crops=["k4"], pics=[("k4", "t4")], boxes=[{"n": 5}])]
    roots = sp.link_roots([("sasha1", "x-outside"), ("x-outside", "sasha2")])   # joined through a group not in the story
    persons = sp.group_people(people, roots, {"k1": 10.0, "k3": 99.0}, [("sasha1", "sasha2")])
    assert [p["individual_ids"] for p in persons] == [["sasha1", "sasha2"], ["anon"]]
    sasha = persons[0]
    assert (sasha["person_key"], sasha["label"], sasha["label_copied"]) == ("sasha1", "Sasha", False)
    assert [c["crop_id"] for c in sasha["pictures"]] == ["k3", "k1"]          # the largest box first
    assert [b["n"] for b in sasha["boxes"]] == [2, 3, 4]
    assert [g["picture_crop_ids"] for g in sasha["groups"]] == [["k1"], ["k3"]]
    assert sasha["asked_ids"] == ["sasha1", "sasha2"]
    assert sasha["links"] == [{"a": "sasha1", "b": "sasha2"}]
    assert persons[1]["links"] == []


def test_two_typed_names_in_one_linked_group_stay_two_persons() -> None:
    people = [_p("a", label="Ann", crops=["c1"], pics=[("c1", "t1")]),
              _p("b", label="Bea", crops=["c2"], pics=[("c2", "t2")]),
              _p("c", crops=["c3"], pics=[("c3", "t3")])]
    roots = sp.link_roots([("a", "b"), ("b", "c")])
    persons = sp.group_people(people, roots)
    # No name of its own: joins the first person of its group, never a second name.
    assert [p["individual_ids"] for p in persons] == [["a", "c"], ["b"]]
    assert sp.group_people(people, roots)[0]["label"] == "Ann"


def test_a_grouped_person_shows_at_most_four_pictures_and_answers_each_group_only_on_its_shown_ones() -> None:
    """Orion's question 2: one answer on a linked person writes at most one row per camera group, at
    most four, and a group none of whose pictures were shown is not answered (never blind)."""
    people = [_p(f"g{i}", label="Sasha", crops=[f"c{i}"], pics=[(f"c{i}", f"t{i}")]) for i in range(6)]
    roots = sp.link_roots([(f"g{i}", f"g{i + 1}") for i in range(5)])
    (person,) = sp.group_people(people, roots)
    assert len(person["pictures"]) == sp.MAX_PICTURES == 4
    shown = {c["crop_id"] for c in person["pictures"]}
    assert {"c0", "c5"} <= shown                              # spread over the passage: first and last kept
    assert len(person["asked_ids"]) == 4
    for g in person["groups"]:
        assert set(g["picture_crop_ids"]) <= shown
        assert g["asked"] == bool(g["picture_crop_ids"])


def test_a_thumbnail_only_group_is_asked_on_its_thumbnail() -> None:
    people = [_p("a", label="Sasha", crops=["c1"], pics=[("c1", "t1")]),
              _p("b", crops=["c9"], thumb="e" * 64)]
    (person,) = sp.group_people(people, sp.link_roots([("a", "b")]))
    assert [g["picture_crop_ids"] for g in person["groups"]] == [["c1"], ["c9"]]
    assert {c.get("thumb") for c in person["pictures"]} == {None, "e" * 64}


def test_without_links_every_camera_group_is_its_own_person() -> None:
    people = [_p("a", crops=["c1"], pics=[("c1", "t1")]), _p("b", crops=["c2"])]
    persons = sp.group_people(people, {})
    assert [p["individual_ids"] for p in persons] == [["a"], ["b"]]
    assert persons[1]["pictured"] is False and persons[1]["asked_ids"] == []


def test_already_said_skips_a_repeat_but_never_a_change() -> None:
    """Orion's question 4: a second identical answer per (story, camera group) is not written."""
    assert sp.already_said({"action": "confirm", "name": "Sasha"}, "confirm", "sasha")
    assert sp.already_said({"action": "name", "name": "Sasha"}, "confirm", "Sasha")   # both are her who-is-this
    assert sp.already_said({"action": "unsure", "name": None}, "unsure", "")
    assert sp.already_said({"action": "someone_else", "name": "Cora"}, "someone_else", "cora")
    assert not sp.already_said({"action": "someone_else", "name": "Cora"}, "someone_else", "")
    assert not sp.already_said({"action": "confirm", "name": "Sasha"}, "name", "Sam")
    assert not sp.already_said({"action": "unsure"}, "someone_else", "")
    assert not sp.already_said(None, "unsure", "")


def test_person_answer_is_the_row_the_writer_applies() -> None:
    a = sp.person_answer(action="someone_else", name="Cora", story_id="f" * 32, picture_crop_ids=["c1", "c2"],
                         person_key="g0", group_size=3)
    assert a["kind"] == "vision_story_person" and a["answer"] is None
    assert a["detail"]["person_key"] == "g0" and a["detail"]["group_size"] == 3
    assert [(p["crop_id"], p["verdict"], p["name"]) for p in a["detail"]["pictures"]] == [
        ("c1", "different", "Cora"), ("c2", "different", "Cora")]
    u = sp.person_answer(action="unsure", name="", story_id="f" * 32, picture_crop_ids=["c1"])
    assert {p["verdict"] for p in u["detail"]["pictures"]} == {"unsure"} and "person_key" not in u["detail"]
    y = sp.person_answer(action="confirm", name="Sasha", story_id="f" * 32, picture_crop_ids=["c1"])
    assert (y["kind"], y["answer"], y["detail"]["pictures"][0]["verdict"]) == ("vision_individual", "Sasha", "same")
    with pytest.raises(ValueError):
        sp.person_answer(action="merge", name="", story_id="f" * 32, picture_crop_ids=[])


def test_not_same_is_the_existing_same_person_answer_no() -> None:
    n = sp.not_same_answer(a="zz", b="aa", story_id="f" * 32, person_key="aa")
    assert (n["kind"], n["source_ref"], n["answer"]) == ("vision_same_individual", "aa|zz", "different")
    assert n["detail"]["story_id"] == "f" * 32       # Juniper's own card: no ask statistic counts it


def test_connected_follows_links_transitively() -> None:
    assert sp.connected([("a", "b"), ("b", "c")], "a", "c")
    assert not sp.connected([("a", "b")], "a", "c")
