"""One record per person: the link planner (orion/person_model/record.py)."""

from __future__ import annotations

from orion.person_model import record as pr
from orion.schemas.person_model import PersonRefV1

YOU = PersonRefV1(node_id="referent-you", display_name="Alex", role="interlocutor", pronouns="she/her")
SELF = PersonRefV1(node_id="referent-self", display_name="Orion", role="self", pronouns="they/them")


def c(iid, label, kind="person", ask=None):
    return pr.Cluster(iid, kind, label, ask)


def run(clusters, different=(), persons=(), links=None, people=(YOU, SELF)):
    return pr.plan(clusters, different, people, persons, dict(links or {}))


def test_two_clusters_one_name_are_one_person() -> None:
    plan = run([c("a", "Sam"), c("b", " sam ")])
    sam = [p for p in plan.new_persons if p.display_name == "Sam"]
    assert len(sam) == 1
    assert {(l.source_ref, l.person_id) for l in plan.add_links} == {("a", sam[0].person_id), ("b", sam[0].person_id)}


def test_a_camera_name_matching_people_yaml_links_to_that_referent() -> None:
    plan = run([c("a", "alex")])
    assert [(l.source_ref, l.person_id) for l in plan.add_links] == [("a", "referent-you")]
    assert {p.person_id for p in plan.new_persons} == {"referent-you", "referent-self"}


def test_different_verdict_keeps_same_named_people_apart() -> None:
    sam = pr.Person(pr.minted_person_id("sam"), "Sam")
    plan = run([c("a", "Sam"), c("b", "Sam")], different=[("a", "b")], persons=[sam],
               links={(pr.SOURCE_INDIVIDUAL, "a"): sam.person_id})
    assert plan.add_links == []
    assert [(x.individual_id, x.reason) for x in plan.conflicts] == [("b", "different_verdict")]


def test_a_new_different_verdict_unlinks_one_of_the_pair() -> None:
    sam = pr.Person(pr.minted_person_id("sam"), "Sam")
    links = {(pr.SOURCE_INDIVIDUAL, "a"): sam.person_id, (pr.SOURCE_INDIVIDUAL, "b"): sam.person_id}
    plan = run([c("a", "Sam"), c("b", "Sam")], different=[("a", "b")], persons=[sam], links=links)
    assert plan.revoke == [(pr.SOURCE_INDIVIDUAL, "a")]
    assert [x.individual_id for x in plan.conflicts] == ["a"]


def test_rename_moves_the_link_and_clearing_revokes_it() -> None:
    sam = pr.Person(pr.minted_person_id("sam"), "Sam")
    links = {(pr.SOURCE_INDIVIDUAL, "a"): sam.person_id, (pr.SOURCE_INDIVIDUAL, "b"): sam.person_id}
    plan = run([c("a", "Alex", ask="ask-2")], persons=[sam], links=links)
    assert sorted(plan.revoke) == [(pr.SOURCE_INDIVIDUAL, "a"), (pr.SOURCE_INDIVIDUAL, "b")]
    assert [(l.source_ref, l.person_id, l.evidence_ref) for l in plan.add_links] == [("a", "referent-you", "ask-2")]


def test_unchanged_input_plans_nothing() -> None:
    first = run([c("a", "Sam"), c("bike", "Sam", kind="bicycle")])
    persons = [pr.Person(p.person_id, p.display_name, p.role) for p in first.new_persons]
    links = {(l.source_kind, l.source_ref): l.person_id for l in first.add_links}
    again = run([c("a", "Sam"), c("bike", "Sam", kind="bicycle")], persons=persons, links=links)
    assert again.empty and again.conflicts == []


def test_named_things_link_as_belongings_and_junk_names_are_ignored() -> None:
    plan = run([c("a", "Sam"), c("bike", "sam", kind="bicycle"), c("car", "no", kind="vehicle")])
    by_ref = {l.source_ref: l for l in plan.add_links}
    assert by_ref["bike"].source_kind == pr.SOURCE_BELONGING
    assert by_ref["bike"].person_id == by_ref["a"].person_id
    assert "car" not in by_ref
    assert not any(p.display_name == "no" for p in plan.new_persons)


def test_a_vehicle_name_never_mints_a_person() -> None:
    plan = run([c("car", "Taylor", kind="vehicle")])
    assert not any(p.display_name == "Taylor" for p in plan.new_persons)
    assert plan.add_links == []


def test_minted_ids_are_stable() -> None:
    assert pr.minted_person_id("sam") == pr.minted_person_id("sam") != pr.minted_person_id("alex")


def test_a_people_yaml_row_added_later_takes_over_a_camera_minted_person() -> None:
    """Review 2026-10-22: a yaml row for a name the camera had already minted split the person,
    left new clusters in conflict forever, and revoked the bike."""
    minted = pr.Person(pr.minted_person_id("sam"), "Sam")
    links = {(pr.SOURCE_INDIVIDUAL, "a"): minted.person_id, (pr.SOURCE_BELONGING, "bike"): minted.person_id}
    sam_ref = PersonRefV1(node_id="referent-sam", display_name="Sam")
    plan = run([c("a", "Sam"), c("b", "Sam"), c("bike", "Sam", kind="bicycle")], persons=[minted], links=links,
               people=(YOU, SELF, sam_ref))
    assert plan.conflicts == []
    assert sorted(plan.revoke) == [(pr.SOURCE_BELONGING, "bike"), (pr.SOURCE_INDIVIDUAL, "a")]
    assert sorted((l.source_ref, l.person_id) for l in plan.add_links) == [
        ("a", "referent-sam"), ("b", "referent-sam"), ("bike", "referent-sam")]


def test_a_name_she_said_is_the_same_person_moves_to_that_person() -> None:
    """2026-10-29, live: 23 clusters named with a first name and one with the full name
    became two people, and she was asked "How do you know <first name>?" after answering
    for the full name. "Same person" (person.merged_into) joins them; nothing is guessed."""
    full = pr.Person(pr.minted_person_id("sam lee"), "Sam Lee")
    first = pr.Person(pr.minted_person_id("sam"), "Sam", merged_into=full.person_id)
    links = {(pr.SOURCE_INDIVIDUAL, "a"): first.person_id, (pr.SOURCE_INDIVIDUAL, "b"): full.person_id,
             (pr.SOURCE_BELONGING, "bike"): first.person_id}
    plan = run([c("a", "Sam"), c("b", "Sam Lee"), c("bike", "sam", kind="bicycle"), c("new", "SAM")],
               persons=[full, first], links=links)
    assert plan.conflicts == [] and plan.new_persons[2:] == []
    assert sorted(plan.revoke) == [(pr.SOURCE_BELONGING, "bike"), (pr.SOURCE_INDIVIDUAL, "a")]
    assert sorted((l.source_ref, l.person_id) for l in plan.add_links) == [
        ("a", full.person_id), ("bike", full.person_id), ("new", full.person_id)]


def test_without_same_person_a_first_name_stays_its_own_person() -> None:
    full = pr.Person(pr.minted_person_id("sam lee"), "Sam Lee")
    plan = run([c("a", "Sam"), c("b", "Sam Lee")], persons=[full], links={(pr.SOURCE_INDIVIDUAL, "b"): full.person_id})
    assert [p.display_name for p in plan.new_persons if p.display_name.startswith("Sam")] == ["Sam"]


def test_a_circular_same_person_chain_does_not_hang() -> None:
    a = pr.Person("p-a", "Sam", merged_into="p-b")
    b = pr.Person("p-b", "Sam B", merged_into="p-a")
    plan = run([c("x", "Sam")], persons=[a, b])
    assert len(plan.add_links) == 1
