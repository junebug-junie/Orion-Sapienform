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
