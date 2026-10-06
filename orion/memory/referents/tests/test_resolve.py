"""Referent resolution, alias admission and source_cooccurrence_v1. Pure, DB-free.

Names and quotes are the live 2026-10-06 shapes (spec "Aliases the writer emitted"),
reduced to what each rule needs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.memory.referents.aliases import (
    TokenFrequency,
    alias_class,
    alias_in_text,
    normalize_alias,
    slug_text,
)
from orion.memory.referents.cooccurrence import (
    MAX_ACCEPTED_PER_MEMORY,
    POLICY,
    EndpointV1,
    cooccurrence_claims,
)
from orion.memory.referents.resolve import (
    AliasIndex,
    ReferentPolicy,
    candidate_aliases,
    node_id_for_key,
    resolve_referent,
)

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
ON = ReferentPolicy()
# Live first-token frequencies in Juniper's 576 prompts (2026-10-06).
FREQ = {t: TokenFrequency(n, 576) for t, n in {
    "my": 46, "the": 155, "a": 133, "camera": 7, "inspur": 1, "agx-2": 1, "8x": 1, "hecate": 1,
    "jackalope": 1, "joker": 2, "offsite": 2, "vincent": 1, "rachel": 1, "boss": 2, "circe": 1,
}.items()}
HECATE_QUOTE = ("p1", "I got us an Inspur NF5288M5 AGX-2 GPU that holds 8x smx2 gpus. We'll call it Hecate")


def resolve(key, aliases, index, quotes=(HECATE_QUOTE,), policy=ON, last_use=NOW):
    cands = candidate_aliases(key, aliases, list(quotes), FREQ)
    return resolve_referent(key, cands, index, now=NOW, last_use=last_use, policy=policy)


def states(res):
    return {r.alias_norm: (r.alias_class, r.promotion_state, r.admitted_by) for r in res.new_rows}


# ── alias text rules ────────────────────────────────────────────────────────


def test_normalization_and_word_bounded_match():
    assert normalize_alias("  “Jackalope  Bar”. ") == "jackalope  bar".replace("  ", " ")
    assert slug_text("project:orion-camera") == "orion camera"
    assert alias_in_text("hecate", "We'll call it Hecate.")
    assert not alias_in_text("hecate", "hecatessen")  # no substring identity


def test_a_thing_s_own_key_name_is_a_proper_name_however_often_juniper_says_it():
    freq = {**FREQ, "circe": TokenFrequency(40, 576)}  # live: Juniper talks about Circe a lot
    cands = {c.norm: c.cls for c in candidate_aliases("project:circe", ["circe box"], [], freq)}
    assert cands == {"circe": "name", "circe box": "descriptor"}


def test_descriptor_is_decided_by_frequency_in_juniper_s_prompts_not_a_word_list():
    assert alias_class("my boss", FREQ["my"]) == "descriptor"
    assert alias_class("camera", FREQ["camera"]) == "descriptor"
    assert alias_class("inspur nf5288m5", FREQ["inspur"]) == "name"
    assert alias_class("anything", None) == "name"  # no corpus evidence -> no expiry


# ── alias_grounding_v1 ──────────────────────────────────────────────────────


def test_grounded_names_are_usable_and_ungrounded_ones_are_not():
    index = AliasIndex([])
    res = resolve("project:hecate", ["Inspur NF5288M5", "AGX-2 GPU", "8x smx2 gpus", "the new server"], index)
    got = states(res)
    assert res.via == "minted" and res.node_id == node_id_for_key("project:hecate")
    assert got["project:hecate"] == ("key", "provisional", "minted")
    for name in ("hecate", "inspur nf5288m5", "agx-2 gpu", "8x smx2 gpus"):
        assert got[name] == ("name", "provisional", "alias_grounding_v1"), name
    # Orion's own phrase, never in Juniper's words: kept, but never resolves.
    assert got["the new server"][1:] == ("proposed", "ungrounded")


def test_the_grounding_kill_switch_flips_grounded_names_to_proposed():
    res = resolve("project:hecate", ["Inspur NF5288M5"], AliasIndex([]),
                  policy=ReferentPolicy(grounding_auto_accept=False))
    assert states(res)["inspur nf5288m5"] == ("name", "proposed", "alias_grounding_v1_disabled")


def test_a_grounded_name_resolves_a_later_key_to_the_same_node():
    index = AliasIndex([])
    first = resolve("project:hecate", ["Inspur NF5288M5"], index)
    later = resolve("project:inspur-nf5288m5", [], index)
    assert (later.via, later.node_id) == ("name_match", first.node_id)


def test_an_ungrounded_name_never_resolves():
    index = AliasIndex([])
    resolve("project:hecate", ["the new server"], index)
    later = resolve("project:the-new-server", [], index, quotes=())
    assert later.via == "minted" and later.node_id != node_id_for_key("project:hecate")


# ── relative names: live as soon as said, 90-day expiry, refresh on reuse ──


def test_a_relative_name_resolves_now_and_lapses_90_days_after_last_use():
    index = AliasIndex([])
    quote = ("p2", "my boss Rachel is coming to the offsite")
    res = resolve("person:rachel", ["my boss"], index, quotes=[quote])
    assert states(res)["my boss"][:2] == ("descriptor", "provisional")
    row = index.rows[(res.node_id, "my boss")]
    assert row.valid_until == NOW + timedelta(days=90)
    assert index.live_names("my boss", NOW + timedelta(days=89))
    assert not index.live_names("my boss", NOW + timedelta(days=91))
    # Juniper says it again 60 days later: it lives 90 days past that use.
    again = resolve("person:rachel", ["my boss"], index, quotes=[quote], last_use=NOW + timedelta(days=60))
    assert again.refreshed and index.rows[(res.node_id, "my boss")].valid_until == NOW + timedelta(days=150)


def test_a_collision_on_a_relative_name_becomes_a_question_never_a_merge():
    index = AliasIndex([])
    rachel = resolve("person:rachel", ["my boss"], index, quotes=[("p2", "my boss Rachel")])
    dana = resolve("person:dana", ["my boss"], index, quotes=[("p3", "my boss Dana starts monday")])
    # Dana's own key is unambiguous: she is a new, usable node. Only the relative name collides.
    assert dana.via == "minted" and dana.node_id != rachel.node_id
    assert states(dana)["my boss"][1:] == ("proposed", "collision")
    (q,) = dana.questions
    assert (q.scope, q.answer_via, q.reason) == ("juniper", "conversation", "alias_collision")
    assert set(q.node_ids) == {rachel.node_id, dana.node_id}
    # Once Rachel's "my boss" has lapsed (90 days without use), it no longer collides.
    later = AliasIndex(index.rows.values())
    sam = resolve_referent("person:sam", candidate_aliases("person:sam", ["my boss"], [("p9", "my boss Sam")], FREQ),
                           later, now=NOW + timedelta(days=91), last_use=NOW + timedelta(days=91), policy=ON)
    assert states(sam)["my boss"][1:] == ("provisional", "alias_grounding_v1") and not sam.questions


# ── identity: refinement, ambiguity ─────────────────────────────────────────


def test_kind_refinement_files_the_same_thing_under_a_second_key():
    index = AliasIndex([])
    project = resolve("project:orion-hub", [], index, quotes=())
    service = resolve("service:orion-hub", [], index, quotes=())
    assert (service.via, service.node_id) == ("kind_refined", project.node_id)
    assert index.key_row("service:orion-hub").node_id == project.node_id


def test_a_name_on_two_nodes_mints_a_proposed_node_and_asks():
    index = AliasIndex([])
    a = resolve("person:sam-a", ["Sam"], index, quotes=[("p4", "Sam called")])
    b_key = "person:sam-b"
    # make "sam" live on a second node too (as Juniper confirming a second Sam would)
    from dataclasses import replace
    index.add(replace(index.rows[(a.node_id, "sam")], node_id="referent-other"))
    index.add(replace(index.node_key_row(a.node_id), node_id="referent-other", alias_norm="person:sam-c"))
    res = resolve(b_key, ["Sam"], index, quotes=[("p5", "Sam is visiting")])
    assert res.via == "ambiguous" and states(res)[b_key] == ("key", "proposed", "ambiguous")
    assert {tuple(sorted(q.node_ids)) for q in res.questions} == {
        tuple(sorted((res.node_id, a.node_id))), tuple(sorted((res.node_id, "referent-other")))}
    # Grounded aliases of an unanswered node are not usable either.
    assert states(res)["sam"][1] == "proposed"


def test_a_name_on_an_incompatible_kind_is_ambiguous():
    index = AliasIndex([])
    resolve("place:circe", [], index, quotes=[("p6", "drive to circe")])
    res = resolve("project:circe", [], index, quotes=[("p7", "circe is rebooting")])
    assert res.via == "ambiguous"
    assert res.questions[0].scope == "self"  # a project question Orion can investigate


# ── source_cooccurrence_v1 ──────────────────────────────────────────────────


def ep(key, *names, state="provisional"):
    return EndpointV1(key=key, node_id=node_id_for_key(key), node_kind="entity", state=state, names=names)


QUOTE = "Vincent and Rachel from the Austin offsite went to the Jackalope bar"


def claims(endpoints, *, accept=True, quotes=(QUOTE,), decided=None):
    return cooccurrence_claims(memory_id="m1", endpoints=endpoints, prompt_quotes=list(quotes),
                               decided_targets=set() if decided is None else decided, accept=accept,
                               recorded_at=NOW)


def test_things_named_in_one_quote_become_accepted_co_occurrences():
    got = claims([ep("person:vincent", "vincent"), ep("event:austin-offsite", "offsite", "austin offsite"),
                  ep("person:juniper", "juniper")])
    assert len(got) == 1
    proposal, decision = got[0].proposal, got[0].decision
    assert proposal.predicate == "co_occurs_with" and decision.policy == POLICY
    assert decision.resulting_state == "provisional" and decision.expected_prior_revision == 0


def test_the_cooccurrence_kill_switch_leaves_proposals_only():
    got = claims([ep("person:vincent", "vincent"), ep("place:jackalope-bar", "jackalope bar")], accept=False)
    assert len(got) == 1 and got[0].decision is None


def test_an_unaccepted_endpoint_keeps_the_claim_proposed():
    got = claims([ep("person:vincent", "vincent"), ep("place:jackalope-bar", "jackalope bar", state="proposed")])
    assert got[0].decision is None


def test_more_than_six_claims_from_one_memory_are_all_held_for_review():
    names = ["vincent", "rachel", "austin", "offsite", "jackalope"]
    eps = [ep(f"person:p{i}", n) for i, n in enumerate(names)]  # 5 named -> 10 pairs
    got = claims(eps)
    assert len(got) == 10 > MAX_ACCEPTED_PER_MEMORY
    assert all(c.decision is None for c in got)
    within = claims(eps[:4])  # 4 named -> 6 pairs
    assert len(within) == 6 and all(c.decision is not None for c in within)


def test_a_second_memory_adds_a_proposal_not_a_second_decision():
    decided: set[str] = set()
    eps = [ep("person:vincent", "vincent"), ep("place:jackalope-bar", "jackalope bar")]
    first = claims(eps, decided=decided)
    second = cooccurrence_claims(memory_id="m2", endpoints=eps, prompt_quotes=[QUOTE], decided_targets=decided,
                                 accept=True, recorded_at=NOW)
    assert first[0].decision is not None and second[0].decision is None
    assert first[0].proposal.target_id == second[0].proposal.target_id


def test_names_in_different_quotes_do_not_co_occur():
    got = claims([ep("person:vincent", "vincent"), ep("place:jackalope-bar", "jackalope bar")],
                 quotes=("Vincent called", "we went to the Jackalope bar"))
    assert got == []


@pytest.mark.parametrize("excluded", ["person:juniper", "person:orion"])
def test_juniper_and_orion_never_co_occur(excluded):
    assert claims([ep(excluded, "vincent"), ep("person:vincent", "vincent")]) == []
