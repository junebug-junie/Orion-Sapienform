"""Referent resolution, alias admission and source_cooccurrence_v1. Pure, DB-free.

All people, places and phrases here are synthetic (the repository is public).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from orion.memory.referents.aliases import alias_in_text, normalize_alias, slug_text
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
from orion.schemas.memory_episode import DistillReferentV1

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
ON = ReferentPolicy()
P, D = "proper_name", "descriptor"
HECATE_QUOTE = ("p1", "I got us an Inspur NF5288M5 AGX-2 GPU that holds 8x smx2 gpus. We'll call it Hecate")


def resolve(key, aliases, index, quotes=(HECATE_QUOTE,), policy=ON, last_use=NOW, now=NOW, key_kind=P):
    cands = candidate_aliases(key, key_kind, aliases, list(quotes))
    return resolve_referent(key, cands, index, now=now, last_use=last_use, policy=policy)


def states(res):
    return {r.alias_norm: (r.alias_class, r.promotion_state, r.admitted_by) for r in res.new_rows}


# ── text rules ──────────────────────────────────────────────────────────────


def test_one_normalization_for_keys_and_aliases():
    assert normalize_alias("  “Fairview,  Ohio”. ") == "fairview ohio"
    assert slug_text("place:fairview-ohio") == "fairview ohio"
    assert alias_in_text("hecate", "We'll call it Hecate.")
    assert not alias_in_text("hecate", "hecatessen")  # word-bounded, no substring identity


def test_unjudged_names_are_descriptors():
    """Answers from before prompt v4 carry bare strings and no alias_kind."""
    ref = DistillReferentV1.model_validate({"key": "person:morgan", "aliases": ["boss", {"text": "Morgan",
                                                                                         "alias_kind": "proper_name"}]})
    assert ref.alias_kind == D
    assert [(a.text, a.alias_kind) for a in ref.aliases] == [("boss", D), ("Morgan", P)]
    assert DistillReferentV1.model_validate({"key": "x:y", "alias_kind": "weird"}).alias_kind == D


# ── alias_grounding_v1 ──────────────────────────────────────────────────────


def test_grounded_names_are_usable_and_ungrounded_ones_are_not():
    res = resolve("project:hecate", [("Inspur NF5288M5", P), ("AGX-2 GPU", P), ("8x smx2 gpus", P),
                                     ("the new server", D)], AliasIndex([]))
    got = states(res)
    assert res.via == "minted" and res.node_id == node_id_for_key("project:hecate")
    assert got["project:hecate"] == ("key", "provisional", "minted")
    for name in ("hecate", "inspur nf5288m5", "agx 2 gpu", "8x smx2 gpus"):
        assert got[name] == ("name", "provisional", "alias_grounding_v1"), name
    assert got["the new server"] == ("descriptor", "proposed", "ungrounded")


def test_the_grounding_kill_switch_flips_grounded_names_to_proposed():
    res = resolve("project:hecate", [("Inspur NF5288M5", P)], AliasIndex([]),
                  policy=ReferentPolicy(grounding_auto_accept=False))
    assert states(res)["inspur nf5288m5"] == ("name", "proposed", "alias_grounding_v1_disabled")


def test_a_grounded_proper_name_resolves_a_later_key_to_the_same_node():
    index = AliasIndex([])
    first = resolve("project:hecate", [("Inspur NF5288M5", P)], index)
    later = resolve("project:inspur-nf5288m5", [], index)
    assert (later.via, later.node_id) == ("name_match", first.node_id)


def test_an_ungrounded_name_never_resolves():
    index = AliasIndex([])
    resolve("project:hecate", [("the new server", D)], index)
    later = resolve("project:the-new-server", [], index, quotes=(("q", "the new server is loud"),))
    assert later.node_id != node_id_for_key("project:hecate")


# ── review repro 1: a descriptor never decides identity ─────────────────────


def test_the_same_descriptor_for_two_people_is_a_question_not_a_merge():
    index = AliasIndex([])
    morgan = resolve("person:morgan", [("boss", D)], index, quotes=[("p1", "boss morgan wants the deck")])
    taylor = resolve("person:taylor", [("boss", D)], index, quotes=[("p2", "boss taylor starts monday")])
    assert taylor.via == "minted" and taylor.node_id != morgan.node_id
    assert states(taylor)["boss"][1:] == ("proposed", "collision")
    (q,) = taylor.questions
    assert (q.scope, q.answer_via, q.reason) == ("juniper", "conversation", "alias_collision")


def test_a_misjudged_proper_name_still_cannot_override_the_key_s_own_name():
    """Even if the distiller wrongly calls "boss" a proper name for both people, Taylor's own
    grounded name says it is someone else: that is a question, never a merge."""
    index = AliasIndex([])
    morgan = resolve("person:morgan", [("boss", P)], index, quotes=[("p1", "boss morgan wants the deck")])
    taylor = resolve("person:taylor", [("boss", P)], index, quotes=[("p2", "boss taylor starts monday")])
    assert taylor.via == "ambiguous" and taylor.node_id != morgan.node_id
    assert states(taylor)[f"person:taylor"][1] == "proposed"
    assert taylor.questions[0].reason == "referent_identity"


# ── review repro 2: a descriptor-derived key never swallows a person ────────


def test_a_descriptor_key_never_resolves_another_key():
    index = AliasIndex([])
    cousin = resolve("person:my-cousin", [], index, quotes=[("p", "my cousin called")], key_kind=D)
    robin = resolve("person:robin", [("my cousin", D)], index, quotes=[("p", "my cousin robin called")])
    assert robin.node_id != cousin.node_id and robin.via == "minted"
    assert states(cousin)["my cousin"][0] == "descriptor"
    assert states(robin)["my cousin"][1:] == ("proposed", "collision")
    # and the other direction: a later descriptor-named key never lands on Robin
    again = resolve("person:my-cousin-robin", [("Robin", D)], index, quotes=[("p", "my cousin robin")], key_kind=D)
    assert again.node_id != robin.node_id


# ── relative names: live at once, 90-day lapse, refresh, no silent second node ──


def test_a_descriptor_lapses_90_days_after_last_use_and_is_refreshed_by_reuse():
    index = AliasIndex([])
    quote = ("p2", "my boss Morgan is coming to the retreat")
    res = resolve("person:morgan", [("my boss", D)], index, quotes=[quote])
    assert states(res)["my boss"][:2] == ("descriptor", "provisional")
    assert index.rows[(res.node_id, "my boss")].valid_until == NOW + timedelta(days=90)
    assert index.live_rows("my boss", NOW + timedelta(days=89))
    assert not index.live_rows("my boss", NOW + timedelta(days=91))
    again = resolve("person:morgan", [("my boss", D)], index, quotes=[quote], last_use=NOW + timedelta(days=60))
    assert again.refreshed and index.rows[(res.node_id, "my boss")].valid_until == NOW + timedelta(days=150)
    # After it lapses, a new boss gets the name without a question; Morgan keeps her node.
    later = NOW + timedelta(days=200)
    taylor = resolve("person:taylor", [("my boss", D)], index, quotes=[("p9", "my boss taylor")],
                     now=later, last_use=later)
    assert states(taylor)["my boss"][1:] == ("provisional", "alias_grounding_v1") and not taylor.questions
    assert resolve("person:morgan", [], index, quotes=(), now=later).node_id == res.node_id


def test_a_place_name_with_punctuation_never_splits_into_a_second_node():
    index = AliasIndex([])
    first = resolve("place:fairview", [("Fairview, Ohio", P)], index, quotes=[("p", "we drove to fairview, ohio")])
    later = NOW + timedelta(days=100)  # proper names do not lapse
    second = resolve("place:fairview-ohio", [("Fairview, Ohio", P)], index, quotes=[("q", "back in fairview, ohio")],
                     now=later, last_use=later)
    assert (second.via, second.node_id) == ("name_match", first.node_id)


# ── ambiguity ───────────────────────────────────────────────────────────────


def test_a_proper_name_on_two_nodes_mints_a_proposed_node_and_asks():
    index = AliasIndex([])
    a = resolve("person:sam-a", [("Sam", P)], index, quotes=[("p4", "Sam called")], key_kind=D)
    index.add(replace(index.rows[(a.node_id, "sam")], node_id="referent-other"))
    index.add(replace(index.node_key_row(a.node_id), node_id="referent-other", alias_norm="person:sam-c"))
    res = resolve("person:sam-b", [("Sam", P)], index, quotes=[("p5", "Sam is visiting")], key_kind=D)
    assert res.via == "ambiguous" and states(res)["person:sam-b"] == ("key", "proposed", "ambiguous")
    assert {tuple(sorted(q.node_ids)) for q in res.questions} == {
        tuple(sorted((res.node_id, a.node_id))), tuple(sorted((res.node_id, "referent-other")))}
    assert states(res)["sam"][1] == "proposed"


def test_the_same_proper_name_on_another_kind_is_ambiguous():
    index = AliasIndex([])
    resolve("place:circe", [], index, quotes=[("p6", "drive to circe")])
    res = resolve("project:circe", [], index, quotes=[("p7", "circe is rebooting")])
    assert res.via == "ambiguous"
    assert res.questions[0].scope == "self"


# ── source_cooccurrence_v1 ──────────────────────────────────────────────────


def ep(key, *names, state="provisional"):
    return EndpointV1(key=key, node_id=node_id_for_key(key), node_kind="entity", state=state, names=names)


QUOTE = "Quill and Morgan from the spring retreat went to the Lantern Pub"
SELF = {node_id_for_key("person:juniper"), node_id_for_key("person:orion")}


def claims(endpoints, *, accept=True, quotes=(QUOTE,), decided=None, exclude=SELF):
    return cooccurrence_claims(memory_id="m1", endpoints=endpoints, prompt_quotes=list(quotes),
                               decided_targets=set() if decided is None else decided, accept=accept,
                               recorded_at=NOW, exclude_node_ids=exclude)


def test_things_named_in_one_quote_become_accepted_co_occurrences():
    got = claims([ep("person:quill", "quill"), ep("event:spring-retreat", "retreat", "spring retreat"),
                  ep("person:juniper", "juniper")])
    assert len(got) == 1
    proposal, decision = got[0].proposal, got[0].decision
    assert proposal.predicate == "co_occurs_with" and decision.policy == POLICY
    assert decision.resulting_state == "provisional" and decision.expected_prior_revision == 0


def test_the_cooccurrence_kill_switch_leaves_proposals_only():
    got = claims([ep("person:quill", "quill"), ep("place:lantern-pub", "lantern pub")], accept=False)
    assert len(got) == 1 and got[0].decision is None


def test_an_unaccepted_endpoint_keeps_the_claim_proposed():
    got = claims([ep("person:quill", "quill"), ep("place:lantern-pub", "lantern pub", state="proposed")])
    assert got[0].decision is None


def test_more_than_six_claims_from_one_memory_are_all_held_for_review():
    names = ["quill", "morgan", "spring", "retreat", "lantern"]
    eps = [ep(f"person:p{i}", n) for i, n in enumerate(names)]  # 5 named -> 10 pairs
    got = claims(eps)
    assert len(got) == 10 > MAX_ACCEPTED_PER_MEMORY
    assert all(c.decision is None for c in got)
    within = claims(eps[:4])  # 4 named -> 6 pairs
    assert len(within) == 6 and all(c.decision is not None for c in within)


def test_a_second_memory_adds_a_proposal_not_a_second_decision():
    decided: set[str] = set()
    eps = [ep("person:quill", "quill"), ep("place:lantern-pub", "lantern pub")]
    first = claims(eps, decided=decided)
    second = cooccurrence_claims(memory_id="m2", endpoints=eps, prompt_quotes=[QUOTE], decided_targets=decided,
                                 accept=True, recorded_at=NOW)
    assert first[0].decision is not None and second[0].decision is None
    assert first[0].proposal.target_id == second[0].proposal.target_id


def test_names_in_different_quotes_do_not_co_occur():
    got = claims([ep("person:quill", "quill"), ep("place:lantern-pub", "lantern pub")],
                 quotes=("Quill called", "we went to the Lantern Pub"))
    assert got == []


def test_juniper_and_orion_are_excluded_by_node_id_not_key():
    """Review LOW 9: a second key resolved onto Juniper's node (e.g. by her proper name) is
    still Juniper."""
    juniper_node = node_id_for_key("person:juniper")
    alias_of_juniper = EndpointV1(key="person:june", node_id=juniper_node, node_kind="entity",
                                  state="provisional", names=("june",))
    assert claims([alias_of_juniper, ep("person:quill", "quill")], quotes=("june and quill",)) == []
