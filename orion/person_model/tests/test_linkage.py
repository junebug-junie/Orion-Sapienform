import dataclasses

import pytest

from orion.person_model import linkage as L
from orion.person_model.tests.conftest import ANCHOR, NODES


@pytest.fixture
def cfg():
    return L.load_config()


def test_the_eleven_node_split_resolves_as_the_spec_says(cfg, ctx, candidates):
    got = {d.candidate_id: (d.relation, d.group) for d in L.run([ANCHOR], candidates, cfg, ctx)}
    want = {nid: (rel, grp) for nid, (_, _, _, rel, grp) in NODES.items()}
    assert got == want


def test_auto_merge_precision_is_one_on_the_fixture(cfg, ctx, candidates):
    auto = [d for d in L.run([ANCHOR], candidates, cfg, ctx) if d.relation == "same_as" and d.group == "auto_merge"]
    assert auto and all(NODES[d.candidate_id][3] == "same_as" for d in auto)


def test_rerun_is_identical(cfg, ctx, candidates):
    a = [d.model_dump_json() for d in L.run([ANCHOR], candidates, cfg, ctx)]
    b = [d.model_dump_json() for d in L.run([ANCHOR], list(reversed(candidates)), cfg, ctx)]
    assert a == b


def test_no_single_feature_can_force_a_merge(cfg):
    assert max(max(v.values()) for v in cfg.weights.values()) < cfg.upper
    assert max(cfg.weights["embedding_centered"].values()) < cfg.upper - cfg.lower
    bad = dataclasses.replace(cfg, upper=5.0)
    with pytest.raises(ValueError):
        L.validate_config(bad)


def test_embedding_alone_never_auto_merges(cfg):
    """A different name with a perfect embedding stays out of auto-merge."""
    ctx = L.LinkageContext(embedding_sim=lambda a, b: 1.0)
    d = L.decide(L.Anchor("a", "teddy"), L.Candidate("c", "ted", producer="x"), cfg, ctx)
    assert d.group != "auto_merge"
    no_emb = L.decide(L.Anchor("a", "teddy"), L.Candidate("c", "ted", producer="x"), cfg, L.LinkageContext())
    assert d.score - no_emb.score == pytest.approx(cfg.w("embedding_centered", "agree"))


def test_exact_label_without_corroboration_goes_to_review(cfg):
    """Exact text alone (extractor and embedding missing) is not enough to merge."""
    d = L.decide(L.Anchor("a", "addy", producer="p"), L.Candidate("c", "Addy", producer="q"), cfg, L.LinkageContext())
    assert d.group == "needs_your_call"


def test_cooccurring_names_are_forced_to_review(cfg):
    ctx = L.LinkageContext(is_person_name=lambda s: True, embedding_sim=lambda a, b: 0.9,
                           cooccurring=frozenset({("jay", "jaye")}))
    d = L.decide(L.Anchor("mention:jay", "Jay"), L.Candidate("mention:jaye", "Jaye", "mention_cluster"), cfg, ctx)
    assert d.group == "needs_your_call"


def test_repeated_name_split_card(cfg):
    out = L.repeated_name_splits({"jay": {"label": "Jay", "node_id": "mention:jay", "repeat_sources": ["p1"]},
                                  "rachel": {"label": "Rachel", "node_id": "mention:rachel", "repeat_sources": []}}, cfg)
    assert [(d.relation, d.group, d.anchor_label) for d in out] == [("distinct", "needs_your_call", "Jay")]


def test_stated_home_flags_other_place_home_alias(cfg):
    rows = [{"node_id": "ref-slc", "alias_norm": "home", "referent_kind": "place", "grounded_in": "chat_prompt:x"},
            {"node_id": "ref-ogden", "alias_norm": "home", "referent_kind": "place"},
            {"node_id": "ref-slc", "alias_norm": "slc", "referent_kind": "place"}]
    out = L.stated_home_conflicts(rows, person=ANCHOR, stated_home="Ogden, Utah",
                                  place_labels={"ref-slc": "slc", "ref-ogden": "ogden"}, cfg=cfg, stated_ref="answer 4")
    assert [(d.candidate_id, d.relation, d.group) for d in out] == [("ref-slc#alias:home", "not_home", "auto_reject")]


def test_best_per_candidate_prefers_the_strongest_claim(cfg, ctx):
    c = L.Candidate("c1", "juniper", "substrate_node", ("SubstrateNode", "Entity"), "topic_foundry_adapter")
    other = L.Anchor("referent-zzz", "orion", (), "memory.referents")
    ds = [L.decide(ANCHOR, c, cfg, ctx), L.decide(other, c, cfg, ctx)]
    assert L.best_per_candidate(ds)[0].anchor_id == ANCHOR.node_id


@pytest.mark.parametrize("label,want", [
    ("orion/harness/finalize.py", True), ("orion:vision:frames", True), ("orion_identity.yaml", True),
    ("orion_readonly user", True), ("orion-*", True), ("orion hub", False), ("Mary-Jane", False), ("juniper feld", False),
])
def test_code_identifier_shape_rule(label, want):
    assert L.is_code_identifier(label) is want


def test_exact_match_does_not_double_count_the_embedding(cfg):
    ctx = L.LinkageContext(is_person_name=lambda s: True, embedding_sim=lambda a, b: 1.0)
    d = L.decide(L.Anchor("a", "orion", producer="p"), L.Candidate("c", "Oríon", producer="q"), cfg, ctx)
    emb = next(f for f in d.features if f.name == "embedding_centered")
    assert emb.level == "implied_by_exact" and emb.weight == 0.0
    assert d.score == pytest.approx(6.0 + 2.0 + 0.5) and d.group == "auto_merge"


def test_name_plus_words_needs_to_have_been_written_as_a_name(cfg):
    """'claude motor' reads like a name when title-cased, but she never wrote it as one."""
    mentioned = {"juniper feld"}
    ctx = L.LinkageContext(is_person_name=lambda s: True, mentioned_as_person=lambda s: s.lower() in mentioned)
    a = L.Anchor("a", "juniper")
    assert L.decide(a, L.Candidate("c1", "juniper motor"), cfg, ctx).relation == "facet_of"
    full = L.decide(a, L.Candidate("c2", "juniper feld"), cfg, ctx)
    assert full.relation == "same_as" and full.group == "needs_your_call"


def test_extractor_readings_count_once(cfg):
    both = L.LinkageContext(is_person_name=lambda s: True, mentioned_as_person=lambda s: True)
    one = L.LinkageContext(is_person_name=lambda s: True)
    a, c = L.Anchor("a", "gaga", producer="p"), L.Candidate("c", "Gaga", producer="q")
    assert L.decide(a, c, cfg, both).score == L.decide(a, c, cfg, one).score


def test_exact_plus_extractor_from_the_same_producer_goes_to_review(cfg):
    """Review finding 1: two features (8.0) must not auto-merge; the cross-producer
    corroboration is what reaches the 8.5 line."""
    ctx = L.LinkageContext(is_person_name=lambda s: True)
    same = L.decide(L.Anchor("a", "addy", producer="p"), L.Candidate("c", "Addy", producer="p"), cfg, ctx)
    cross = L.decide(L.Anchor("a", "addy", producer="p"), L.Candidate("c", "Addy", producer="q"), cfg, ctx)
    assert same.score == pytest.approx(8.0) and same.group == "needs_your_call"
    assert cross.score == pytest.approx(8.5) and cross.group == "auto_merge"


def test_sharing_only_a_chat_message_is_counter_evidence(cfg):
    """Review finding 2: two names in one message are not evidence of one person."""
    a = L.Anchor("mention:jay", "Jay", (), "x", ("chat_prompt:p1",))
    msg = L.decide(a, L.Candidate("mention:jason", "Jason", "mention_cluster", (), "x", ("chat_prompt:p1",)), cfg, L.LinkageContext())
    mem = L.decide(a, L.Candidate("n", "Jason", "substrate_node", (), "y", ("episode_memory:m1",)), cfg, L.LinkageContext())
    f_msg = next(f for f in msg.features if f.name == "shared_evidence")
    assert f_msg.level == "same_message" and f_msg.weight < 0
    a2 = L.Anchor("r", "jay", (), "x", ("episode_memory:m1",))
    f_mem = next(f for f in L.decide(a2, L.Candidate("n", "Jason", producer="y", evidence_refs=("episode_memory:m1",)), cfg, L.LinkageContext()).features if f.name == "shared_evidence")
    assert f_mem.level == "agree" and mem is not None


def test_two_people_claiming_one_candidate_go_to_review(cfg):
    """Review finding 3: the tie is not broken silently."""
    ctx = L.LinkageContext(is_person_name=lambda s: True)
    c = L.Candidate("c", "Sam", producer="q")
    ds = [L.decide(L.Anchor("a1", "sam", producer="p"), c, cfg, ctx), L.decide(L.Anchor("a2", "sam", producer="p2"), c, cfg, ctx)]
    assert all(d.group == "auto_merge" for d in ds)
    (kept,) = L.best_per_candidate(ds)
    assert kept.group == "needs_your_call" and kept.rule == "multiple_anchor_claims"
    assert kept.evidence["claimed_by"] == ["sam (a1)", "sam (a2)"]


def test_home_needs_the_town_not_just_the_state(cfg):
    rows = [{"node_id": "ref-slc", "alias_norm": "home", "referent_kind": "place"}]
    out = L.stated_home_conflicts(rows, person=ANCHOR, stated_home="Ogden, Utah",
                                  place_labels={"ref-slc": "Salt Lake City, Utah"}, cfg=cfg, stated_ref="a4")
    assert len(out) == 1


def test_capitalized_dotted_names_are_not_code(cfg):
    assert not L.is_code_identifier("St.John")
    assert not L.is_code_identifier("Dr.Smith")
