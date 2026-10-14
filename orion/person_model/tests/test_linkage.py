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
