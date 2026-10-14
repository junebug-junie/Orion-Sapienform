import copy

from orion.person_model import linkage as L
from orion.person_model.merge_log import apply_ops, build_entry, merge_id, unmerge_ops
from orion.person_model.tests.conftest import ANCHOR


class MemStore:
    def __init__(self):
        self.props = {("tf-ent-ca5c", "label"): "juniper"}
        self.edges = set()
        self.aliases = {("ref-slc", "home"): {"alias_text": "home", "valid_until": None}}

    def snapshot(self):
        return copy.deepcopy((self.props, sorted(self.edges), self.aliases))

    def get_prop(self, n, p):
        return self.props.get((n, p))

    def set_prop(self, n, p, v):
        if v is None:
            self.props.pop((n, p), None)
        else:
            self.props[(n, p)] = v

    def has_edge(self, s, d, r):
        return (s, d, r) in self.edges

    def add_edge(self, s, d, r):
        self.edges.add((s, d, r))

    def remove_edge(self, s, d, r):
        self.edges.discard((s, d, r))

    def get_alias(self, n, a):
        return copy.deepcopy(self.aliases.get((n, a)))

    def put_alias(self, n, a, row):
        if row is None:
            self.aliases.pop((n, a), None)
        else:
            self.aliases[(n, a)] = copy.deepcopy(row)


def _decisions(ctx, candidates):
    cfg = L.load_config()
    ds = L.run([ANCHOR], candidates, cfg, ctx)
    ds += L.stated_home_conflicts([{"node_id": "ref-slc", "alias_norm": "home", "referent_kind": "place"}], person=ANCHOR,
                                  stated_home="Ogden", place_labels={"ref-slc": "slc"}, cfg=cfg, stated_ref="a4")
    return [d for d in ds if d.relation in ("same_as", "facet_of", "junk", "not_home")]


def test_apply_then_unmerge_restores_the_store_exactly(ctx, candidates):
    store = MemStore()
    before = store.snapshot()
    entries = []
    for d in _decisions(ctx, candidates):
        e = build_entry(d, store, decided_by="test", decision="auto", now_iso="2026-10-14T00:00:00+00:00")
        apply_ops(store, e.ops)
        entries.append(e)
    assert store.snapshot() != before
    assert store.get_prop("tf-ent-ca5c", "merged_into") == ANCHOR.node_id
    assert store.get_alias("ref-slc", "home")["valid_until"] == "2026-10-14T00:00:00+00:00"
    for e in reversed(entries):
        apply_ops(store, unmerge_ops(e))
    assert store.snapshot() == before


def test_merge_id_is_idempotent_per_version():
    assert merge_id("a", "b", "v1") == merge_id("a", "b", "v1") != merge_id("a", "b", "v2")


def test_unmerge_keeps_an_edge_that_existed_before(ctx, candidates):
    store = MemStore()
    d = next(d for d in _decisions(ctx, candidates) if d.relation == "facet_of")
    store.add_edge(d.candidate_id, d.anchor_id, "about")
    e = build_entry(d, store, decided_by="t", decision="auto", now_iso="x")
    apply_ops(store, e.ops)
    apply_ops(store, unmerge_ops(e))
    assert store.has_edge(d.candidate_id, d.anchor_id, "about")
