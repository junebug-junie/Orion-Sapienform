import copy

from orion.person_model import linkage as L
from orion.person_model.merge_log import apply_ops, build_entry, merge_id, unmerge_ops
from orion.person_model.tests.conftest import ANCHOR


class MemStore:
    def __init__(self):
        self.props = {("tf-ent-ca5c", "label"): "juniper"}
        self.edges = set()
        self.aliases = {("ref-shelby", "home"): {"alias_text": "home", "valid_until": None}}

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
    ds += L.stated_home_conflicts([{"node_id": "ref-shelby", "alias_norm": "home", "referent_kind": "place"}], person=ANCHOR,
                                  stated_home="Springfield", place_labels={"ref-shelby": "shelby"}, cfg=cfg, stated_ref="a4")
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
    assert store.get_alias("ref-shelby", "home")["valid_until"] == "2026-10-14T00:00:00+00:00"
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


def test_existing_survivor_alias_is_not_overwritten(ctx, candidates):
    store = MemStore()
    store.put_alias(ANCHOR.node_id, "juniper", {"alias_text": "juniper", "admitted_by": "episode_writer"})
    d = next(d for d in _decisions(ctx, candidates) if d.relation == "same_as" and d.candidate_label == "Juniper")
    e = build_entry(d, store, decided_by="t", decision="auto", now_iso="x")
    assert not any(op["op"] == "put_alias" for op in e.ops)
    apply_ops(store, e.ops)
    assert store.get_alias(ANCHOR.node_id, "juniper")["admitted_by"] == "episode_writer"


def _two_juniper_merges(store):
    """Two graph nodes ('Juniper', 'juniper') both need the same survivor alias."""
    cfg = L.load_config()
    from orion.person_model.merge_log import plan_entries

    ctx = L.LinkageContext(is_person_name=lambda s: True)
    anchor = L.Anchor("ref-j", "jun", (), "memory.referents")
    ds = [L.decide(anchor, L.Candidate(n, lab, producer="tf"), cfg, ctx) for n, lab in (("n1", "Juniper"), ("n2", "juniper"))]
    ds = [d.model_copy(update={"relation": "same_as", "group": "auto_merge"}) for d in ds]
    entries, conflicts = plan_entries(ds, store, now_iso="x")
    assert not conflicts
    return entries


def test_undo_a_middle_entry_alone_keeps_what_the_other_needs():
    """Review finding 9: entries can be undone one at a time, in either order."""
    for order in ((0, 1), (1, 0)):
        store = MemStore()
        before = store.snapshot()
        entries = _two_juniper_merges(store)
        assert [op["op"] for op in entries[1].ops].count("claim_alias") == 1
        for e in entries:
            apply_ops(store, e.ops)
        first, second = entries[order[0]], entries[order[1]]
        apply_ops(store, unmerge_ops(first, still_active=[second]))
        assert store.get_alias("ref-j", "juniper") is not None  # the other merge still needs it
        apply_ops(store, unmerge_ops(second, still_active=[]))
        assert store.snapshot() == before


def test_planner_refuses_a_node_already_merged_elsewhere():
    from orion.person_model.merge_log import plan_entries

    store = MemStore()
    store.set_prop("n1", "merged_into", "someone-else")
    cfg = L.load_config()
    d = L.decide(L.Anchor("ref-j", "jun"), L.Candidate("n1", "jun"), cfg, L.LinkageContext()).model_copy(update={"relation": "same_as"})
    entries, conflicts = plan_entries([d], store, now_iso="x")
    assert not entries and len(conflicts) == 1


def test_not_home_requires_the_alias_to_exist():
    import pytest

    store = MemStore()
    cfg = L.load_config()
    d = L.stated_home_conflicts([{"node_id": "ref-x", "alias_norm": "home", "referent_kind": "place"}], person=ANCHOR,
                                stated_home="Springfield", place_labels={"ref-x": "shelby"}, cfg=cfg, stated_ref="a4")[0]
    with pytest.raises(ValueError):
        build_entry(d, store, decided_by="t", decision="auto", now_iso="x")
