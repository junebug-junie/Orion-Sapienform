import re

from orion.person_model import linkage as L
from orion.person_model.review_html import render
from orion.person_model.tests.conftest import ANCHOR


def _html(ctx, candidates, **kw):
    ds = L.run([ANCHOR], candidates, L.load_config(), ctx)
    return ds, render(ds, title="t", run_id="r1", **kw)


def test_self_contained_no_network(ctx, candidates):
    _, h = _html(ctx, candidates)
    assert not re.search(r"https?://", h)
    assert not re.search(r"<(link|img|iframe)\b|\bsrc=|fetch\(|XMLHttpRequest", h)


def test_groups_cards_controls_and_export(ctx, candidates):
    ds, h = _html(ctx, candidates)
    assert h.count("<article") == len(ds)
    for g in ("Auto-merge (4)", "Needs your call (1)", "Auto-reject (5)"):
        assert g in h
    assert h.count('value=approve') == len(ds) and h.count('value=reject') == len(ds)
    assert h.count("<textarea") == len(ds)
    assert 'id=export' in h and "text/csv" in h and "download" in h


def test_labels_are_escaped(ctx):
    evil = L.Candidate("x", "<script>alert(1)</script> juniper", "substrate_node", ("SubstrateNode", "Entity"), "p")
    _, h = _html(ctx, [evil])
    assert "<script>alert(1)" not in h
