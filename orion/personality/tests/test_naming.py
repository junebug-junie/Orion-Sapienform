"""Self-naming, pure side: naming_due, build_packet, the own-name echo gate, apply_namings, frame filter.

Design: docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md, decisions
2026-10-20 (names go to every lane at once; own-name evidence adds no weight; grounding recorded only).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from orion.personality.graph import (
    DEFAULT_PARAMS,
    Evidence,
    NamingRow,
    Node,
    _derived_weight,
    apply_namings,
    fold,
    project_frame,
)
from orion.personality.naming import (
    build_packet,
    grounding,
    naming_due,
    packet_id,
    prior_verdict,
    situation_mix,
)
from orion.schemas.personality import DispositionNamingPacketV1

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
DIM = 8
CAP = math.expm1(DEFAULT_PARAMS.lane_cap)


def axis(i: int, wobble: float = 0.0, j: int | None = None) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    if wobble:
        v[(i + 1) % DIM if j is None else j] = wobble
    return v


def node(did="d1", **kw) -> Node:
    base = dict(disposition_id=did, origin="recurrence", status="active", label=f"label of {did}",
                label_similarity=0.9, centroid=axis(0), centroid_n=5, evidence_count=10,
                lane_counts={"dream": 5, "journal": 5}, chosen_count=0,
                first_seen_at=NOW - timedelta(days=5), last_reinforced_at=NOW - timedelta(hours=1),
                retired_at=None, updated_at=NOW)
    base.update(kw)
    base.setdefault("lane_recent", {"dream": 2.0, "journal": 2.0})
    base.setdefault("weight", _derived_weight(base["lane_recent"], DEFAULT_PARAMS))
    return Node(**base)


def log_row(did="d1", *, verdict="name", hub="accepted", at=NOW, nid=None, evidence_count=10, lanes=None, **kw):
    return {"naming_id": nid or f"n-{did}-{at.isoformat()}", "disposition_id": did, "verdict": verdict,
            "hub_check": hub, "written_at": at,
            "shown": {"evidence_count": evidence_count, "lane_counts": lanes or {"dream": 5, "journal": 5}}, **kw}


def ids(nodes):
    return [n.disposition_id for n in nodes]


# --- naming_due -------------------------------------------------------------------------------


def test_due_only_active_never_shown_strongest_first():
    nodes = [node("weak", lane_recent={"dream": 2.0}), node("strong"),
             node("cand", status="candidate"), node("gone", status="retired")]
    assert ids(naming_due(nodes, [], NOW)) == ["strong", "weak"]


def test_unusable_answer_retries_after_3_days_at_most_3_times():
    rows = [log_row(verdict="no_answer", at=NOW - timedelta(days=2))]
    n = node(shown_count=1)
    assert naming_due([n], rows, NOW) == []  # too soon
    assert ids(naming_due([n], rows, NOW + timedelta(days=1))) == ["d1"]
    refused = [log_row(verdict="name", hub="refused", at=NOW - timedelta(days=d), nid=f"r{d}") for d in (12, 8, 4)]
    # Three unusable tries: waits for doubled evidence, not for the calendar.
    assert naming_due([node(shown_count=3)], refused, NOW + timedelta(days=30)) == []
    assert ids(naming_due([node(shown_count=3, evidence_count=20)], refused, NOW + timedelta(days=30))) == ["d1"]


def test_applied_verdict_reshown_only_on_doubled_evidence_after_14_days():
    rows = [log_row(nid="v1", at=NOW - timedelta(days=20))]
    named = node(naming_id="v1", orion_verdict="named", verdict_evidence_count=10, shown_count=1)
    assert naming_due([named], rows, NOW) == []  # 20 days, evidence not doubled: never re-asked
    assert ids(naming_due([replace(named, evidence_count=20)], rows, NOW)) == ["d1"]
    soon = [log_row(nid="v1", at=NOW - timedelta(days=5))]
    assert naming_due([replace(named, evidence_count=40)], soon, NOW) == []  # 14-day floor
    rejected = replace(named, orion_verdict="rejected", evidence_count=25)
    assert ids(naming_due([rejected], rows, NOW)) == ["d1"]


def test_new_promoting_source_reshows_without_doubling():
    rows = [log_row(nid="v1", at=NOW - timedelta(days=20), lanes={"dream": 5, "journal": 5})]
    named = node(naming_id="v1", verdict_evidence_count=10, shown_count=1, evidence_count=12,
                 lane_recent={"dream": 2.0, "journal": 2.0, "chat": 2.0})
    assert ids(naming_due([named], rows, NOW)) == ["d1"]
    # Reverie is the substrate-narration source, which a shown journal lane may already have been.
    narr = replace(named, lane_recent={"dream": 2.0, "substrate_narration": 2.0})
    assert naming_due([narr], rows, NOW) == []
    # Below the promotion floor it does not count.
    assert naming_due([replace(named, lane_recent={"dream": 2.0, "chat": 1.0})], rows, NOW) == []


def test_never_shown_comes_before_retry_and_reshow():
    rows = [log_row("retry", verdict="no_answer", at=NOW - timedelta(days=4))]
    out = naming_due([node("retry", shown_count=1, lane_recent={"dream": 3.0, "journal": 3.0}),
                      node("fresh", lane_recent={"dream": 1.6})], rows, NOW)
    assert ids(out) == ["fresh", "retry"]


# --- build_packet ------------------------------------------------------------------------------


def ev_row(eid, lane, sim, *, excerpt=None, situation=None, at=NOW - timedelta(hours=1)):
    meta = {"situation": situation} if situation is not None else {}
    return {"evidence_id": eid, "lane": lane, "occurred_at": at, "excerpt": excerpt or f"text {eid}",
            "similarity": sim, "meta": meta}


def test_build_packet_one_per_lane_first_then_similarity_and_neighbors():
    rows = [ev_row("r1", "reverie", 0.99), ev_row("r2", "reverie", 0.98), ev_row("r3", "reverie", 0.97),
            ev_row("j1", "journal", 0.85), ev_row("c1", "chat", 0.83), ev_row("r4", "reverie", 0.96),
            ev_row("bad", "reverie", 0.999, excerpt="   ")]
    me = node("me", centroid=axis(0))
    others = [node("near", centroid=axis(0, 0.3)), node("far", centroid=axis(3)),
              node("mid", centroid=axis(0, 1.0), name="the long watch"),
              node("next", centroid=axis(0, 2.0)), node("dead", status="retired", centroid=axis(0)), me]
    pkt = build_packet(me, rows, others, None, NOW)
    assert isinstance(pkt, DispositionNamingPacketV1)
    assert [e.evidence_id for e in pkt.evidence] == ["r1", "j1", "c1", "r2", "r3"]
    assert [n.disposition_id for n in pkt.neighbors] == ["near", "mid", "next"]
    assert pkt.neighbors[1].label == "the long watch"  # a named neighbor shows its name
    assert pkt.packet_id == packet_id("me", ["r1", "j1", "c1", "r2", "r3"], NOW) and len(pkt.packet_id) == 16
    assert pkt.packet_id == hashlib.sha1(f"me|r1,j1,c1,r2,r3|{NOW.isoformat()}".encode()).hexdigest()[:16]
    assert pkt.evidence_count == 10 and pkt.label_excerpt == "label of me"
    assert DispositionNamingPacketV1.model_validate(pkt.model_dump(mode="json")) == pkt
    assert build_packet(me, [ev_row("x", "chat", 0.9, excerpt=" ")], others, None, NOW) is None


def test_situation_mix_and_coverage():
    rows = [ev_row("a", "reverie", 0.9, situation={"arousal": "strained", "day_phase": "dusk", "company": True}),
            ev_row("b", "reverie", 0.9, situation={"arousal": "strained", "day_phase": "afternoon", "company": False}),
            ev_row("c", "reverie", 0.9), ev_row("d", "reverie", 0.9, situation={"arousal": None})]
    mix, cov = situation_mix(rows)
    assert mix == {"arousal": {"strained": 2}, "day_phase": {"dusk": 1, "afternoon": 1},
                   "company": {"true": 1, "false": 1}}
    assert cov == 0.75
    assert situation_mix([]) == ({}, 0.0)


def test_prior_verdict_is_last_usable():
    rows = [log_row(nid="a", at=NOW - timedelta(days=9), name="the watch", sentence="x" * 30),
            log_row(nid="b", verdict="no_answer", at=NOW - timedelta(days=2))]
    pv = prior_verdict(rows, "d1")
    assert pv.verdict == "name" and pv.name == "the watch" and pv.at == NOW - timedelta(days=9)
    assert prior_verdict(rows, "other") is None


def test_grounding_rank_and_margin():
    nodes = [node("a", centroid=axis(0)), node("b", centroid=axis(1)), node("c", status="retired", centroid=axis(2))]
    rank, margin = grounding(axis(0, 0.2), "a", nodes)
    assert rank == 1 and margin > 0
    rank, margin = grounding(axis(1), "a", nodes)
    assert rank == 2 and margin < 0
    assert grounding(axis(0), "c", nodes) is None  # retired: not ranked


# --- own-name echo gate (fold) ------------------------------------------------------------------


def ev(emb, lane="chat", eid="e1", excerpt="something", text=""):
    return Evidence(evidence_id=eid, lane=lane, source_ref=f"s:{eid}", occurred_at=NOW - timedelta(minutes=1),
                    excerpt=excerpt, embedding=emb, text=text)


def test_echo_of_own_name_counts_but_adds_no_weight():
    n = node(name="The Long Watch", lane_recent={"dream": 2.0})
    res = fold([n], [ev(axis(0, 0.1), excerpt="I keep the long watch tonight.")], now=NOW)
    (out,) = res.nodes
    assert out.evidence_count == 11 and out.lane_counts["chat"] == 1
    assert "chat" not in out.lane_recent and out.weight == n.weight
    (row,) = res.evidence_rows
    assert row.meta["cites_own_name"] is True and row.weight_delta == 0.0 and row.disposition_id == "d1"
    assert out.label == n.label  # an echo never becomes the label
    # The full text counts too, not only the excerpt.
    res = fold([n], [ev(axis(0, 0.1), excerpt="First line.", text="First line. Then THE LONG WATCH again.")], now=NOW)
    assert res.evidence_rows[0].meta.get("cites_own_name") is True and "chat" not in res.nodes[0].lane_recent


def test_echo_gate_only_own_name_and_only_when_named():
    unnamed = node(lane_recent={"dream": 2.0})
    res = fold([unnamed], [ev(axis(0, 0.1), excerpt="the long watch")], now=NOW)
    assert res.nodes[0].lane_recent["chat"] == 1.0 and "cites_own_name" not in res.evidence_rows[0].meta
    other_named = [node("a", lane_recent={"dream": 2.0}), node("b", centroid=axis(4), name="the long watch")]
    res = fold(other_named, [ev(axis(0, 0.1), excerpt="the long watch")], now=NOW)
    a = next(x for x in res.nodes if x.disposition_id == "a")
    assert a.lane_recent["chat"] == 1.0  # another node's name is not this node's echo


# --- apply_namings ------------------------------------------------------------------------------


def nrow(verdict, did="d1", *, hub="accepted", nid=None, **kw):
    return NamingRow(naming_id=nid or f"n-{verdict}-{did}", disposition_id=did, verdict=verdict, hub_check=hub,
                     written_at=NOW - timedelta(minutes=5), **kw)


def test_name_sets_fields_never_weight():
    n = node()
    res = apply_namings([n], [nrow("name", name=" The Long Watch ", sentence="I keep watching the field at night.")],
                        {"n-name-d1": (2, -0.03)}, now=NOW)
    (out,) = res.nodes
    assert out.name == "The Long Watch" and out.name_sentence.startswith("I keep")
    assert out.orion_verdict == "named" and out.naming_id == "n-name-d1" and out.named_at == NOW
    assert (out.name_grounding_rank, out.name_grounding_margin) == (2, -0.03)
    assert out.weight == n.weight and out.lane_recent == n.lane_recent and out.status == n.status
    assert out.shown_count == 1 and out.last_shown_at == NOW - timedelta(minutes=5)
    assert out.verdict_evidence_count == 10 and res.history == []
    (o,) = res.outcomes
    assert (o.verdict, o.apply_status, o.grounding_rank) == ("name", "applied", 2)
    assert n.name is None  # inputs untouched


def test_reject_marks_only():
    n = node()
    res = apply_namings([n], [nrow("reject", sentence="These are machine status lines, not me.")], now=NOW)
    (out,) = res.nodes
    assert out.orion_verdict == "rejected" and out.status == "active" and out.weight == n.weight
    assert out.name is None and res.outcomes[0].apply_status == "applied"


def test_refused_and_no_answer_only_bookkeeping():
    n = node(shown_count=1)
    res = apply_namings([n], [nrow("name", hub="refused", name="x"), nrow("no_answer")], now=NOW)
    (out,) = res.nodes
    assert out.shown_count == 3 and out.name is None and out.orion_verdict is None and out.naming_id is None
    assert [(o.verdict, o.apply_status) for o in res.outcomes] == [("refused", "refused"), ("no_answer", "no_answer")]
    unknown = apply_namings([n], [nrow("name", did="nope")], now=NOW)
    assert unknown.nodes == [] and unknown.outcomes[0].apply_status == "unknown_disposition"


def test_merge_pools_clamped_and_retires_b():
    a = node("a", centroid=axis(0), centroid_n=3, evidence_count=6, lane_counts={"dream": 6},
             lane_recent={"dream": 3.0, "chat": 1.0}, status="candidate", chosen_count=1)
    b = node("b", centroid=axis(0, 0.5), centroid_n=1, evidence_count=4, lane_counts={"dream": 2, "chat": 2},
             lane_recent={"dream": 3.0, "chat": 0.6}, first_seen_at=NOW - timedelta(days=9))
    res = apply_namings([a, b], [nrow("merge", did="b", merge_with="a")], now=NOW)
    by = {n.disposition_id: n for n in res.nodes}
    ma, mb = by["a"], by["b"]
    assert ma.lane_recent == {"dream": CAP, "chat": 1.6}  # 6.0 clamped at the per-lane cap
    assert ma.weight == _derived_weight(ma.lane_recent, DEFAULT_PARAMS)
    assert ma.evidence_count == 10 and ma.lane_counts == {"dream": 8, "chat": 2} and ma.centroid_n == 4
    assert ma.first_seen_at == NOW - timedelta(days=9) and ma.chosen_count == 1
    assert abs(sum(x * x for x in ma.centroid) - 1.0) < 1e-9 and ma.centroid[1] > 0
    assert ma.status == "candidate"  # no promotion in the same tick
    assert mb.status == "retired" and mb.merged_into == "a" and mb.orion_verdict == "merged" and mb.retired_at == NOW
    assert {h.disposition_id for h in res.history} == {"a", "b"}
    assert res.outcomes[0].apply_status == "applied"
    # The next tick's fold promotes it (two sources at the floor, enough evidence).
    nxt = fold([ma], [], now=NOW + timedelta(minutes=10))
    assert nxt.nodes[0].status == "active"


def test_merge_too_far_or_missing_changes_nothing():
    a, b = node("a", centroid=axis(0)), node("b", centroid=axis(1))
    res = apply_namings([a, b], [nrow("merge", did="b", merge_with="a")], now=NOW)
    assert res.outcomes[0].apply_status == "merge_too_far"
    (onlyb,) = res.nodes
    assert onlyb.disposition_id == "b" and onlyb.status == "active" and onlyb.shown_count == 1
    assert onlyb.naming_id is None and res.history == []
    miss = apply_namings([b], [nrow("merge", did="b", merge_with="zzz")], now=NOW)
    assert miss.outcomes[0].apply_status == "merge_target_missing"
    self_merge = apply_namings([b], [nrow("merge", did="b", merge_with="b")], now=NOW)
    assert self_merge.outcomes[0].apply_status == "merge_target_missing"


# --- frame ---------------------------------------------------------------------------------------


def test_frame_excludes_rejected_and_carries_name():
    nodes = [node("named", name="The Long Watch", name_sentence="I keep watching the field at night.",
                  lane_recent={"dream": 3.0, "journal": 3.0}),
             node("rejected", orion_verdict="rejected", lane_recent={"dream": 4.0, "journal": 4.0}),
             node("plain")]
    frame = project_frame(nodes, now=NOW, choice_style=None, evidence_rows=0)
    assert [d.disposition_id for d in frame.dispositions] == ["named", "plain"]
    assert frame.dispositions[0].name == "The Long Watch" and frame.dispositions[1].name is None
    assert frame.dispositions[0].name_sentence.startswith("I keep")
