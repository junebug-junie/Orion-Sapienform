"""Personality graph reducer, rev 4: choices only (pure, no I/O)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from orion.personality import graph as g
from orion.personality.graph import (
    DEFAULT_PARAMS,
    Evidence,
    GraphParams,
    Node,
    apply_fade,
    choice_direction,
    choice_label,
    label_base,
    choice_style_from_rows,
    evidence_weight,
    fold,
    gist,
    pair_label,
    project_frame,
)
from orion.schemas.personality import DISPOSITION_LABEL_MAX_CHARS, FRAME_MAX_DISPOSITIONS, PersonalityFrameV1

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
DIM = 8


def axis(i: int, wobble: float = 0.0, j: int | None = None) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    if wobble:
        v[(i + 1) % DIM if j is None else j] = wobble
    return v


_n = [0]


def ev(direction, lane="prime", *, eid=None, at=NOW, label="people over pipeline", weight=1.0, text="") -> Evidence:
    _n[0] += 1
    eid = eid or f"choice:c{_n[0]}"
    return Evidence(evidence_id=eid, lane=lane, source_ref=f"act:{eid}", occurred_at=at, excerpt=label,
                    embedding=direction, weight=weight, text=text)


def node(**kw) -> Node:
    base = dict(disposition_id="d1", origin="choice", status="active", label="people over pipeline, 5 of 5",
                label_similarity=0.9, centroid=axis(0), centroid_n=5, evidence_count=5, lane_counts={"prime": 5},
                chosen_count=5, first_seen_at=NOW - timedelta(days=3), last_reinforced_at=NOW - timedelta(days=1),
                retired_at=None, updated_at=NOW, lane_recent={"prime": 4.0})
    base.update(kw)
    base.setdefault("weight", g._derived_weight(base["lane_recent"]))
    base.setdefault("history_weight", base["weight"])
    return Node(**base)


# --- evidence weight (revealed preference) ----------------------------------------------------


def test_weight_follows_market_rank():
    assert evidence_weight(1) == 0.25  # took the top: weak evidence
    assert evidence_weight(2) == 1.0 and evidence_weight(None) == 1.0
    assert evidence_weight(None, declined_all=True) == 1.0
    assert evidence_weight(1, declined_all=True) == 1.0  # declined_all ignores any rank
    assert evidence_weight(3, care_promoted=True) == 0.5  # runaway guard: half weight
    assert evidence_weight(3, picks_in_decision=4) == 0.25  # one decision weighs once
    assert evidence_weight(1, picks_in_decision=0) == 0.25  # never divides by zero


# --- direction --------------------------------------------------------------------------------


def test_direction_is_chosen_minus_mean_declined_unit():
    d = choice_direction(axis(0), [axis(1), axis(2)])
    assert d is not None and abs(sum(x * x for x in d) - 1.0) < 1e-9
    assert d[0] > 0 and d[1] < 0 and d[2] < 0 and abs(d[1] - d[2]) < 1e-12
    # Scale-free: unnormalised inputs give the same direction.
    d2 = choice_direction([5 * x for x in axis(0)], [[3 * x for x in axis(1)], [0.5 * x for x in axis(2)]])
    assert d2 == pytest.approx(d)


def test_shared_component_cancels():
    common = [0.6] * DIM
    a = [c + x for c, x in zip(common, axis(0))]
    b = [c + x for c, x in zip(common, axis(1))]
    d = choice_direction(a, [b])
    assert d is not None and d[0] > 0.6 and d[1] < -0.6 and all(abs(x) < 1e-9 for x in d[2:])


def test_direction_unusable_is_none():
    assert choice_direction(axis(0), []) is None
    assert choice_direction(axis(0), [axis(0)]) is None  # same point: no preference to read
    assert choice_direction(axis(0), [[1.0, 0.0]]) is None  # dimension mismatch
    assert choice_direction([float("nan")] * DIM, [axis(1)]) is None
    assert choice_direction([], [axis(1)]) is None


# --- labels: computed, never prose ------------------------------------------------------------


def test_gist_and_pair_label():
    assert gist("  peer   briefs refused budget ") == "peer briefs refused budget"
    assert gist("one two three four five six seven eight") == "one two three four five six"
    # A cut never ends on a word that needs the next one.
    assert gist("outward learning in record not in loop") == "outward learning in record"
    assert gist("peer briefs refused the budget of the loop") == "peer briefs refused the budget"
    assert gist("peer briefs were refused by the budget") == "peer briefs were refused"
    assert pair_label("people", ["pipeline"]) == "people over pipeline"
    assert pair_label("people", ["pipeline", "heat"]) == "people over pipeline and heat"
    assert pair_label("people", ["pipeline", "heat", "camera"]) == "people"  # no "(+k more)"
    assert pair_label("say nothing", ["a", " "]) == "say nothing over a"
    assert len(pair_label("x " * 200, ["y " * 200])) <= DISPOSITION_LABEL_MAX_CHARS


def test_label_from_a_realistic_curiosity_prior_choice():
    """Live 2026-11-13 (personality_choice, curiosity lane): Orion tested one prior of 16 shown. The
    option text is what the offer row holds, the prior made readable (``choices.readable_prior``)."""
    from orion.personality.choices import readable_prior

    chosen = readable_prior("self:turn_start_context_is_stale_snapshot_20261110")
    declined = [readable_prior(p) for p in (
        "self:camera_layer_mediation_not_closeness_20261108", "self:peer_briefs_refused_budget_20260930",
        "self:curiosity_dies_as_missing_final_frame_20261101")] + ["x"] * 12
    label = choice_label(pair_label(chosen, declined), 21, 21)
    assert label == "turn start context is stale snapshot, 21 of 21"
    assert "more)" not in label and "_" not in label and "self:" not in label
    # One prior turned down: the pair is named.
    assert pair_label(chosen, declined[:1]) == "turn start context is stale snapshot over camera layer mediation not closeness"


def test_old_labels_are_tidied_on_the_next_fold():
    old = "turn start context is stale snapshot over camera layer mediation not closeness (+13 more), 21 of 21"
    assert label_base(old) == "turn start context is stale snapshot"
    assert choice_label(old, 22, 22) == "turn start context is stale snapshot, 22 of 22"
    assert label_base("one two three four five six seven… over a b c d e f g…, 3 of 3") == \
        "one two three four five six over a b c d e f"
    assert label_base("people over pipeline, 11 of 14") == "people over pipeline"  # current shape unchanged


def test_choice_label_keeps_counts_whole_and_replaces_old_counts():
    assert choice_label("people over pipeline", 11, 14) == "people over pipeline, 11 of 14"
    assert choice_label("people over pipeline, 3 of 4", 4, 5) == "people over pipeline, 4 of 5"
    long = choice_label("w" * 300, 12, 120)
    assert len(long) == DISPOSITION_LABEL_MAX_CHARS and long.endswith(", 12 of 120")


# --- fold ---------------------------------------------------------------------------------------


def test_first_choice_founds_a_candidate_choice_node():
    res = fold([], [ev(axis(0), weight=1.0)], now=NOW)
    (n,) = res.nodes
    assert n.origin == "choice" and n.status == "candidate" and n.label == "people over pipeline, 1 of 1"
    assert n.weight == 1.0 and n.lane_recent == {"prime": 1.0} and n.chosen_count == 1
    (row,) = res.evidence_rows
    assert row.disposition_id == n.disposition_id and row.weight_delta == 1.0 and row.similarity == 1.0
    assert len(res.history) == 1


def test_similar_direction_reinforces_far_one_founds():
    p = DEFAULT_PARAMS
    near = axis(0, 0.3)
    near_cos = 1 / math.sqrt(1 + 0.09)
    assert near_cos >= p.match_threshold
    res = fold([node()], [ev(near, weight=0.25), ev(axis(3))], now=NOW)
    by = {n.disposition_id: n for n in res.nodes}
    d1 = by["d1"]
    assert d1.evidence_count == 6 and d1.weight == pytest.approx(4.25) and d1.label.endswith(", 6 of 6")
    assert len(by) == 2  # the orthogonal choice founded its own node


def test_reverse_choice_counts_against_the_node_label_n_of_m():
    reverse = [-x for x in axis(0)]
    res = fold([node()], [ev(reverse, label="pipeline over people")], now=NOW)
    by = {n.disposition_id: n for n in res.nodes}
    d1 = by["d1"]
    assert d1.opposed_count == 1 and d1.label == "people over pipeline, 5 of 6"
    assert d1.weight == 4.0  # opposition records, it does not subtract
    other = next(n for n in res.nodes if n.disposition_id != "d1")
    assert other.label == "pipeline over people, 1 of 1"
    row = next(r for r in res.evidence_rows)
    assert row.meta["opposed"] == ["d1"]


def test_backlog_choice_is_faded_by_its_own_age():
    old = ev(axis(0), at=NOW - timedelta(days=DEFAULT_PARAMS.half_life_days))
    res = fold([], [old], now=NOW)
    assert res.nodes[0].weight == pytest.approx(0.5)
    assert res.evidence_rows[0].weight_delta == pytest.approx(0.5)


def test_promotion_by_weighted_choices_not_lanes():
    p = DEFAULT_PARAMS
    # Three rank-1 picks (0.25 each) never promote: following the order reveals little.
    res = fold([], [ev(axis(0), weight=0.25) for _ in range(3)], now=NOW)
    assert res.nodes[0].status == "candidate" and res.nodes[0].weight == pytest.approx(0.75)
    # Three departures from one lane promote (no second lane needed).
    res = fold([], [ev(axis(0), weight=1.0) for _ in range(3)], now=NOW)
    assert res.nodes[0].status == "active" and res.nodes[0].weight == pytest.approx(p.promote_min_weight)


def test_candidate_ttl_and_active_fade_retire():
    cand = node(status="candidate", lane_recent={"prime": 1.0}, last_folded_at=NOW - timedelta(days=15),
                last_reinforced_at=NOW - timedelta(days=15))
    (out,) = apply_fade([cand], now=NOW).nodes
    assert out.status == "retired" and out.retired_at == NOW
    weak = node(lane_recent={"prime": 0.3}, updated_at=NOW - timedelta(days=14))
    (out,) = apply_fade([weak], now=NOW).nodes
    assert out.weight == pytest.approx(0.15) and out.status == "retired"


def test_retired_and_non_choice_nodes_are_never_touched():
    retired = node(disposition_id="r", status="retired", retired_at=NOW - timedelta(days=1))
    prose = node(disposition_id="p", origin="recurrence")
    res = fold([retired, prose], [ev(axis(0))], now=NOW + timedelta(days=1))
    assert {n.disposition_id for n in res.nodes} - {"r", "p"} and not {"r", "p"} & {n.disposition_id for n in res.nodes}
    assert res.evidence_rows[0].disposition_id not in {"r", "p"}


def test_bad_evidence_skipped_and_recorded():
    rows = fold([], [ev(axis(0), lane="reverie"), ev([], label="x"), ev(axis(0), label="  "),
                     ev(axis(0), weight=0.0), ev([1.0, 0.0])], now=NOW).evidence_rows
    reasons = sorted(r.meta.get("skip_reason") for r in rows if r.disposition_id is None)
    assert reasons == ["bad_direction", "bad_weight", "empty_label", "unknown_lane"]
    # The first valid direction fixes the graph's dimension; a later one of another size is skipped.
    ok = fold([], [ev([1.0, 0.0]), ev(axis(0))], now=NOW).evidence_rows
    assert sorted(str(r.meta.get("skip_reason")) for r in ok) == ["None", "direction_dim_mismatch"]


def test_label_moves_only_to_a_closer_choice():
    n = node(label_similarity=0.95)
    res = fold([n], [ev(axis(0, 0.5), label="further")], now=NOW)
    assert res.nodes[0].label == "people over pipeline, 6 of 6"
    n = node(label_similarity=0.5)
    res = fold([n], [ev(axis(0, 0.01), label="closer pick over something")], now=NOW)
    assert res.nodes[0].label == "closer pick over something, 6 of 6"


def test_weight_delta_sums_to_the_change():
    n = node()
    res = fold([n], [ev(axis(0, 0.1), weight=1.0), ev(axis(0, 0.2), weight=0.25)], now=NOW)
    assert sum(r.weight_delta for r in res.evidence_rows) == pytest.approx(res.nodes[0].weight - n.weight)


def test_inputs_not_mutated_and_deterministic():
    n = node()
    evs = [ev(axis(0, 0.1), eid="choice:a"), ev(axis(4), eid="choice:b")]
    r1 = fold([n], evs, now=NOW)
    r2 = fold([n], evs, now=NOW)
    assert n.evidence_count == 5 and n.label == "people over pipeline, 5 of 5"
    assert [x.disposition_id for x in r1.nodes] == [x.disposition_id for x in r2.nodes]


def test_params_documented():
    p = GraphParams()
    assert (p.follow_weight, p.depart_weight, p.care_promoted_factor) == (0.25, 1.0, 0.5)
    assert (p.match_threshold, p.oppose_threshold) == (0.5, 0.5)


# --- frame + choice style ---------------------------------------------------------------------


def test_project_frame_valid_sorted_capped():
    nodes = [node(disposition_id=f"a{i}", lane_recent={"prime": float(i + 1)}) for i in range(FRAME_MAX_DISPOSITIONS + 3)]
    nodes += [node(disposition_id="c", status="candidate"), node(disposition_id="x", orion_verdict="rejected")]
    frame = project_frame(nodes, now=NOW, choice_style=None, evidence_rows=42, warnings=["w"] * 30)
    PersonalityFrameV1.model_validate(frame.model_dump())
    assert len(frame.dispositions) == FRAME_MAX_DISPOSITIONS
    assert frame.dispositions[0].weight == FRAME_MAX_DISPOSITIONS + 3 and frame.dispositions[0].origin == "choice"
    assert "x" not in {d.disposition_id for d in frame.dispositions}
    assert frame.evidence_rows == 42 and len(frame.warnings) == 20


def _row(cid, rank, *, act=None, at=None, declined=False):
    return {"choice_id": cid, "act_id": act, "chosen_market_rank": rank, "declined_all": declined,
            "occurred_at": at or NOW - timedelta(days=1)}


def test_choice_style_counts_decisions_not_picks():
    rows = [
        _row("curiosity:r1:a", 1, act="curiosity_run:r1"), _row("curiosity:r1:b", 4, act="curiosity_run:r1"),
        _row("prime:t1", 2, act="prime:t1"), _row("outreach:o1", None, act="outreach:o1", declined=True),
        _row("prime:t2", 1, act=None),
        _row("prime:old", 1, act="prime:old", at=NOW - timedelta(days=20)),
        {"choice_id": "prime:iso", "act_id": "prime:iso", "chosen_market_rank": 1, "declined_all": False,
         "occurred_at": (NOW - timedelta(hours=1)).isoformat()},
    ]
    style = choice_style_from_rows(rows, now=NOW)
    # r1 counts once by its best rank (1); declined_all is "own way".
    assert (style.choices, style.took_top, style.went_own_way) == (5, 3, 2)
    assert choice_style_from_rows([], now=NOW) is None
