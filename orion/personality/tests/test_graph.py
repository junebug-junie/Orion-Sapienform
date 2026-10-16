from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from orion.personality.graph import (
    _derived_weight,
    DEFAULT_PARAMS,
    Evidence,
    GraphParams,
    Node,
    apply_fade,
    choice_style_from_rows,
    fold,
    project_frame,
    seed_nodes,
)
from orion.schemas.personality import FRAME_MAX_DISPOSITIONS, PersonalityFrameV1

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
DIM = 8


def _axis(i: int, wobble: float = 0.0, j: int | None = None) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    if wobble:
        v[(i + 1) % DIM if j is None else j] = wobble
    return v


_counter = [0]


def _ev(emb, lane="dream", *, eid=None, at=None, excerpt=None, meta=None) -> Evidence:
    _counter[0] += 1
    eid = eid or f"ev-{_counter[0]}"
    return Evidence(
        evidence_id=eid, lane=lane, source_ref=f"src:{eid}", occurred_at=at or NOW - timedelta(minutes=1),
        excerpt=excerpt or f"words {eid}", embedding=emb, meta=meta or {},
    )


def _node(**kw) -> Node:
    base = dict(
        disposition_id="d1", origin="recurrence", status="active", label="the body",
        label_similarity=0.9, centroid=_axis(0), centroid_n=5, evidence_count=5,
        lane_counts={"dream": 3, "journal": 2}, chosen_count=0,
        first_seen_at=NOW - timedelta(days=3), last_reinforced_at=NOW - timedelta(days=1),
        retired_at=None, updated_at=NOW,
    )
    base.update(kw)
    # Weight is derived from lane_recent; keep fixtures consistent with that.
    base.setdefault("lane_recent", {k: float(v) for k, v in base["lane_counts"].items()})
    base.setdefault("weight", _derived_weight(base["lane_recent"], DEFAULT_PARAMS))
    return Node(**base)


def test_nodes_born_in_batch_are_matchable_by_later_evidence():
    t0 = NOW - timedelta(hours=2)
    res = fold([], [_ev(_axis(2), at=t0 + timedelta(minutes=5)), _ev(_axis(2, 0.05), lane="journal", at=t0)],
               now=NOW)
    assert len(res.nodes) == 1
    node = res.nodes[0]
    assert node.evidence_count == 2 and node.first_seen_at == t0
    assert node.last_reinforced_at == t0 + timedelta(minutes=5)


def test_fade_with_no_elapsed_time_returns_nothing():
    res = apply_fade([_node(updated_at=NOW)], now=NOW)
    assert res.nodes == [] and res.history == []


def test_candidate_ttl_retires():
    n = _node(status="candidate", lane_counts={"dream": 1}, evidence_count=1, weight=1.0,
              last_reinforced_at=NOW - timedelta(days=8), updated_at=NOW)
    res = apply_fade([n], now=NOW)
    assert res.nodes[0].status == "retired" and res.nodes[0].retired_at == NOW
    assert res.history[0].status == "retired"
    fresh = _node(status="candidate", lane_counts={"dream": 1}, evidence_count=1,
                  last_reinforced_at=NOW - timedelta(days=6), updated_at=NOW)
    assert apply_fade([fresh], now=NOW).nodes == []


def test_retired_is_never_matched_or_faded():
    dead = _node(status="retired", retired_at=NOW - timedelta(days=1), updated_at=NOW - timedelta(days=30))
    res = fold([dead], [_ev(_axis(0), eid="again")], now=NOW)
    assert len(res.nodes) == 1
    assert res.nodes[0].disposition_id != "d1" and res.nodes[0].status == "candidate"
    assert res.evidence_rows[0].disposition_id == res.nodes[0].disposition_id


def test_label_updates_only_on_closer_evidence():
    n = _node(label="old label", label_similarity=0.9)
    far = fold([n], [_ev(_axis(0, 0.6), excerpt="farther")], now=NOW).nodes[0]  # sim ~0.857
    assert far.label == "old label"
    close = fold([n], [_ev(_axis(0, 0.1), excerpt="closer words")], now=NOW).nodes[0]  # sim ~0.995
    assert close.label == "closer words" and close.label_similarity == pytest.approx(1 / math.sqrt(1.01))


def test_centroid_cap_controls_drift():
    target = _axis(0, 0.5)
    params = GraphParams(centroid_n_cap=4)
    capped = fold([_node(centroid_n=1000)], [_ev(target)], now=NOW, params=params).nodes[0]
    uncapped = fold([_node(centroid_n=1000)], [_ev(target)], now=NOW,
                    params=GraphParams(centroid_n_cap=10_000)).nodes[0]
    assert capped.centroid[1] > uncapped.centroid[1] > 0
    assert math.isclose(math.sqrt(sum(x * x for x in capped.centroid)), 1.0)
    assert capped.centroid_n == 1001


def test_bad_embeddings_skipped_without_raising():
    bad = [
        _ev([], eid="empty"), _ev([0.0] * DIM, eid="zero"), _ev([float("nan")] + [1.0] * 7, eid="nan"),
        _ev([1.0, 0.0], eid="short"), _ev(["x"] * DIM, eid="str"), _ev(None, eid="none"),
        _ev(_axis(0), lane="not-a-lane", eid="lane"), _ev(_axis(0), excerpt="  ", eid="blank"),
        _ev(_axis(0), eid="naive", at=datetime(2026, 10, 16)),
    ]
    res = fold([_node(updated_at=NOW)], bad, now=NOW)
    assert res.nodes == [] and res.history == []
    skipped = {r.evidence_id: r for r in res.evidence_rows}
    assert "naive" not in skipped
    assert set(skipped) == {"empty", "zero", "nan", "short", "str", "none", "lane", "blank"}
    assert all(r.disposition_id is None and r.weight_delta == 0.0 for r in skipped.values())
    assert skipped["short"].meta["skip_reason"] == "embedding_dim_mismatch"


def test_first_evidence_fixes_dimension_when_graph_is_empty():
    res = fold([], [_ev([1.0, 0.0, 0.0], eid="a", at=NOW - timedelta(hours=1)), _ev(_axis(0), eid="b")], now=NOW)
    assert len(res.nodes) == 1 and len(res.nodes[0].centroid) == 3


def test_unnormalized_embedding_is_normalized():
    node = fold([], [_ev([3.0] + [0.0] * 7)], now=NOW).nodes[0]
    assert node.centroid[0] == pytest.approx(1.0)


def test_duplicate_evidence_ids_in_batch():
    res = fold([], [_ev(_axis(1), eid="dup"), _ev(_axis(1), eid="dup", lane="journal")], now=NOW)
    assert len(res.evidence_rows) == 1 and res.nodes[0].evidence_count == 1


def _active(i: int, w: float) -> Node:
    return _node(disposition_id=f"a{i}", weight=w, label="x" * 300 if i == 0 else f"label {i}")


def test_project_frame_valid_sorted_capped():
    nodes = [_active(i, float(i + 1)) for i in range(FRAME_MAX_DISPOSITIONS + 3)]
    nodes += [_node(disposition_id="c", status="candidate", weight=99.0),
              _node(disposition_id="r", status="retired", weight=99.0, retired_at=NOW)]
    frame = project_frame(nodes, now=NOW, choice_style=None, evidence_rows=42, warnings=["w"] * 30)
    PersonalityFrameV1.model_validate(frame.model_dump())
    assert len(frame.dispositions) == FRAME_MAX_DISPOSITIONS
    weights = [d.weight for d in frame.dispositions]
    assert weights == sorted(weights, reverse=True) and weights[0] == FRAME_MAX_DISPOSITIONS + 3
    assert {d.status for d in frame.dispositions} == {"active"}
    assert (frame.active_count, frame.candidate_count, frame.retired_count) == (FRAME_MAX_DISPOSITIONS + 3, 1, 1)
    assert frame.evidence_rows == 42 and len(frame.warnings) == 20


def test_project_frame_truncates_label():
    frame = project_frame([_active(0, 1.0)], now=NOW, choice_style=None, evidence_rows=0)
    assert len(frame.dispositions[0].label) == 160


def _choice_row(cid, rank, at=None, lane="curiosity_choice"):
    return {"lane": lane, "occurred_at": at or NOW - timedelta(days=1),
            "meta": {"choice_id": cid, "chosen_best_rank": rank}}


def test_choice_style_distinct_and_window():
    rows = [
        _choice_row("c1", 1), _choice_row("c1", 1),  # same choice, two evidence rows
        _choice_row("c2", 3), _choice_row("c3", None),
        _choice_row("c4", 1, at=NOW - timedelta(days=20)),  # outside window
        _choice_row("c5", 1, lane="dream"),
        {"lane": "curiosity_choice", "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
         "meta": '{"choice_id": "c6", "chosen_best_rank": 1}'},
        {"lane": "curiosity_choice", "occurred_at": NOW, "meta": {}},
    ]
    style = choice_style_from_rows(rows, now=NOW)
    assert (style.choices, style.took_top, style.went_own_way, style.window_days) == (4, 2, 2, 14)
    assert choice_style_from_rows([], now=NOW) is None
    assert choice_style_from_rows([_choice_row("old", 1, at=NOW - timedelta(days=30))], now=NOW) is None


# --- Habituation (2026-10-16): a lane's volume cannot buy weight; lane breadth does. --------------

def _run_plan(plan: list[dict[str, int]]) -> Node:
    """Fold one theme day by day: plan[day] = hits per lane that day. Return the node."""
    nodes: list[Node] = []
    for day, rate_by_lane in enumerate(plan):
        at = NOW + timedelta(days=day)
        batch = [
            _ev(_axis(0), lane, at=at - timedelta(seconds=i + 1))
            for lane, rate in rate_by_lane.items() for i in range(rate)
        ]
        res = fold(nodes, batch, now=at)
        changed = {n.disposition_id: n for n in res.nodes}
        nodes = [changed.get(n.disposition_id, n) for n in nodes] + [
            n for i, n in changed.items() if i not in {m.disposition_id for m in nodes}
        ]
    assert len(nodes) == 1
    return nodes[0]


def _run_days(rate_by_lane: dict[str, int], days: int) -> Node:
    return _run_plan([rate_by_lane] * days)


def test_node_born_from_self_inquiry_in_fold_is_a_seed():
    res = fold([], [_ev(_axis(5), "self_inquiry_seed")], now=NOW)
    assert res.nodes[0].origin == "seed_self_inquiry"


# --- Derived weight: expected values are computed by hand from the documented formula ---------
#   weight = sum over lanes of base(lane) * min(lane_cap, ln(1 + lane_recent[lane]))
#   base: lane_gain 1.0, curiosity_choice 2.0, self_inquiry_seed 0.5; lane_cap 1.5.
# These helpers deliberately do NOT call graph._derived_weight (that is the code under test).

CAP = 1.5
BASE = {"curiosity_choice": 2.0, "self_inquiry_seed": 0.5}
HALF = 0.5 ** (1 / 14)  # one day of fade at half_life_days=14
RMAX = math.expm1(CAP)  # a lane's recent-hit counter is clamped where its contribution caps


def _recent_after_days(per_day: int, days: int) -> float:
    """Hand model of one lane's clamped counter under _run_days: fade a day, then each hit +1."""
    r = 0.0
    for day in range(days):
        if day:
            r *= HALF
        for _ in range(per_day):
            r = min(r + 1.0, RMAX)
    return r


def _w(lane_recent: dict[str, float]) -> float:
    return sum(BASE.get(lane, 1.0) * min(CAP, math.log(1 + r)) for lane, r in lane_recent.items())


def test_params_match_the_documented_formula():
    p = DEFAULT_PARAMS
    assert (p.lane_gain, p.choice_gain, p.seed_weight, p.lane_cap) == (1.0, 2.0, 0.5, CAP)
    assert (p.promote_min_lanes, p.promote_lane_min_recent, p.promote_min_evidence) == (2, 1.5, 3)


def test_match_reinforces_and_far_evidence_creates_candidate():
    res = fold([_node()], [_ev(_axis(0, 0.1)), _ev(_axis(3))], now=NOW)
    by_id = {n.disposition_id: n for n in res.nodes}
    d1 = by_id["d1"]
    assert d1.evidence_count == 6 and d1.lane_counts["dream"] == 4
    assert d1.lane_recent == pytest.approx({"dream": RMAX, "journal": 2.0})  # 3 + 1 clamps
    assert d1.weight == pytest.approx(_w({"dream": 4, "journal": 2}))  # = 1.5 + ln 3 (dream capped)
    assert d1.weight == pytest.approx(CAP + math.log(3))
    new = [n for n in res.nodes if n.disposition_id != "d1"]
    assert len(new) == 1 and new[0].status == "candidate" and new[0].origin == "recurrence"
    assert new[0].disposition_id.startswith("disp-") and len(new[0].disposition_id) == 21
    assert new[0].weight == pytest.approx(math.log(2)) and new[0].evidence_count == 1
    assert new[0].lane_recent == {"dream": 1.0}
    assert new[0].label_similarity == 1.0 and new[0].centroid_n == 1
    rows = {r.disposition_id: r for r in res.evidence_rows}
    assert set(rows) == {"d1", new[0].disposition_id}
    # (g) weight_delta is the actual weight change, including the capped lane.
    assert rows["d1"].weight_delta == pytest.approx(_w({"dream": 4, "journal": 2}) - _w({"dream": 3, "journal": 2}))
    assert rows[new[0].disposition_id].weight_delta == pytest.approx(math.log(2))


def test_weight_delta_sums_to_the_actual_weight_change():
    # (g) mixed lanes, one already capped, plus a curiosity pick; no fade (now == updated_at).
    start = _node(lane_recent={"dream": 6.0, "journal": 1.0}, lane_counts={"dream": 6, "journal": 1})
    lanes = ["dream", "journal", "reading", "curiosity_choice", "journal", "curiosity_choice"]
    t0 = NOW - timedelta(hours=1)
    evs = [_ev(_axis(0), lane, at=t0 + timedelta(minutes=i)) for i, lane in enumerate(lanes)]
    res = fold([start], evs, now=NOW)
    node = res.nodes[0]
    recent = {"dream": 6.0, "journal": 1.0}
    expected_deltas = []
    for lane in lanes:
        before = _w(recent)
        recent[lane] = recent.get(lane, 0.0) + 1.0
        expected_deltas.append(_w(recent) - before)
    got = [r.weight_delta for r in sorted(res.evidence_rows, key=lambda r: r.occurred_at)]
    assert got == pytest.approx(expected_deltas)
    assert expected_deltas[0] == pytest.approx(0.0)  # dream already capped: a 7th hit buys nothing
    assert sum(got) == pytest.approx(node.weight - start.weight)
    assert node.weight == pytest.approx(_w(recent)) and node.chosen_count == 2


def test_curiosity_choice_gain_and_chosen_count():
    t0 = NOW - timedelta(hours=1)
    res = fold([], [_ev(_axis(1), "curiosity_choice", at=t0), _ev(_axis(1), "curiosity_choice", at=t0 + timedelta(minutes=1))],
               now=NOW)
    node = res.nodes[0]
    assert node.weight == pytest.approx(2.0 * math.log(3)) and node.chosen_count == 2
    first, second = sorted(res.evidence_rows, key=lambda r: r.occurred_at)
    assert first.weight_delta == pytest.approx(2.0 * math.log(2))
    assert second.weight_delta == pytest.approx(2.0 * (math.log(3) - math.log(2)))


def test_single_lane_flood_is_capped():
    # (a) 2,600 reverie hits over 7 days, folded day by day, weighs at most lane_cap * lane_gain.
    plan = [{"reverie": 372}] * 6 + [{"reverie": 368}]
    assert sum(d["reverie"] for d in plan) == 2600
    flood = _run_plan(plan)
    assert flood.evidence_count == 2600
    assert flood.weight <= CAP * DEFAULT_PARAMS.lane_gain + 1e-12
    assert flood.weight == pytest.approx(CAP)
    assert flood.status == "candidate"  # one lane can never promote


def test_four_lanes_two_recent_hits_each_beat_any_single_lane_flood():
    # (b) The strongest possible single-lane flood is the capped curiosity_choice lane (2.0 * 1.5).
    best_flood = max(CAP * b for b in (DEFAULT_PARAMS.lane_gain, DEFAULT_PARAMS.choice_gain, DEFAULT_PARAMS.seed_weight))
    assert best_flood == pytest.approx(3.0)
    t0 = NOW - timedelta(hours=1)
    evs = [_ev(_axis(2), lane, at=t0 + timedelta(minutes=i))
           for i, lane in enumerate(["dream", "journal", "reading", "chat"] * 2)]
    broad = fold([], evs, now=NOW).nodes[0]
    assert broad.weight == pytest.approx(4 * math.log(3))
    assert broad.weight > best_flood
    assert broad.status == "active"
    # Same against an actual 2,600-hit flood folded the slow way.
    assert broad.weight > _run_plan([{"reverie": 372}] * 6 + [{"reverie": 368}]).weight


def test_lane_breadth_beats_one_lane_flood_over_two_weeks():
    # The live dry run's reverie fixation (244/day) versus a theme four modest lanes keep returning to.
    flood = _run_days({"reverie": 244}, 14)
    rates = {"dream": 1, "journal": 2, "reading": 1, "chat": 1}
    broad = _run_days(rates, 14)
    # After 14 daily folds a lane holding r hits/day has r * (1 + f + ... + f^13) recent hits.
    expected = {lane: _recent_after_days(r, 14) for lane, r in rates.items()}
    assert broad.lane_recent == pytest.approx(expected)
    assert broad.weight == pytest.approx(_w(expected))
    assert flood.weight == pytest.approx(CAP)
    assert broad.weight > 2.5 * flood.weight
    assert broad.evidence_count < flood.evidence_count / 40
    assert (broad.status, flood.status) == ("active", "candidate")


def test_promotion_needs_two_lanes_with_two_recent_hits_and_three_evidence():
    t0 = NOW - timedelta(hours=1)
    two = fold([], [_ev(_axis(1), "dream", at=t0), _ev(_axis(1), "journal", at=t0 + timedelta(minutes=1))], now=NOW)
    assert two.nodes[0].status == "candidate"
    one_lane = fold([], [_ev(_axis(1), at=t0 + timedelta(minutes=i)) for i in range(4)], now=NOW).nodes[0]
    assert one_lane.status == "candidate"
    both = fold([], [_ev(_axis(1), lane, at=t0 + timedelta(minutes=i))
                     for i, lane in enumerate(["dream", "journal", "dream", "journal"])], now=NOW)
    assert both.nodes[0].status == "active"
    assert any(h.status == "active" for h in both.history)


def test_one_stray_second_lane_hit_does_not_promote_a_flood():
    # (c) 500 reverie hits + 1 journal echo stays candidate; a second journal hit promotes.
    t0 = NOW - timedelta(hours=2)
    evs = [_ev(_axis(3), "reverie", at=t0 + timedelta(seconds=i)) for i in range(500)]
    evs.append(_ev(_axis(3), "journal", at=t0 + timedelta(minutes=30)))
    res = fold([], evs, now=NOW)
    node = res.nodes[0]
    assert node.evidence_count == 501 and node.lane_recent == pytest.approx({"reverie": RMAX, "journal": 1.0})
    assert node.status == "candidate"
    promoted = fold([node], [_ev(_axis(3), "journal", at=NOW - timedelta(minutes=1))], now=NOW).nodes[0]
    assert promoted.lane_recent["journal"] == 2.0 and promoted.status == "active"


def test_promotion_counts_only_lanes_at_or_above_the_recent_floor():
    def tick(journal_recent: float) -> str:
        n = _node(status="candidate", lane_counts={"dream": 3, "journal": 2},
                  lane_recent={"dream": 3.0, "journal": journal_recent})
        res = apply_fade([n], now=NOW)
        return res.nodes[0].status if res.nodes else n.status
    assert tick(1.49) == "candidate"
    assert tick(1.5) == "active"
    # Two hits whose older one is past a half-life (1 + < 0.5) no longer count.
    faded = _node(status="candidate", lane_counts={"dream": 3, "journal": 2},
                  lane_recent={"dream": 3.0, "journal": 1.4}, updated_at=NOW - timedelta(days=1))
    res = apply_fade([faded], now=NOW)
    assert res.nodes[0].lane_recent["journal"] == pytest.approx(1.4 * HALF)
    assert res.nodes[0].status == "candidate"


def test_two_hits_in_separate_ticks_promote_a_lane():
    # Regression (2026-10-16): with a 2.0 floor the first hit had faded by the second (1.9997),
    # so a lane needed three hits unless two landed in one tick.
    n = _node(status="candidate", lane_counts={"dream": 3}, lane_recent={"dream": 3.0},
              evidence_count=3)
    first = fold([n], [_ev(_axis(0), "journal", at=NOW - timedelta(minutes=1))], now=NOW).nodes[0]
    assert first.status == "candidate"
    later = NOW + timedelta(days=3)
    second = fold([first], [_ev(_axis(0), "journal", at=later - timedelta(minutes=1))], now=later).nodes[0]
    assert 1.5 <= second.lane_recent["journal"] < 2.0
    assert second.status == "active"


def test_seeds_alone_never_promote_but_behavior_can():
    # (f)
    seeded = seed_nodes([_ev(_axis(4), lane="dream"), _ev(_axis(4, 0.05)), _ev(_axis(5))], now=NOW)
    assert len(seeded.nodes) == 2
    s = next(n for n in seeded.nodes if n.evidence_count == 2)
    assert s.origin == "seed_self_inquiry" and s.lane_counts == {"self_inquiry_seed": 2}
    assert s.lane_recent == {"self_inquiry_seed": 2.0}
    assert s.weight == pytest.approx(0.5 * math.log(3))
    assert all(r.lane == "self_inquiry_seed" for r in seeded.evidence_rows)
    many = seed_nodes([_ev(_axis(6, 0.01 * i)) for i in range(20)], now=NOW).nodes
    assert len(many) == 1 and many[0].status == "candidate"
    assert many[0].weight == pytest.approx(0.5 * CAP)
    # Seeds through fold() are also excluded from the lane count.
    via_fold = fold([], [_ev(_axis(7), "self_inquiry_seed", at=NOW - timedelta(minutes=i)) for i in range(1, 6)]
                    + [_ev(_axis(7), "dream", at=NOW - timedelta(seconds=i)) for i in (1, 2)], now=NOW).nodes[0]
    assert via_fold.lane_recent == pytest.approx({"self_inquiry_seed": RMAX, "dream": 2.0})
    assert via_fold.status == "candidate"
    # One behavioral lane with 2 hits on top of seeds is not enough; a second one is.
    s1 = fold([s], [_ev(_axis(4), "dream", at=NOW - timedelta(minutes=2)),
                    _ev(_axis(4), "dream", at=NOW - timedelta(minutes=1))], now=NOW).nodes[0]
    assert s1.evidence_count == 4 and s1.status == "candidate"
    s2 = fold([s1], [_ev(_axis(4), "reading", at=NOW - timedelta(seconds=2)),
                     _ev(_axis(4), "reading", at=NOW - timedelta(seconds=1))], now=NOW).nodes[0]
    assert s2.status == "active"
    assert s2.weight == pytest.approx(_w({"self_inquiry_seed": 2, "dream": 2, "reading": 2}))


def test_seed_does_not_match_preexisting_nodes_outside_the_call():
    res = seed_nodes([_ev(_axis(0))], now=NOW)
    assert len(res.nodes) == 1
    assert res.nodes[0].weight == pytest.approx(DEFAULT_PARAMS.seed_weight * math.log(2))


def test_fade_halves_lane_recent_and_weight_follows_the_formula():
    # (d)
    n = _node(lane_recent={"dream": 8.0, "journal": 2.0}, updated_at=NOW - timedelta(days=14))
    before = n.weight
    assert before == pytest.approx(CAP + math.log(3))
    res = apply_fade([n], now=NOW)
    faded = res.nodes[0]
    assert faded.lane_recent == pytest.approx({"dream": 4.0, "journal": 1.0})
    assert faded.weight == pytest.approx(_w({"dream": 4.0, "journal": 1.0}))  # = 1.5 + ln 2
    assert faded.weight < before
    assert faded.weight > before / 2  # the log makes weight fade sub-proportionally
    assert faded.updated_at == NOW
    assert n.weight == before and n.lane_recent == {"dream": 8.0, "journal": 2.0}  # input not mutated
    assert len(res.history) == 1 and res.history[0].weight == pytest.approx(faded.weight)


def test_active_retires_once_derived_weight_falls_below_threshold():
    # (e) recent 0.5 -> ln 1.5 = 0.405. After 7 days: ln(1 + 0.3536) = 0.303 (stays).
    # After 14 days: ln 1.25 = 0.223 < 0.25 (retires).
    def faded(days: float) -> Node:
        n = _node(lane_counts={"dream": 3, "journal": 2}, lane_recent={"dream": 0.5},
                  updated_at=NOW - timedelta(days=days))
        return apply_fade([n], now=NOW).nodes[0]
    week = faded(7)
    assert week.weight == pytest.approx(math.log(1 + 0.5 * 0.5 ** 0.5))
    assert week.weight >= DEFAULT_PARAMS.retire_below_weight and week.status == "active"
    gone = faded(14)
    assert gone.weight == pytest.approx(math.log(1.25)) and gone.weight < DEFAULT_PARAMS.retire_below_weight
    assert gone.status == "retired" and gone.retired_at == NOW


def test_history_threshold():
    tiny = _node(updated_at=NOW - timedelta(minutes=10))  # lane_recent {3, 2}: ~0.02% weight move
    res = apply_fade([tiny], now=NOW)
    assert len(res.nodes) == 1 and res.history == []
    big = _node(updated_at=NOW - timedelta(days=1))
    expected = _w({"dream": 3 * HALF, "journal": 2 * HALF})
    assert (big.weight - expected) / big.weight > 0.01
    hist = apply_fade([big], now=NOW).history
    assert len(hist) == 1 and hist[0].weight == pytest.approx(expected)
    born = fold([], [_ev(_axis(6))], now=NOW)
    assert born.history[0].weight == pytest.approx(math.log(2)) and born.history[0].at == NOW


def test_flooded_lane_counter_is_clamped_so_a_quiet_theme_fades_in_weeks():
    import math as _m

    flood = _run_days({"reverie": 500}, 3)
    assert flood.lane_recent["reverie"] == pytest.approx(_m.expm1(DEFAULT_PARAMS.lane_cap))
    # Then silence: an active node built the same way retires within ~10 weeks, not ~6 months.
    n = _node(status="active", lane_recent={"reverie": _m.expm1(1.5), "dream": _m.expm1(1.5)},
              lane_counts={"reverie": 1500, "dream": 6}, updated_at=NOW)
    # Two capped lanes: 2 * ln(1 + RMAX * f) < 0.25 once f < ~0.038, about 4.7 half-lives (~66 d).
    later = apply_fade([n], now=NOW + timedelta(days=70)).nodes[0]
    assert later.status == "retired"
    assert apply_fade([n], now=NOW + timedelta(days=21)).nodes[0].status == "active"
