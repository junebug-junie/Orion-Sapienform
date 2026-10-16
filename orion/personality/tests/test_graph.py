from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from orion.personality.graph import (
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
        label_similarity=0.9, centroid=_axis(0), centroid_n=5, weight=4.0, evidence_count=5,
        lane_counts={"dream": 3, "journal": 2}, chosen_count=0,
        first_seen_at=NOW - timedelta(days=3), last_reinforced_at=NOW - timedelta(days=1),
        retired_at=None, updated_at=NOW,
    )
    base.update(kw)
    return Node(**base)


def test_match_reinforces_and_far_evidence_creates_candidate():
    res = fold([_node()], [_ev(_axis(0, 0.1)), _ev(_axis(3))], now=NOW)
    by_id = {n.disposition_id: n for n in res.nodes}
    assert by_id["d1"].evidence_count == 6
    assert by_id["d1"].lane_counts["dream"] == 4
    new = [n for n in res.nodes if n.disposition_id != "d1"]
    assert len(new) == 1 and new[0].status == "candidate" and new[0].origin == "recurrence"
    assert new[0].disposition_id.startswith("disp-") and len(new[0].disposition_id) == 21
    assert new[0].weight == pytest.approx(1.0) and new[0].evidence_count == 1
    assert new[0].label_similarity == 1.0 and new[0].centroid_n == 1
    rows = {r.disposition_id for r in res.evidence_rows}
    assert rows == {"d1", new[0].disposition_id}


def test_nodes_born_in_batch_are_matchable_by_later_evidence():
    t0 = NOW - timedelta(hours=2)
    res = fold([], [_ev(_axis(2), at=t0 + timedelta(minutes=5)), _ev(_axis(2, 0.05), lane="journal", at=t0)],
               now=NOW)
    assert len(res.nodes) == 1
    node = res.nodes[0]
    assert node.evidence_count == 2 and node.first_seen_at == t0
    assert node.last_reinforced_at == t0 + timedelta(minutes=5)


def test_saturation_multi_lane_beats_one_lane():
    one = fold([], [_ev(_axis(1)) for _ in range(5)], now=NOW).nodes[0]
    multi_ev = [_ev(_axis(1), lane=lane) for lane in ("dream", "journal", "reading") for _ in range(2)]
    multi = fold([], multi_ev, now=NOW).nodes[0]
    assert one.weight < multi.weight
    assert one.weight == pytest.approx(sum(1 / math.sqrt(1 + k) for k in range(5)))


def test_curiosity_choice_gain_and_chosen_count():
    node = fold([], [_ev(_axis(1), lane="curiosity_choice")], now=NOW).nodes[0]
    assert node.weight == pytest.approx(2.0) and node.chosen_count == 1


def test_promotion_needs_two_lanes_and_three_evidence():
    two_ev = fold([], [_ev(_axis(1), lane="dream"), _ev(_axis(1), lane="journal")], now=NOW).nodes[0]
    assert two_ev.status == "candidate"
    one_lane = fold([], [_ev(_axis(1)) for _ in range(4)], now=NOW).nodes[0]
    assert one_lane.status == "candidate"
    res = fold([], [_ev(_axis(1), lane="dream"), _ev(_axis(1), lane="journal"), _ev(_axis(1), lane="dream")],
               now=NOW)
    assert res.nodes[0].status == "active"
    assert any(h.status == "active" for h in res.history)


def test_seeds_cannot_promote_themselves_but_behavior_can():
    seeded = seed_nodes([_ev(_axis(4), lane="dream"), _ev(_axis(4, 0.05)), _ev(_axis(5))], now=NOW)
    assert len(seeded.nodes) == 2
    s = next(n for n in seeded.nodes if n.evidence_count == 2)
    assert s.origin == "seed_self_inquiry" and s.lane_counts == {"self_inquiry_seed": 2}
    assert s.status == "candidate"
    assert all(r.lane == "self_inquiry_seed" for r in seeded.evidence_rows)
    assert s.weight == pytest.approx(0.5 + 0.5 / math.sqrt(2))
    # One behavioral lane + seeds (3 evidence) is still not enough lanes.
    r1 = fold([s], [_ev(_axis(4))], now=NOW)
    s1 = r1.nodes[0]
    assert s1.evidence_count == 3 and s1.status == "candidate"
    s2 = fold([s1], [_ev(_axis(4), lane="reading")], now=NOW).nodes[0]
    assert s2.status == "active"


def test_seed_does_not_match_preexisting_nodes_outside_the_call():
    res = seed_nodes([_ev(_axis(0))], now=NOW)
    assert len(res.nodes) == 1 and res.nodes[0].weight == pytest.approx(DEFAULT_PARAMS.seed_weight)


def test_fade_halves_weight_in_one_half_life():
    n = _node(weight=4.0, updated_at=NOW - timedelta(days=14))
    res = apply_fade([n], now=NOW)
    assert res.nodes[0].weight == pytest.approx(2.0)
    assert res.nodes[0].updated_at == NOW
    assert n.weight == 4.0  # input not mutated
    assert len(res.history) == 1


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


def test_active_retires_below_weight():
    n = _node(weight=0.3, updated_at=NOW - timedelta(days=14))
    res = apply_fade([n], now=NOW)
    assert res.nodes[0].status == "retired" and res.nodes[0].weight == pytest.approx(0.15)


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


def test_history_threshold():
    tiny = _node(weight=4.0, updated_at=NOW - timedelta(minutes=10))  # fade ~0.003%
    res = apply_fade([tiny], now=NOW)
    assert len(res.nodes) == 1 and res.history == []
    big = _node(weight=4.0, updated_at=NOW - timedelta(days=1))  # ~4.8%
    assert len(apply_fade([big], now=NOW).history) == 1
    born = fold([], [_ev(_axis(6))], now=NOW)
    assert born.history[0].weight == pytest.approx(1.0) and born.history[0].at == NOW


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
