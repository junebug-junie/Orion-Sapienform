"""Street continuity rules (app/vision_handoff.py), pure.

docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md sections 2-4, 6.
Real-Postgres tests: tests/test_vision_handoff_pg.py.
"""

from __future__ import annotations

import math
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_handoff as vh  # noqa: E402
from orion.vision.topology import TopologyError, load_topology, parse_topology  # noqa: E402

T0 = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)
TOPO = load_topology(REPO / "config" / "vision_topology.yaml")
W2D = TOPO.by_id("walkway_to_door")
D2W = TOPO.by_id("door_to_walkway")
APPROACH = TOPO.by_id("approach_door")
CFG = vh.HandoffConfig()


def facts(**kw) -> vh.StoryFacts:
    base = dict(story_id="s1", stream_id="walkway", kind="person", started_at=T0 - timedelta(seconds=20),
                ended_at=T0, dir_lateral=-1, dir_depth=0, exit_edge="inside", entry_edge="inside",
                exit_foot_x=0.3, entry_foot_x=0.8, exit_speed=0.05)
    base.update(kw)
    return vh.StoryFacts(**base)


# --- topology -------------------------------------------------------------------------------------


def test_repo_topology_has_the_four_portals_juniper_described() -> None:
    assert [p.id for p in TOPO.portals] == ["central_walk", "approach_door", "door_to_walkway", "walkway_to_door"]
    assert TOPO.by_id("central_walk").streams == ("doorbell", "walkway") and TOPO.by_id("central_walk").max_skew_sec == 3
    assert D2W.from_edge == "right" and D2W.to_edge == "left" and D2W.lateral_sign == 1 and D2W.arrival_sign == 1
    assert W2D.from_edge == "left" and W2D.to_edge == "right" and W2D.lateral_sign == -1 and W2D.arrival_sign == -1
    assert APPROACH.from_dir_depth == 1 and set(APPROACH.to_events) == {"doorbell_motion", "doorbell_ring"}


@pytest.mark.parametrize("raw, msg", [
    ({"version": 2, "portals": []}, "version"),
    ({"version": 1, "portals": []}, "no portals"),
    ({"version": 1, "portals": [{"id": "a", "kind": "teleport"}]}, "kind"),
    ({"version": 1, "portals": [{"id": "a", "kind": "overlap", "streams": ["x", "y"], "max_skew_sec": 1},
                                {"id": "a", "kind": "overlap", "streams": ["x", "y"], "max_skew_sec": 1}]}, "duplicate"),
    ({"version": 1, "portals": [{"id": "g", "kind": "gap", "from": {"stream": "a", "edge": "top"},
                                 "to": {"stream": "b", "edge": "left"}, "prior_gap_frame_widths": 1}]}, "left or right"),
    ({"version": 1, "portals": [{"id": "g", "kind": "gap", "from": {"stream": "a", "edge": "left"},
                                 "to": {"stream": "a", "edge": "left"}, "prior_gap_frame_widths": 1}]}, "two different"),
    ({"version": 1, "portals": [{"id": "g", "kind": "gap", "from": {"stream": "a", "edge": "left", "exclude_dir_depth": 2},
                                 "to": {"stream": "b", "edge": "left"}, "prior_gap_frame_widths": 1}]}, "exclude_dir_depth"),
    ({"version": 1, "portals": [{"id": "p", "kind": "approach", "from": {"stream": "w", "dir_depth": 0},
                                 "to": {"stream": "d", "event": ["x"]}, "prior_sec": [3, 30]}]}, "dir_depth"),
    ({"version": 1, "portals": [{"id": "p", "kind": "approach", "from": {"stream": "w", "dir_depth": 1},
                                 "to": {"stream": "d", "event": ["x"]}, "prior_sec": [30, 3]}]}, "prior_sec"),
])
def test_bad_topology_is_refused_with_the_reason(raw, msg) -> None:
    with pytest.raises(TopologyError, match=msg):
        parse_topology(raw)


# --- open: projection from "inside" ---------------------------------------------------------------


def test_inside_exit_moving_toward_the_portal_edge_is_projected() -> None:
    # 35 of 36 live walkway stories end "inside": the last box is 30% from the left edge.
    dist, speed = vh.gap_departure(facts(), W2D, CFG)
    assert dist == pytest.approx(0.3) and speed == 0.05
    band = vh.choose_band("person", {}, {}, W2D, CFG)
    ws, we, late = vh.window_for(T0, dist, speed, band, CFG)
    # (0.3 + 0.4*1.5) / 0.05 = 18 s; (0.3 + 3*1.5) / 0.05 = 96 s; late margin 2 x 78 s.
    assert (ws - T0).total_seconds() == pytest.approx(18) and (we - T0).total_seconds() == pytest.approx(96)
    assert (late - we).total_seconds() == pytest.approx(156)


def test_a_doorbell_visitor_walking_up_to_the_door_is_not_a_sidewalk_departure() -> None:
    # Replay 2026-10-20 21:02 UTC: right and toward the door on the porch -> a false "missed".
    visitor = facts(stream_id="doorbell", dir_lateral=1, dir_depth=1, exit_foot_x=0.55, exit_speed=0.035)
    assert D2W.from_exclude_dir_depth == 1 and vh.gap_departure(visitor, D2W, CFG) is None
    assert vh.gap_departure(facts(stream_id="doorbell", dir_lateral=1, dir_depth=0, exit_foot_x=0.55,
                                  exit_speed=0.035), D2W, CFG) is not None


def test_exit_on_the_edge_needs_no_projection() -> None:
    assert vh.gap_departure(facts(exit_edge="left", exit_foot_x=0.0), W2D, CFG) == (0.0, 0.05)


@pytest.mark.parametrize("kw", [
    {"dir_lateral": 1},                  # walking the other way
    {"dir_lateral": 0},                  # not moving sideways
    {"exit_edge": "top"},                # left through another edge
    {"exit_edge": "right"},
    {"exit_foot_x": 0.8},                # too far from the edge to project
    {"exit_speed": None},                # fewer than two unclipped boxes
    {"exit_speed": 0.001},
    {"kind": "vehicle"},
    {"stream_id": "doorbell"},
    {"drop_reason": "prop"},
])
def test_not_a_departure(kw) -> None:
    assert vh.gap_departure(facts(**kw), W2D, CFG) is None


def test_inside_is_never_read_as_entering_a_door() -> None:
    # A story that ends "inside" but is not moving toward a portal edge opens nothing
    # (it is not a departure; and nothing here treats "inside" as a door).
    for portal in TOPO.of_kind("gap"):
        assert vh.gap_departure(facts(dir_lateral=0, exit_edge="inside"), portal, CFG) is None


def test_derived_edges_from_evidence_when_columns_are_missing() -> None:
    ev = [{"t": (T0 + timedelta(seconds=5 * i)).isoformat(), "box": [1800 - 250 * i, 600, 1900 - 250 * i, 1000],
           "crop_id": f"c{i}", "individual_id": "ind-a"} for i in range(4)]
    row = {"story_id": "s", "stream_id": "walkway", "kind": "person", "started_at": T0,
           "ended_at": T0 + timedelta(seconds=15), "dir_lateral": -1, "dir_depth": 0, "evidence": ev}
    f = vh.facts_from_row(row, (2560.0, 1920.0))
    assert f.edges_source == "derived" and f.exit_edge == "inside" and f.individual_id == "ind-a"
    assert f.exit_foot_x == pytest.approx(1100 / 2560, abs=1e-3)
    assert f.exit_speed == pytest.approx(750 / 2560 / 15, rel=1e-3)
    # Stored columns win when present.
    row.update(exit_edge="left", exit_foot_x=0.0, exit_speed_fw_per_sec=0.2)
    g = vh.facts_from_row(row, (2560.0, 1920.0))
    assert g.edges_source == "stored" and g.exit_edge == "left" and g.exit_speed == 0.2


def test_approach_needs_a_walkway_story_toward_the_house() -> None:
    assert vh.approach_departure(facts(dir_depth=1), APPROACH, CFG)
    assert not vh.approach_departure(facts(dir_depth=-1), APPROACH, CFG)
    assert not vh.approach_departure(facts(dir_depth=1, kind="bicycle"), APPROACH, CFG)
    band = vh.choose_band("person", {}, {}, APPROACH, CFG)
    ws, we, _ = vh.window_for(T0, 0.0, 1.0, band, CFG)
    assert ((ws - T0).total_seconds(), (we - T0).total_seconds()) == (3.0, 30.0)


# --- predict: band switch at 10 met ---------------------------------------------------------------


def test_prior_band_until_ten_met_then_conformal() -> None:
    gs = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    b9 = vh.choose_band("person", {"person": gs[:9]}, {"person": 9}, W2D, CFG)
    assert b9.source == "prior" and (b9.lo, b9.hi) == pytest.approx((0.6, 4.5))
    b10 = vh.choose_band("person", {"person": gs}, {"person": 10}, W2D, CFG)
    # n=10: order statistics floor(1.1)=1 and ceil(9.9)=10.
    assert b10.source == "learned" and (b10.lo, b10.hi) == (1.0, 1.9)


def test_a_dog_uses_the_person_band_widened_until_its_own_ten() -> None:
    gs = [1.0 + 0.1 * i for i in range(10)]
    b = vh.choose_band("dog", {"person": gs}, {"person": 10, "dog": 3}, W2D, CFG)
    assert b.source == "learned_person_widened" and b.lo < 1.0 and b.hi > 1.9
    own = vh.choose_band("dog", {"person": gs, "dog": [2.0 + 0.1 * i for i in range(10)]},
                         {"person": 10, "dog": 10}, W2D, CFG)
    assert own.source == "learned" and own.lo == 2.0
    # No person band either: the prior.
    assert vh.choose_band("dog", {}, {"dog": 2}, W2D, CFG).source == "prior"


def test_conformal_band_needs_enough_values() -> None:
    assert vh.conformal_band([1.0] * 5) is None
    assert vh.conformal_band(list(range(1, 21))) == (2, 19)


def test_slow_walkers_get_later_wider_windows() -> None:
    band = vh.choose_band("person", {}, {}, W2D, CFG)
    fast = vh.window_for(T0, 0.2, 0.1, band, CFG)
    slow = vh.window_for(T0, 0.2, 0.05, band, CFG)
    assert slow[0] > fast[0] and (slow[1] - slow[0]) > (fast[1] - fast[0])


# --- p-values: the calm state ----------------------------------------------------------------------


def test_prior_p_is_uniform_when_the_world_matches_the_prior() -> None:
    rng = random.Random(7)
    band = vh.choose_band("person", {}, {}, W2D, CFG)
    mu = (math.log(band.lo) + math.log(band.hi)) / 2
    sigma = (math.log(band.hi) - math.log(band.lo)) / (2 * vh.Z90)
    ps = [vh.p_value_arrival(math.exp(rng.gauss(mu, sigma)), band) for _ in range(4000)]
    assert 0.03 < sum(p < 0.05 for p in ps) / len(ps) < 0.07
    assert 0.45 < sum(p < 0.5 for p in ps) / len(ps) < 0.55


def test_learned_p_is_uniform_on_exchangeable_arrivals() -> None:
    # Conformal validity is marginal (over calibration sets), so draw many sets.
    rng = np.random.default_rng(3)
    ps = []
    for _ in range(300):
        calib = tuple(rng.lognormal(0.3, 0.3, 40))
        band = vh.Band(*vh.conformal_band(calib), "learned", calib)
        ps.extend(vh.p_value_arrival(float(g), band) for g in rng.lognormal(0.3, 0.3, 20))
    ps = np.array(ps)
    assert 0.035 < (ps < 0.05).mean() < 0.065 and 0.44 < (ps < 0.5).mean() < 0.56


def test_learned_p_has_a_granularity_floor_of_two_over_n_plus_one() -> None:
    # With n calibration values the smallest two-sided p is 2/(n+1): below n=39 an
    # arrival can never pass p < 0.05, however strange. The gate then cannot fire on
    # arrivals -- the honest answer with that little data.
    calib = tuple(float(x) for x in range(1, 11))
    band = vh.Band(1.0, 10.0, "learned", calib)
    assert vh.p_value_arrival(100.0, band) == pytest.approx(2 / 11)
    big = tuple(float(x) for x in range(1, 41))
    assert vh.p_value_arrival(100.0, vh.Band(1.0, 40.0, "learned", big)) < 0.05


def test_rest_state_is_reachable_an_on_time_arrival_is_not_surprising() -> None:
    band = vh.choose_band("person", {}, {}, W2D, CFG)
    centre = math.sqrt(band.lo * band.hi)
    assert vh.p_value_arrival(centre, band) == pytest.approx(1.0, abs=1e-6)
    assert vh.p_value_arrival(band.hi * 5, band) < 0.01
    assert vh.salience("handoff_met", "person", known_person=True, p=1.0) == 0.0


def test_missed_p_is_the_counted_chance_of_no_arrival() -> None:
    assert vh.p_value_missed(0, 0) == 0.5
    assert vh.p_value_missed(0, 18) == pytest.approx(0.05)
    assert vh.p_value_missed(9, 9) == 0.5


def test_attention_gate() -> None:
    assert vh.attention_gate(0.01, known_person=True, unusual_hour=False, cfg=CFG)
    assert vh.attention_gate(0.01, known_person=False, unusual_hour=True, cfg=CFG)
    assert not vh.attention_gate(0.01, known_person=False, unusual_hour=False, cfg=CFG)
    assert not vh.attention_gate(0.2, known_person=True, unusual_hour=True, cfg=CFG)
    assert not vh.attention_gate(None, known_person=True, unusual_hour=True, cfg=CFG)
    assert vh.unusual_for_hour(1, 28, 0.2) and not vh.unusual_for_hour(20, 28, 0.2)


# --- match ----------------------------------------------------------------------------------------


def cand(t_sec: float, score=None, channel=None, **kw) -> vh.Candidate:
    s = facts(story_id=f"t{t_sec}{score}", stream_id="doorbell", started_at=T0 + timedelta(seconds=t_sec),
              ended_at=T0 + timedelta(seconds=t_sec + 10), entry_edge="right", **kw)
    return vh.Candidate(s, score, channel)


WS, WE, LATE = T0 + timedelta(seconds=18), T0 + timedelta(seconds=96), T0 + timedelta(seconds=252)


def test_lone_candidate_in_window_is_met_on_timing() -> None:
    d = vh.decide_match([cand(40)], window_start=WS, window_end=WE, late_until=LATE, cfg=CFG)
    assert d.status == "met" and d.chosen.score is None


def test_an_arrival_just_before_the_band_is_early_not_missed() -> None:
    # A 10-90% band leaves ~10% of ordinary walkers before it: they must be found.
    early = vh.early_from(T0, WS, WE)
    assert early == T0   # one window width (78 s) before 18 s is before the departure: clamped
    d = vh.decide_match([cand(10)], window_start=WS, window_end=WE, late_until=LATE, cfg=CFG, early=early)
    assert d.status == "early"
    assert vh.decide_match([cand(10)], window_start=WS, window_end=WE, late_until=LATE, cfg=CFG).status == "none"
    assert vh.early_from(T0, T0 + timedelta(seconds=100), T0 + timedelta(seconds=130)) == T0 + timedelta(seconds=70)


def test_late_margin_is_late_and_outside_is_nothing() -> None:
    assert vh.decide_match([cand(150)], window_start=WS, window_end=WE, late_until=LATE, cfg=CFG).status == "late"
    assert vh.decide_match([cand(400)], window_start=WS, window_end=WE, late_until=LATE, cfg=CFG).status == "none"
    assert vh.decide_match([cand(5)], window_start=WS, window_end=WE, late_until=LATE, cfg=CFG).status == "none"


def test_two_candidates_without_appearance_are_ambiguous() -> None:
    d = vh.decide_match([cand(40), cand(60)], window_start=WS, window_end=WE, late_until=LATE, cfg=CFG)
    assert d.status == "ambiguous"


def test_unique_strong_appearance_wins_and_a_tie_is_ambiguous() -> None:
    d = vh.decide_match([cand(40, 0.95, "body"), cand(60, 0.80, "body")], window_start=WS, window_end=WE,
                        late_until=LATE, cfg=CFG)
    assert d.status == "met" and d.chosen.score == 0.95
    tie = vh.decide_match([cand(40, 0.95, "body"), cand(60, 0.94, "body")], window_start=WS, window_end=WE,
                          late_until=LATE, cfg=CFG)
    assert tie.status == "ambiguous"


def test_arrival_must_come_through_the_portal_edge() -> None:
    ok = facts(stream_id="doorbell", dir_lateral=-1, entry_edge="inside", entry_foot_x=0.75)
    assert vh.arrival_consistent(ok, W2D, CFG)
    assert vh.arrival_consistent(facts(stream_id="doorbell", dir_lateral=0, entry_edge="right"), W2D, CFG)
    assert not vh.arrival_consistent(facts(stream_id="doorbell", dir_lateral=1, entry_edge="inside",
                                           entry_foot_x=0.75), W2D, CFG)
    assert not vh.arrival_consistent(facts(stream_id="doorbell", dir_lateral=-1, entry_edge="inside",
                                           entry_foot_x=0.2), W2D, CFG)
    assert not vh.arrival_consistent(facts(stream_id="doorbell", dir_lateral=-1, entry_edge="left"), W2D, CFG)


def test_appearance_prefers_face_then_body_then_osnet() -> None:
    a = {"face": np.array([1.0, 0.0]), "body": np.array([0.0, 1.0])}
    b = {"face": np.array([1.0, 0.0]), "body": np.array([1.0, 0.0])}
    assert vh.appearance([a], [b]) == (pytest.approx(1.0), "face")
    assert vh.appearance([{"body": a["body"]}], [b]) == (pytest.approx(0.0), "body")
    assert vh.appearance([{"osnet": np.array([1.0, 1.0])}], [{"osnet": np.array([1.0, 1.0])}])[1] == "osnet"
    assert vh.appearance([], [b]) == (None, None)


# --- grade: coverage ------------------------------------------------------------------------------


def test_a_camera_that_was_not_looking_is_unscorable_never_missed() -> None:
    assert vh.grade_unmatched(0.79, CFG) == "unscorable"
    assert vh.grade_unmatched(0.0, CFG) == "unscorable"
    assert vh.grade_unmatched(0.8, CFG) == "missed"


def test_doorbell_coverage_counts_only_frames_delivered() -> None:
    # A single 5 s Ring burst inside a 78 s window is ~17% coverage with the 8 s tolerance.
    from app.vision_rhythm import coverage_fraction

    burst = [(WS.timestamp() + 10, WS.timestamp() + 15)]
    cov = coverage_fraction(burst, WS, WE, gap_tolerance_sec=CFG.census_gap_sec)
    assert cov == pytest.approx(13 / 78, abs=0.01)
    assert vh.grade_unmatched(cov, CFG) == "unscorable"


# --- overlap ---------------------------------------------------------------------------------------


def test_overlap_skew() -> None:
    door = facts(stream_id="doorbell", started_at=T0, ended_at=T0 + timedelta(seconds=10))
    walk_in = facts(started_at=T0 + timedelta(seconds=12), ended_at=T0 + timedelta(seconds=20))
    walk_out = facts(started_at=T0 + timedelta(seconds=14), ended_at=T0 + timedelta(seconds=20))
    assert vh.overlapping(door, walk_in, 3) and not vh.overlapping(door, walk_out, 3)
    assert vh.overlapping(door, facts(started_at=T0 - timedelta(seconds=20), ended_at=T0 - timedelta(seconds=2)), 3)
    assert not vh.overlapping(door, facts(started_at=T0 - timedelta(seconds=20), ended_at=T0 - timedelta(seconds=4)), 3)


def test_decide_overlap() -> None:
    assert vh.decide_overlap([], CFG).status == "unpaired"
    assert vh.decide_overlap([cand(1)], CFG).status == "paired"
    assert vh.decide_overlap([cand(1), cand(2)], CFG).status == "ambiguous"
    two = vh.decide_overlap([cand(1, 0.80, "body"), cand(2, 0.95, "body")], CFG)
    assert two.status == "paired" and two.chosen.score == 0.95


# --- misses ---------------------------------------------------------------------------------------


def test_priors_start_at_one_each() -> None:
    p = vh.hypothesis_priors({})
    assert set(p) == set(vh.HYPOTHESES) and all(v == pytest.approx(0.2) for v in p.values())
    p2 = vh.hypothesis_priors({"turned_back": 3})
    assert p2["turned_back"] == pytest.approx(4 / 8) and sum(p2.values()) == pytest.approx(1.0)


def test_hypothesis_deadlines() -> None:
    d = vh.hypothesis_deadlines(graded_at=T0 + timedelta(minutes=5), departed_at=T0, late_until=T0 + timedelta(minutes=4),
                                cfg=CFG)
    assert d["detector_miss"] == T0 + timedelta(minutes=5)
    assert d["late"] == d["unexplained"] == d["entered_place"] == T0 + timedelta(minutes=14)
    assert d["turned_back"] == T0 + timedelta(minutes=10)


def test_turned_back_needs_the_opposite_way_on_the_source_camera() -> None:
    src = facts(dir_lateral=-1)
    back = facts(story_id="s2", started_at=T0 + timedelta(seconds=30), ended_at=T0 + timedelta(seconds=40), dir_lateral=1)
    assert vh.turned_back(src, back, W2D)
    assert not vh.turned_back(src, facts(story_id="s3", started_at=T0 + timedelta(seconds=30), dir_lateral=-1), W2D)
    assert not vh.turned_back(src, facts(story_id="s4", stream_id="doorbell", started_at=T0 + timedelta(seconds=30),
                                         dir_lateral=1), W2D)
    assert not vh.turned_back(src, facts(story_id="s5", started_at=T0 + timedelta(seconds=30), dir_lateral=1,
                                         kind="dog"), W2D)
    away = facts(story_id="s6", started_at=T0 + timedelta(seconds=30), dir_depth=-1)
    assert vh.turned_back(facts(dir_depth=1), away, APPROACH)


def test_settle_final_cases() -> None:
    base = {"detector_miss": "false", "late": "open", "turned_back": "open", "entered_place": "open",
            "unexplained": "open"}
    out = vh.settle_final(base, last_zone="walkway", place_zones=("patio",))
    assert out == {"detector_miss": "false", "late": "false", "turned_back": "false", "entered_place": "false",
                   "unexplained": "true"}
    assert vh.settle_final(base, last_zone="patio", place_zones=("patio",))["entered_place"] == "true"
    assert vh.settle_final(base, last_zone="patio", place_zones=("patio",))["unexplained"] == "false"
    dm = dict(base, detector_miss="true")
    out = vh.settle_final(dm, last_zone="patio", place_zones=("patio",))
    assert out["unexplained"] == "false" and out["entered_place"] == "false"
    tb = dict(base, turned_back="true")
    assert vh.settle_final(tb, last_zone=None, place_zones=())["unexplained"] == "false"


# --- salience and street events -------------------------------------------------------------------


def test_salience_table_is_who_x_what_x_surprise() -> None:
    assert vh.salience("doorbell_ring", "person", known_person=False, p=None) == pytest.approx(0.6)
    assert vh.salience("approach_door", "person", known_person=True, p=None) == pytest.approx(0.6)
    assert vh.salience("handoff_missed", "person", known_person=False, p=0.05) == pytest.approx(0.4 * 0.6 * 0.95)
    assert vh.salience("handoff_missed", "dog", known_person=False, p=0.5) < \
        vh.salience("handoff_missed", "person", known_person=False, p=0.5)
    assert vh.salience("overlap_pair", "person", known_person=False, p=None) <= 0.1


def test_street_event_validates_against_the_schema() -> None:
    ev = vh.street_event(vh.StreetFacts(kind="handoff_missed", hid="h1", portal_id="walkway_to_door",
                                        stream_ids=["walkway", "doorbell"], occurred_at=T0, subject="bicycle",
                                        p_value=0.04, gate=True))
    assert ev["schema_version"] == "vision.street.event.v1" and ev["subject_kind"] == "person"
    assert "attention_gate:pass" in ev["evidence_refs"] and ev["correlation_id"] == vh.correlation_for("h1")
    assert ev["salience"] == pytest.approx(0.4 * 0.6 * 0.96, abs=1e-4)


def test_expect_value_keeps_rhythm_ids_and_the_latest_handoff_end() -> None:
    import json

    raw = json.dumps({"expectation_ids": ["e1"], "set_at": T0.isoformat()})
    v = json.loads(vh.merged_expect_value(raw, "h1", T0 + timedelta(seconds=60)))
    assert v["expectation_ids"] == ["e1"] and v["handoff_ids"] == ["h1"]
    v2 = json.loads(vh.merged_expect_value(json.dumps(v), "h2", T0 + timedelta(seconds=30)))
    assert v2["handoff_ids"] == ["h1", "h2"] and v2["handoff_until"] == (T0 + timedelta(seconds=60)).isoformat()
    from app.vision_rhythm import handoff_hold

    assert handoff_hold(json.dumps(v2), T0)[0] == 60 and handoff_hold(json.dumps(v2), T0 + timedelta(seconds=61)) == (None, {})


# --- never a label, never a merge ------------------------------------------------------------------


@pytest.mark.parametrize("name", ["vision_handoff.py", "vision_handoff_loop.py"])
def test_handoff_code_never_writes_a_label_or_merges_individuals(name: str) -> None:
    src = (SERVICE / "app" / name).read_text(encoding="utf-8")
    banned = [r"UPDATE\s+vision_individual\b", r"INSERT\s+INTO\s+vision_individual\b",
              r"INSERT\s+INTO\s+vision_individual_pair_verdict", r"_propagate_label", r"_apply_typed_name",
              r"_apply_pair_answer", r"UPDATE\s+vision_crop_observation", r"INSERT\s+INTO\s+vision_person_exemplar",
              r"\bSET\s[^\"]*\blabel\b"]
    for pat in banned:
        assert not re.search(pat, src, re.IGNORECASE), f"{name} matches {pat}"
