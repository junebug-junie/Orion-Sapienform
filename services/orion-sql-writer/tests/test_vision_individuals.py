"""Pure-function tests for the individuals reducer, attention score, and asks."""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.vision_attention_score import (
    WEIGHTS,
    hour_history,
    score_sighting,
    unusual_time_component,
)
from app.vision_individuals import (
    CropRow,
    Individual,
    Sighting,
    apply_batch,
    ask_question,
    circular_mean_minute,
    cosine,
    local_day_start,
    patio_snapshot,
    remaining_ask_budget,
    should_ask,
    update_centroid,
)
from app.vision_individuals_loop import next_backoff

TZ = ZoneInfo("America/Denver")
T0 = datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc)  # 08:00 Denver


def _ids():
    c = itertools.count()
    return lambda: f"id{next(c)}"


def _crop(i, t, emb, *, label="person", zone="walkway", obs=None):
    return CropRow(
        crop_id=f"c{i}", observation_id=obs or f"o{i}", stream_id="walkway", observed_at=t, label=label,
        box_xyxy=(0, 0, 1, 1), zone=zone, embedding=tuple(emb) if emb is not None else None,
        embedding_ref=f"emb{i}" if emb is not None else None, artifact_id=f"art{i}",
    )


def _run(crops, individuals=None, latest=None, **kw):
    return apply_batch(
        crops, individuals=individuals or {}, latest_sightings=latest or {}, no_embed_zones={"patio"},
        match_threshold=kw.get("threshold", 0.8), merge_gap_sec=kw.get("gap", 60), tz=TZ, new_id=_ids(),
    )


# --- clustering -------------------------------------------------------------


def test_first_crop_opens_a_new_individual() -> None:
    r = _run([_crop(1, T0, [1.0, 0.0])])
    assert r.new_individual_ids == ["id0"]
    ind = r.individuals["id0"]
    assert ind.kind == "person" and ind.sighting_count == 1 and ind.distinct_days == 1


def test_similar_crop_joins_existing_and_dissimilar_opens_new() -> None:
    r = _run([_crop(1, T0, [1.0, 0.0]), _crop(2, T0 + timedelta(seconds=5), [0.95, 0.1]),
              _crop(3, T0 + timedelta(seconds=10), [0.0, 1.0])])
    assert len(r.individuals) == 2
    assert r.individuals["id0"].centroid_n == 2


def test_different_kind_never_joins() -> None:
    r = _run([_crop(1, T0, [1.0, 0.0]), _crop(2, T0 + timedelta(seconds=5), [1.0, 0.0], label="dog")])
    assert {i.kind for i in r.individuals.values()} == {"person", "dog"}


def test_two_crops_in_one_frame_are_two_individuals() -> None:
    r = _run([_crop(1, T0, [1.0, 0.0], obs="same"), _crop(2, T0, [1.0, 0.0], obs="same")])
    assert len(r.individuals) == 2


def test_centroid_update_is_a_renormalized_running_mean() -> None:
    c = update_centroid([1.0, 0.0], 1, [0.0, 1.0])
    assert c == pytest.approx([2 ** -0.5, 2 ** -0.5])
    assert sum(x * x for x in c) == pytest.approx(1.0)
    # weight of history grows with n
    c3 = update_centroid([1.0, 0.0], 3, [0.0, 1.0])
    assert cosine(c3, [1.0, 0.0]) > cosine(c, [1.0, 0.0])


def test_inputs_are_not_mutated() -> None:
    ind = Individual("A", "walkway", "person", [1.0, 0.0], 1, T0, T0, 1, 1)
    _run([_crop(1, T0 + timedelta(seconds=5), [0.9, 0.1])], individuals={"A": ind})
    assert ind.centroid == [1.0, 0.0] and ind.centroid_n == 1


# --- sightings --------------------------------------------------------------


def test_sighting_extends_within_merge_gap_and_splits_beyond_it() -> None:
    r = _run([_crop(1, T0, [1.0, 0.0]), _crop(2, T0 + timedelta(seconds=50), [1.0, 0.0]),
              _crop(3, T0 + timedelta(seconds=200), [1.0, 0.0])], gap=60)
    sights = sorted(r.sightings.values(), key=lambda s: s.started_at)
    assert len(sights) == 2
    assert sights[0].observation_count == 2 and sights[0].dwell_sec == 50
    assert r.individuals["id0"].sighting_count == 2


def test_sighting_extends_one_from_a_previous_tick() -> None:
    ind = Individual("A", "walkway", "person", [1.0, 0.0], 3, T0, T0, 1, 1)
    prev = Sighting("S", "A", "walkway", T0 - timedelta(seconds=30), T0, 3, {"walkway": 3})
    r = _run([_crop(1, T0 + timedelta(seconds=20), [1.0, 0.0], zone="mailbox")],
             individuals={"A": ind}, latest={"A": prev})
    s = r.sightings["S"]
    assert s.observation_count == 4 and s.dwell_sec == 50
    assert s.zone == "walkway"  # 3 walkway vs 1 mailbox
    assert r.individuals["A"].sighting_count == 1  # extended, not a new sighting


def test_zone_is_where_most_observations_were() -> None:
    crops = [_crop(i, T0 + timedelta(seconds=i), [1.0, 0.0], zone=z)
             for i, z in enumerate(["walkway", "mailbox", "mailbox", "mailbox"])]
    (s,) = _run(crops).sightings.values()
    assert s.zone == "mailbox" and s.zone_counts == {"walkway": 1, "mailbox": 3}


def test_distinct_days_counts_local_days_not_utc_days() -> None:
    # 05:30 UTC on 9/25 is 23:30 Denver on 9/24: same local day as T0.
    ind = Individual("A", "walkway", "person", [1.0, 0.0], 1, T0, T0, 1, 1)
    late = datetime(2026, 9, 25, 5, 30, tzinfo=timezone.utc)
    r = _run([_crop(1, late, [1.0, 0.0])], individuals={"A": ind})
    assert r.individuals["A"].distinct_days == 1
    next_day = datetime(2026, 9, 25, 14, 0, tzinfo=timezone.utc)
    r2 = _run([_crop(1, next_day, [1.0, 0.0])], individuals={"A": ind})
    assert r2.individuals["A"].distinct_days == 2


# --- patio ------------------------------------------------------------------


def test_patio_never_becomes_an_individual() -> None:
    r = _run([_crop(1, T0, [1.0, 0.0], zone="patio"), _crop(2, T0, None, zone="patio")])
    assert r.individuals == {} and r.sightings == {}
    assert len(r.patio_crops) == 2


def test_patio_snapshot_shape_and_state_machine() -> None:
    snap = patio_snapshot(prev=None, batch_last_seen_at=T0, batch_count=3, now=T0 + timedelta(seconds=30),
                          present_sec=120, grace_sec=600)
    assert snap["state"] == "present" and snap["subject"] == {"count": 3}
    for k in ("state", "since_sec", "last_seen_sec", "subject"):
        assert k in snap
    # JSON round trip (stored as presence_json), then time passes with no patio boxes.
    prev = json.loads(json.dumps(snap))
    later = patio_snapshot(prev=prev, batch_last_seen_at=None, batch_count=0, now=T0 + timedelta(seconds=300),
                           present_sec=120, grace_sec=600)
    assert later["state"] == "recent" and later["subject"] == {"count": 3}
    gone = patio_snapshot(prev=later, batch_last_seen_at=None, batch_count=0, now=T0 + timedelta(seconds=900),
                          present_sec=120, grace_sec=600)
    assert gone["state"] == "absent" and gone["subject"] == {"count": 0}
    still = patio_snapshot(prev=gone, batch_last_seen_at=None, batch_count=0, now=T0 + timedelta(seconds=1200),
                           present_sec=120, grace_sec=600)
    assert still["since_sec"] == 300.0  # state_since carried across ticks


def test_patio_snapshot_never_carries_pictures_or_vectors() -> None:
    snap = patio_snapshot(prev=None, batch_last_seen_at=T0, batch_count=1, now=T0, present_sec=120, grace_sec=600)
    blob = json.dumps(snap)
    assert "embedding" not in blob and "box" not in blob and "crop" not in blob


# --- attention score --------------------------------------------------------


def test_null_components_contribute_zero_and_stay_null() -> None:
    r = score_sighting(labeled=False, local_hour=3, dwell_sec=10, dwell_rare_sec=None, prior_sightings=0,
                       individual_history=None, kind_history=None)
    assert r.components["unusual_time"] is None and r.components["long_dwell"] is None
    assert r.score == pytest.approx(WEIGHTS["unknown"] + WEIGHTS["few_prior_sightings"])
    assert json.loads(json.dumps(r.components))["unusual_time"] is None


def test_unusual_time_null_under_five_days_of_support() -> None:
    hist = hour_history([(8, f"2026-09-0{d}") for d in range(1, 5)] * 3, "individual")  # 4 days
    assert unusual_time_component(3, hist) is None
    hist5 = hour_history([(8, f"2026-09-0{d}") for d in range(1, 6)], "individual")
    assert unusual_time_component(8, hist5) == 0.0
    assert unusual_time_component(3, hist5) == 1.0


def test_unusual_time_falls_back_to_kind_history() -> None:
    kind = hour_history([(8, f"2026-09-0{d}") for d in range(1, 8)], "kind")
    r = score_sighting(labeled=False, local_hour=2, dwell_sec=0, dwell_rare_sec=120, prior_sightings=0,
                       individual_history=None, kind_history=kind)
    assert r.basis == "kind" and r.components["unusual_time"] == 1.0


def test_every_component_logged_and_no_category_words() -> None:
    kind = hour_history([(8, f"2026-09-0{d}") for d in range(1, 8)], "kind")
    r = score_sighting(labeled=False, local_hour=1, dwell_sec=400, dwell_rare_sec=180, prior_sightings=0,
                       individual_history=None, kind_history=kind)
    assert set(r.components) == set(WEIGHTS)
    assert r.score >= 0.7
    text = " ".join(r.reasons).lower()
    for banned in ("suspicious", "stranger", "intruder", "threat"):
        assert banned not in text


def test_labeled_regular_scores_low() -> None:
    hist = hour_history([(8, f"2026-09-{d:02d}") for d in range(1, 21)], "individual")
    r = score_sighting(labeled=True, local_hour=8, dwell_sec=30, dwell_rare_sec=120, prior_sightings=20,
                       individual_history=hist, kind_history=None)
    assert r.score < 0.2


# --- asks -------------------------------------------------------------------


def test_should_ask_thresholds() -> None:
    assert should_ask(label=None, sighting_count=10, distinct_days=5, min_sightings=10, min_days=5)
    assert not should_ask(label=None, sighting_count=9, distinct_days=5, min_sightings=10, min_days=5)
    assert not should_ask(label=None, sighting_count=30, distinct_days=4, min_sightings=10, min_days=5)
    assert not should_ask(label="Mrs. K", sighting_count=30, distinct_days=9, min_sightings=10, min_days=5)


def test_ask_budget_comes_from_the_db_count() -> None:
    assert remaining_ask_budget(daily_cap=2, asked_today=0) == 2
    assert remaining_ask_budget(daily_cap=2, asked_today=2) == 0
    assert remaining_ask_budget(daily_cap=2, asked_today=5) == 0


def test_ask_day_starts_at_local_midnight() -> None:
    # 03:00 UTC 9/25 = 21:00 Denver 9/24 -> the day began 06:00 UTC 9/24.
    start = local_day_start(datetime(2026, 9, 25, 3, 0, tzinfo=timezone.utc), TZ)
    assert start == datetime(2026, 9, 24, 6, 0, tzinfo=timezone.utc)


def test_ask_question_wording() -> None:
    mean, r = circular_mean_minute([7 * 60 + 38, 7 * 60 + 42, 7 * 60 + 41])
    q = ask_question(kind="person", sighting_count=14, distinct_days=6, mean_minute=mean, r=r)
    assert q == "I have seen this same person 14 times over 6 days, usually around 07:40. Do you know who this is?"
    q2 = ask_question(kind="dog", sighting_count=12, distinct_days=5, mean_minute=None, r=0.0)
    assert "at different times of day" in q2 and q2.endswith("Do you know whose dog this is?")


def test_circular_mean_wraps_midnight() -> None:
    mean, r = circular_mean_minute([23 * 60 + 50, 10])
    assert mean in (0, 1439) and r > 0.9


def test_backoff_doubles_and_caps() -> None:
    assert next_backoff(60, None) == 120
    assert next_backoff(60, 120) == 240
    assert next_backoff(60, 3000) == 3600


def test_many_candidates_still_pick_the_true_match() -> None:
    import random

    rng = random.Random(3)
    others = {}
    for i in range(500):
        v = [rng.gauss(0, 1) for _ in range(64)]
        others[f"x{i}"] = Individual(f"x{i}", "walkway", "person", v, 1, T0, T0, 1, 1)
    target = [1.0] + [0.0] * 63
    others["T"] = Individual("T", "walkway", "person", target, 5, T0, T0, 1, 1)
    r = _run([_crop(1, T0 + timedelta(seconds=5), [0.98, 0.05] + [0.0] * 62)], individuals=others)
    assert r.new_individual_ids == [] and r.individuals["T"].centroid_n == 6


def test_sighting_evidence_is_a_crop_ref_never_a_whole_frame() -> None:
    (s,) = _run([_crop(1, T0, [1.0, 0.0])]).sightings.values()
    assert s.evidence_ref == "crop:c1"
