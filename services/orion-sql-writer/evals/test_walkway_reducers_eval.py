"""Eval: a synthetic walkway, three weeks long, through the pure reducers.

Not a unit test of one function -- a behavioral check of the whole pure
chain (clustering -> sightings -> rhythm fit -> plan -> score) on a street
with a real regular, random passers-by, and noise:

- one dog walked every weekday around 07:40 (+-8 min jitter)
- one neighbor at random times, a few days a week
- ~15 one-off passers-by a day, each with their own appearance
- embeddings are unit vectors with per-sighting noise

Checks:
1. Clustering recovers the two regulars as stable individuals and does not
   merge the passers-by into them (cluster count within a sane band).
2. The dog gets a weekday expectation near 07:40 with support >= 5 days;
   the random-time neighbor gets none (no expectation from noise).
3. Held-out week: the dog's expectations score >= 80% met.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.vision_individuals import CropRow, apply_batch
from app.vision_rhythm import fit_subject, plan_expectations, score_window

TZ = ZoneInfo("America/Denver")
DIM = 32
START = date(2026, 9, 7)  # Monday


def _unit(rng: random.Random) -> list[float]:
    v = [rng.gauss(0, 1) for _ in range(DIM)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _noisy(base, rng, sigma=0.12):
    v = [b + rng.gauss(0, sigma / math.sqrt(DIM)) for b in base]
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _street(days: int, seed: int = 7):
    rng = random.Random(seed)
    dog, neighbor = _unit(rng), _unit(rng)
    crops, dog_times = [], []
    k = 0

    def walk(t0, base, label):
        nonlocal k
        for j in range(4):  # 4 observations, 20s apart
            k += 1
            crops.append(CropRow(f"c{k}", f"o{k}", "walkway", t0 + timedelta(seconds=20 * j), label,
                                 (0, 0, 1, 1), "walkway", _noisy(base, rng), None, None))

    for i in range(days):
        d = START + timedelta(days=i)
        midnight = datetime(d.year, d.month, d.day, tzinfo=TZ)
        if d.weekday() < 5:
            t = (midnight + timedelta(minutes=7 * 60 + 40 + rng.uniform(-8, 8))).astimezone(timezone.utc)
            walk(t, dog, "dog")
            dog_times.append(t)
        if rng.random() < 0.5:
            walk((midnight + timedelta(minutes=rng.uniform(6 * 60, 21 * 60))).astimezone(timezone.utc),
                 neighbor, "person")
        for _ in range(15):
            walk((midnight + timedelta(minutes=rng.uniform(6 * 60, 21 * 60))).astimezone(timezone.utc),
                 _unit(rng), "person")
    return crops, dog_times


def test_walkway_reducers_eval() -> None:
    crops, dog_times = _street(21)
    res = apply_batch(crops, individuals={}, latest_sightings={}, no_embed_zones={"patio"},
                      match_threshold=0.80, merge_gap_sec=60, tz=TZ)
    by_sightings = sorted(res.individuals.values(), key=lambda i: -i.sighting_count)
    dog_ind, neighbor_ind = by_sightings[0], by_sightings[1]
    print(f"\nindividuals={len(res.individuals)} dog_sightings={dog_ind.sighting_count} "
          f"neighbor_sightings={neighbor_ind.sighting_count}")

    # 1. Regulars recovered, passers-by not merged into them.
    assert dog_ind.kind == "dog" and dog_ind.sighting_count == 15
    assert neighbor_ind.sighting_count >= 5 and neighbor_ind.sighting_count <= 21
    passersby = 21 * 15
    assert passersby * 0.9 <= len(res.individuals) - 2 <= passersby

    # 2. Fit on the first two weeks only.
    cutoff = datetime(2026, 9, 21, tzinfo=TZ).astimezone(timezone.utc)
    dog_starts = sorted(s.started_at for s in res.sightings.values()
                        if s.individual_id == dog_ind.individual_id and s.started_at < cutoff)
    watched = {START + timedelta(days=i) for i in range(14)}  # the camera ran every day
    fits = {dk: fit_subject(dog_starts, tz=TZ, day_kind=dk, observed_days=watched)
            for dk in ("weekday", "weekend", "any")}
    assert fits["weekday"] and not fits["weekend"]
    f = fits["weekday"][0]
    assert abs(f.peak_minute - (7 * 60 + 40)) <= 10 and f.support_days >= 5
    print(f"dog weekday peak={f.peak_minute // 60:02d}:{f.peak_minute % 60:02d} "
          f"width={f.end_offset_min - f.peak_minute}min conf={f.confidence}")

    nb_starts = [s.started_at for s in res.sightings.values()
                 if s.individual_id == neighbor_ind.individual_id and s.started_at < cutoff]
    for dk in ("weekday", "weekend", "any"):
        # Neither with the watched-days denominator nor without it (the
        # per-window support rule alone must reject coincidences).
        assert not fit_subject(nb_starts, tz=TZ, day_kind=dk, observed_days=watched)
        assert not fit_subject(nb_starts, tz=TZ, day_kind=dk)

    # 3. Held-out week: plan each day from the fit, score against real arrivals.
    outcomes = []
    for i in range(7):
        d = date(2026, 9, 21) + timedelta(days=i)
        now = datetime(d.year, d.month, d.day, 1, 0, tzinfo=TZ).astimezone(timezone.utc)
        for p in plan_expectations(fits, now=now, tz=TZ, horizon_h=20):
            hit = any(p.window_start <= t <= p.window_end for t in dog_times)
            outcomes.append(score_window(occurred=hit, census_frames=100))
    met = outcomes.count("met") / len(outcomes)
    print(f"held-out outcomes={outcomes} met_rate={met:.2f}")
    assert len(outcomes) == 5 and met >= 0.8
