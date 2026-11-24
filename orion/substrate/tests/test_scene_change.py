"""Camera scene change (PE retire Patch 3): the two-stage rank rests where theory says it should.

orion/substrate/scene_change.py. Under exchangeable frames p2 is uniform, so the value
max(0, -ln p2 - 1) is exactly 0 with probability 1 - 1/e = 0.632, crosses 2.0 nats with
probability e^-3 = 0.050, and never exceeds ln(289) - 1 = 4.67.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from orion.substrate import scene_change as sc

T0 = datetime(2026, 11, 24, 0, 0, tzinfo=timezone.utc)


def _feed(scorer: sc.SceneChangeScorer, start: datetime, seconds: float, *, fps: float, rng, stream="cam0",
          tick_sec: float = 30.0, sampler_sec: float | None = None):
    """Feed frames at ``fps`` and tick every ``tick_sec``; return (ts, reading) at sampler passes."""
    out = []
    t = 0.0
    next_tick = tick_sec
    next_sample = sampler_sec
    frame_gap = 1.0 / fps
    while t < seconds:
        scorer.observe(stream, start + timedelta(seconds=t), rng.random())
        t += frame_gap
        if t >= next_tick:
            now = start + timedelta(seconds=next_tick)
            reading, _ = scorer.tick(now)
            if sampler_sec is None or next_tick >= next_sample:
                out.append((now, reading))
                if sampler_sec is not None:
                    next_sample += sampler_sec
            next_tick += tick_sec
    return out


def test_cap_is_derived_from_the_window_size() -> None:
    assert sc.SCENE_CHANGE_CAP_NATS == pytest.approx(math.log(289) - 1.0)
    assert sc.SCENE_CHANGE_CAP_NATS == pytest.approx(4.666, abs=1e-3)
    assert sc.excess_nats(1.0) == 0.0
    assert sc.excess_nats(math.exp(-1.0)) == pytest.approx(0.0)
    assert sc.excess_nats(math.exp(-3.0)) == pytest.approx(2.0)


def test_stage1_ranks_against_the_last_two_hours_of_time_not_frames() -> None:
    w = sc.Stage1Window()
    # 200 frames over 30 min: ready
    for i in range(200):
        assert w.observe(T0 + timedelta(seconds=9 * i), 0.5) is None or i >= 200
    ts = T0 + timedelta(seconds=9 * 200)
    assert w.ready(ts)
    # a frame larger than every past one gets the smallest p, (1 + 0) / (n + 1)
    p = w.observe(ts, 0.9)
    assert p == pytest.approx(1.0 / 201.0)
    # a frame equal to all the past ones: (1 + n) / (n + 1) = 1
    assert w.observe(ts + timedelta(seconds=1), 0.5) == pytest.approx(1.0)
    # 3 h later every old frame left the 2 h window: cold again
    assert w.observe(ts + timedelta(hours=3), 0.5) is None
    assert len(w) == 1


def test_stage1_needs_200_frames_and_30_minutes() -> None:
    w = sc.Stage1Window()
    for i in range(300):  # 300 frames in 5 min: enough frames, too little time
        w.observe(T0 + timedelta(seconds=i), 0.1)
    assert not w.ready(T0 + timedelta(seconds=300))
    w2 = sc.Stage1Window()
    for i in range(100):  # 100 frames over 50 min: enough time, too few frames
        w2.observe(T0 + timedelta(seconds=30 * i), 0.1)
    assert not w2.ready(T0 + timedelta(minutes=50))


def test_warming_until_stage2_holds_48_values_and_never_a_calm_one() -> None:
    rng = random.Random(1)
    scorer = sc.SceneChangeScorer()
    readings = _feed(scorer, T0, 30 * 60 + 3 * 3600, fps=0.2, rng=rng)
    # Before stage 1 is ready (first 30 min + 200 frames at 0.2 fps = 1000 s, so 30 min rules):
    early = [r for t, r in readings if t < T0 + timedelta(minutes=30)]
    assert early and all(r.value is None and r.missing_reason == sc.MISSING_WARMING for r in early)
    # 3 h after stage 1 warmed, stage 2 holds ~36 values (< 48): still warming, never 0.0
    assert all(r.value is None for _, r in readings)
    assert all(r.missing_reason == sc.MISSING_WARMING for _, r in readings)
    assert scorer.stage2 and len(scorer.stage2) < sc.STAGE2_MIN_VALUES


def test_calm_rests_at_theory_at_the_sampler_cadence() -> None:
    """Exchangeable frames, read every ~240 s like the history sampler: exact-0 share ~0.632,
    crossing 2.0 ~5 %, max <= cap. Bands = the eval's acceptance bands."""
    rng = random.Random(7)
    scorer = sc.SceneChangeScorer()
    readings = _feed(scorer, T0, 4 * 86400, fps=0.2, rng=rng, sampler_sec=240.0)
    values = [r.value for t, r in readings if r.value is not None and t > T0 + timedelta(days=1, hours=1)]
    scales = {r.scale for t, r in readings if r.value is not None and t > T0 + timedelta(days=1, hours=1)}
    assert scales == {"v2"}
    assert len(values) > 800
    zero = sum(v == 0.0 for v in values) / len(values)
    cross = sum(v > 2.0 for v in values) / len(values)
    assert 0.58 <= zero <= 0.72, zero
    assert 0.025 <= cross <= 0.065, cross
    assert max(values) <= sc.SCENE_CHANGE_CAP_NATS + 1e-9


def test_a_real_change_crosses_the_bar_and_ties_at_the_stage1_floor_limit_it() -> None:
    """A change bigger than anything in the last 2 h reaches the stage-1 floor p = 1 / (n + 1).
    Stage 2 ranks with ties counted (p2 = (1 + #{past <= m}) / (n + 1)), so the value is limited by
    how many past 5-min intervals also hit that floor (a new 2-h maximum): about 4 % of calm
    intervals at 60 frames per interval against ~1,440 in the window. Measured, not hidden."""
    rng = random.Random(3)
    scorer = sc.SceneChangeScorer()
    _feed(scorer, T0, 26 * 3600, fps=0.2, rng=rng)  # warm, full stage 2
    start = T0 + timedelta(hours=26)
    for i in range(12):
        scorer.observe("cam0", start + timedelta(seconds=5 * i), 5.0 + i)
    reading, _ = scorer.tick(start + timedelta(seconds=60))
    assert reading.scale == "v2"
    assert reading.value is not None and reading.value > 2.0  # crosses the curiosity bar
    ties = sum(1 for _, m in list(scorer.stage2)[:-1] if m <= reading.m)
    assert reading.p2 == pytest.approx((1 + ties) / (reading.n_stage2 + 1))


def test_scale_is_partial_until_stage2_spans_24h() -> None:
    scorer = sc.SceneChangeScorer()
    now = T0 + timedelta(hours=10)
    scorer.stage2.extend((now - timedelta(minutes=5 * i), 0.5) for i in range(120, 0, -1))
    assert scorer.stage2_scale(now) == "v2_partial"
    scorer.stage2.appendleft((now - timedelta(hours=23, minutes=56), 0.5))
    assert scorer.stage2_scale(now) == "v2"


def test_no_frame_in_the_trailing_five_minutes_is_missing_not_calm() -> None:
    rng = random.Random(5)
    scorer = sc.SceneChangeScorer()
    _feed(scorer, T0, 6 * 3600, fps=0.2, rng=rng)
    reading, sampled = scorer.tick(T0 + timedelta(hours=6, minutes=6))
    assert reading.value is None and reading.missing_reason == sc.MISSING_NO_FRAME
    assert not sampled  # an empty interval is never added to stage 2


def test_stage2_never_ranks_a_value_against_itself_and_keeps_at_most_288() -> None:
    scorer = sc.SceneChangeScorer()
    now = T0 + timedelta(days=2)
    scorer.stage2.extend((now - timedelta(minutes=5 * i), 0.01) for i in range(400, 0, -1))
    scorer.last_stage2_at = now - timedelta(minutes=5)
    scorer.recent.append((now - timedelta(seconds=10), 0.001))  # rarer than every past minimum
    reading, sampled = scorer.tick(now)
    assert sampled and len(scorer.stage2) <= sc.STAGE2_MAX_VALUES
    assert reading.n_stage2 <= sc.STAGE2_MAX_VALUES
    assert reading.p2 == pytest.approx(1.0 / (reading.n_stage2 + 1))
    assert reading.value <= sc.SCENE_CHANGE_CAP_NATS + 1e-9


def test_stage2_persistence_round_trip_drops_old_and_malformed_values() -> None:
    scorer = sc.SceneChangeScorer()
    now = T0 + timedelta(days=1)
    scorer.stage2.extend([(now - timedelta(hours=30), 0.2), (now - timedelta(hours=1), 0.3),
                          (now - timedelta(minutes=5), 0.4)])
    scorer.last_stage2_at = now - timedelta(minutes=5)
    payload = scorer.stage2_to_json()
    payload["values"].append(["not a time", 0.5])
    payload["values"].append([now.isoformat(), 7.0])  # not a p
    fresh = sc.SceneChangeScorer()
    assert fresh.restore_stage2(payload, now=now) == 2
    assert [m for _, m in fresh.stage2] == [0.3, 0.4]
    assert fresh.last_stage2_at == now - timedelta(minutes=5)
    assert fresh.restored
    cold = sc.SceneChangeScorer()
    assert cold.restore_stage2(None, now=now) == 0 and cold.restored
    assert cold.restore_stage2({"version": 2, "values": [[now.isoformat(), 0.5]]}, now=now) == 0


def test_store_key_is_never_a_camera_name() -> None:
    from orion.vision.stream_ids import is_url_like

    key = sc.stage2_store_key("node:substrate.perception")
    assert key == "scene_change_stage2|node:substrate.perception"
    assert "|" in key and not is_url_like(key)
