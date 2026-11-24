"""Mechanics of the camera scene-change calm eval (PE retire Patch 3): v2-only bands, seed share,
displacement in full sets."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_scene_change_calm_eval as ev  # noqa: E402

ROOM, WALK = ev.CAMERAS


def _calm(n: int, seed: int = 1) -> list[float]:
    rng = random.Random(seed)
    return [max(0.0, -__import__("math").log((rng.randrange(288) + 1) / 289) - 1.0) for _ in range(n)]


def test_calm_values_pass_and_a_saturating_scale_fails():
    assert all(c.ok for c in ev.history_checks(ROOM, _calm(4000), min_rows=500))
    stuck = [0.0] * 300 + [4.67] * 700
    assert not all(c.ok for c in ev.history_checks(ROOM, stuck, min_rows=500))
    (c,) = ev.history_checks(ROOM, _calm(100), min_rows=500)
    assert c.ok is None


def _set(*seeds):
    return [{"focal_node_refs": [n], "notes": ["endogenous_seed", s]} for n, s in seeds]


def test_seed_share_band_and_total_and_displacement():
    other = ("node:prediction.room", "source:prediction_error")
    before = [_set(other, ("x", "source:repair_pressure"), ("y", "source:attention_open_loop"))] * 50
    after = [_set((ROOM, "source:scene_change"))] * 5 + [_set((WALK, "source:scene_change"))] * 5 + [_set()] * 90
    after += [_set(other, ("x", "source:repair_pressure"), ("y", "source:attention_open_loop"))] * 45
    checks = {c.name: c for c in ev.curiosity_checks(after, before)}
    assert checks[f"{ROOM} seeds in share of sets"].ok
    assert checks["camera seeds total (7 d)"].ok
    assert checks["displacement in full sets (worst drop)"].ok  # 50 -> 45 is -10 %
    worse = after[:100] + [_set(other, ("x", "source:repair_pressure"), ("z", "source:q"))] * 30
    checks = {c.name: c for c in ev.curiosity_checks(worse, before)}
    assert checks["displacement in full sets (worst drop)"].ok is False


def test_the_eval_sql_reads_v2_rows_only():
    assert "scale = 'v2'" in ev.ROWS_SQL
    assert "scale IN ('v2','v2_partial')" in ev.CUTOVER_SQL
