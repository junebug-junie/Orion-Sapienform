"""Learning progress (choice-as-EFE design Step 1, acceptance S1.1 and S1.2)."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

from app import learning_progress as lp

TODAY = date(2026, 11, 20)
ROSTER = frozenset({"naive", "direct_ar"})
WIGGLE = (0.02, -0.03, 0.01, -0.01, 0.03, -0.02, 0.0)  # a calm day-to-day spread, same in both windows


def days_from(skills: list[float], mae_baseline: float = 0.2, pairs: int = 288) -> list[lp.DayStat]:
    """``skills[i]`` is the skill of TODAY - len + i (the last entry is yesterday)."""
    n = len(skills)
    return [lp.DayStat(TODAY - timedelta(days=n - i), pairs, mae_baseline * (1 - s), mae_baseline)
            for i, s in enumerate(skills)]


def rosters(n: int, change_at: int | None = None) -> dict[date, frozenset[str]]:
    out = {}
    for i in range(n):
        d = TODAY - timedelta(days=n - i)
        out[d] = ROSTER | ({"direct_arx_overlap"} if change_at is not None and i >= change_at else frozenset())
    return out


def test_flat_skill_reads_flat_rest_point():
    skills = [0.25 + w for w in WIGGLE] * 2
    p = lp.progress("s", days_from(skills), rosters(14), TODAY)
    assert p.status == "ok"
    assert abs(p.learning_progress) < 1e-9
    assert p.direction == "flat"


def test_gradual_improvement_reads_better():
    skills = [0.10 + 0.02 * i + w for i, w in enumerate(WIGGLE * 2)]
    p = lp.progress("s", days_from(skills), rosters(14), TODAY)
    assert p.status == "ok" and p.learning_progress > 0.1
    assert p.direction == "better" and p.t > p.t_crit > 2.0 and p.p_value < lp.ALPHA


def test_decline_reads_worse():
    skills = [0.40 - 0.02 * i + w for i, w in enumerate(WIGGLE * 2)]
    p = lp.progress("s", days_from(skills), rosters(14), TODAY)
    assert p.direction == "worse" and p.learning_progress < 0


def test_noisy_small_difference_reads_flat():
    # 0.03 better on average but daily swings of 0.2: not distinguishable from a hard week.
    recent = [0.3, -0.1, 0.35, 0.05, 0.4, 0.0, 0.3]
    prior = [0.25, 0.0, 0.3, -0.05, 0.35, 0.05, 0.2]
    p = lp.progress("s", days_from(prior + recent), rosters(14), TODAY)
    assert p.status == "ok" and p.direction == "flat"


def test_tiny_but_consistent_drift_reads_flat():
    # Champion serving the baseline: skill pinned near 0, tiny spread. A 0.02 drift is "significant" but not real.
    tiny = (0.001, -0.001, 0.0, 0.002, -0.002, 0.001, 0.0)
    skills = [0.01 + w for w in tiny] + [-0.01 + w for w in tiny]
    p = lp.progress("s", days_from(skills), rosters(14), TODAY)
    assert p.t < -p.t_crit and p.direction == "flat"


def test_t_between_normal_and_student_cut_reads_flat():
    # 5 days a side: 8 df, Student's cut 2.31. t = 2.25 passes the normal 2.0 but is not a real move here.
    base = (0.0, 0.12, 0.04, 0.16, 0.08)
    grid = days_from([0.0] * 14)
    days = ([lp.DayStat(grid[i].day, 288, 0.2 * (1 - (0.20 + b)), 0.2) for i, b in enumerate(base)]
            + [lp.DayStat(grid[7 + i].day, 288, 0.2 * (1 - (0.29 + b)), 0.2) for i, b in enumerate(base)])
    p = lp.progress("s", days, rosters(14), TODAY, rho=0.0)  # the Student cut itself, days independent
    assert p.status == "ok" and 2.0 < p.t < p.t_crit and p.direction == "flat"
    assert abs(p.t_crit - 2.306) < 0.01


def test_step_on_roster_change_is_not_credited():
    # S1.1: skill steps 0.05 -> 0.30 the day a model joins. That is our deploy, not Orion learning.
    skills = [0.05 + w for w in WIGGLE] + [0.30 + w for w in WIGGLE]
    p = lp.progress("s", days_from(skills), rosters(14, change_at=7), TODAY)
    assert p.status == "insufficient"
    assert p.learning_progress is None and p.direction is None
    assert p.roster_changed_on == TODAY - timedelta(days=7)
    # Without the roster change the same numbers would read as a big gain.
    assert lp.progress("s", days_from(skills), rosters(14), TODAY).direction == "better"


def test_roster_change_long_ago_leaves_both_windows_usable():
    skills = [0.25 + w for w in WIGGLE] * 3
    p = lp.progress("s", days_from(skills), rosters(21, change_at=2), TODAY)
    assert p.status == "ok" and p.direction == "flat"


def test_thin_days_and_today_are_excluded():
    skills = [0.25 + w for w in WIGGLE] * 2
    days = days_from(skills)
    days[-1] = lp.DayStat(days[-1].day, lp.MIN_PAIRS - 1, 0.0, 0.2)  # yesterday thin: would be skill 1.0
    days.append(lp.DayStat(TODAY, 288, 0.0, 0.2))  # today partial: never used
    p = lp.progress("s", days, rosters(14), TODAY)
    assert p.recent.days == 6 and p.direction == "flat"


def test_too_few_days_is_insufficient_not_zero():
    p = lp.progress("s", days_from([0.2, 0.3, 0.25]), rosters(3), TODAY)
    assert p.status == "insufficient" and p.learning_progress is None


def test_last_roster_change_ignores_days_without_fits():
    r = {date(2026, 11, 1): ROSTER, date(2026, 11, 3): ROSTER, date(2026, 11, 5): ROSTER | {"x"}}
    assert lp.last_roster_change(r) == date(2026, 11, 5)
    assert lp.last_roster_change({date(2026, 11, 1): ROSTER}) is None


def test_http_route_reads_both_series_and_refuses_others(monkeypatch):
    from datetime import datetime, timezone

    from fastapi.testclient import TestClient

    import app.main as main

    skills = [0.10 + 0.02 * i + w for i, w in enumerate(WIGGLE * 2)]
    seen = {}

    class Store:
        def daily_skill_pairs(self, series_ids, horizon_min, since):
            seen["pairs"] = (tuple(series_ids), horizon_min, since)
            return [{"series_id": sid, "day": d.day, "pairs": d.pairs, "mae_champion": d.mae_champion,
                     "mae_baseline": d.mae_baseline} for sid in series_ids for d in days_from(skills)]

        def daily_rosters(self, series_ids, since):
            return [{"series_id": sid, "day": d, "models": sorted(m)} for sid in series_ids
                    for d, m in rosters(14).items()]

    class Engine:
        clock = staticmethod(lambda: datetime(2026, 11, 20, 9, tzinfo=timezone.utc))

    monkeypatch.setattr(main, "engine", Engine())
    monkeypatch.setattr(main, "_store", Store())
    c = TestClient(main.app)
    body = c.get("/learning-progress").json()
    assert body["as_of"] == "2026-11-20" and body["horizon_min"] == 15
    assert [s["series_id"] for s in body["series"]] == list(lp.SERIES)
    assert all(s["direction"] == "better" for s in body["series"])
    # The day-to-day correlation is estimated from RHO_HISTORY_DAYS of history, not just the two windows.
    assert seen["pairs"][1] == 15 and seen["pairs"][2] == datetime(2026, 9, 24, tzinfo=timezone.utc)
    assert all(s["rho"] is not None and s["se_iid"] is not None and s["rho_pairs"] == 12
               and s["rho_source"] == "estimated" for s in body["series"])
    assert c.get("/learning-progress", params={"series_id": "office.temp_c"}).json()["series"][0]["series_id"] \
        == "office.temp_c"
    assert c.get("/learning-progress", params={"series_id": "node:prediction.chat_turn"}).status_code == 404
    monkeypatch.setattr(main, "engine", None)
    assert c.get("/learning-progress").status_code == 503


def ar1(seed: int, rho: float, n: int = 56, sd: float = 0.13) -> list[float]:
    """Daily skill around 0.3 with AR(1) weather spells and no real change."""
    rng = random.Random(seed)
    e, out = sd * rng.gauss(0, 1), []
    for _ in range(n + 20):
        e = rho * e + sd * math.sqrt(1 - rho * rho) * rng.gauss(0, 1)
        out.append(0.3 + e)
    return out[-n:]


def test_planted_weather_spell_fires_iid_rule_but_reads_flat():
    # #3691: no real change, but rho 0.6 spells. Independent days would call it "better" (t 2.7 > 2.2).
    d = days_from(ar1(21, 0.6))
    iid = lp.progress("s", d, rosters(56), TODAY, rho=0.0)
    p = lp.progress("s", d, rosters(56), TODAY)
    assert iid.direction == "better" and iid.t > iid.t_crit
    assert p.direction == "flat" and abs(p.t) < p.t_crit
    assert p.rho > 0.4 and p.rho_pairs == 48 and p.rho_source == "estimated" and p.se > p.se_iid
    assert p.learning_progress == iid.learning_progress and p.p_value > lp.ALPHA


def test_null_false_move_rate_holds_under_spells():
    # Seeded null at the live scale: the corrected rule stays near 5% per look; independent days do not. Both the
    # live regime today (14 days since the roster change: 12 pairs) and a full history (56 days: 48 pairs).
    for n_days in (14, 56):
        n, fired, fired_iid = 120, 0, 0
        for seed in range(n):
            d = days_from(ar1(1000 + seed, 0.3, n=n_days))
            fired += lp.progress("s", d, {}, TODAY).direction != "flat"
            fired_iid += lp.progress("s", d, {}, TODAY, rho=0.0).direction != "flat"
        assert fired / n <= 0.07 and fired_iid > fired, (n_days, fired, fired_iid)


def test_rho_zero_is_the_plain_welch_test():
    skills = [0.10 + 0.02 * i + w for i, w in enumerate(WIGGLE * 2)]
    p = lp.progress("s", days_from(skills), rosters(14), TODAY, rho=0.0)
    assert p.rho == 0.0 and p.rho_source == "given" and math.isclose(p.se, p.se_iid)


def test_estimate_rho_sees_spells_but_not_steps_or_trends():
    end = TODAY.toordinal()

    def pts(xs):
        return [(end - len(xs) + i, x) for i, x in enumerate(xs)]

    assert lp.moment_rho(pts(ar1(5, 0.6, n=196)), end)[0] > 0.4  # 28 weeks of spells
    assert lp.moment_rho(pts(ar1(6, 0.0, n=196)), end)[0] == 0.0
    assert lp.moment_rho(pts([0.3 + w for w in WIGGLE] * 8), end) == (0.0, 48)
    # A step at a window edge, or a steady trend, is change, not persistence.
    assert lp.moment_rho(pts([0.05 + w for w in WIGGLE] * 4 + [0.35 + w for w in WIGGLE] * 4), end)[0] == 0.0
    assert lp.moment_rho(pts([0.1 + 0.01 * i + w for i, w in enumerate(WIGGLE * 8)]), end)[0] == 0.0
    # The used rho is shrunk toward the prior: 12 pairs alone are too noisy to trust down to 0.
    rho, pairs, source = lp.estimate_rho(pts([0.3 + w for w in WIGGLE] * 2), end)
    assert source == "estimated" and pairs == 12
    assert math.isclose(rho, lp.RHO_PRIOR_WEIGHT * lp.RHO_PRIOR / (12 + lp.RHO_PRIOR_WEIGHT))
    # Too few consecutive pairs: the prior, not a guess from 3 points.
    assert lp.estimate_rho(pts([0.3, 0.2, 0.4, 0.3, 0.25]), end) == (lp.RHO_PRIOR, 4, "prior")


def test_roster_keys_see_a_config_change_under_the_same_names():
    d1, d2, d3 = date(2026, 10, 20), date(2026, 10, 21), date(2026, 10, 22)
    models = [{"series_id": "s", "day": d, "models": ["direct_arx", "naive"]} for d in (d1, d2, d3)]
    config = [{"series_id": "s", "day": d1, "gate_version": "3", "covariates": '["a"]', "overlap_covariates": None},
              {"series_id": "s", "day": d2, "gate_version": "3", "covariates": '["a", "b"]', "overlap_covariates": None}]
    keys = lp.roster_keys(models, config)
    r = {x["day"]: frozenset(x["models"]) for x in keys}
    assert lp.last_roster_change(r) == d2  # covariates widened, names unchanged
    assert r[d3] == r[d2]  # no blend run on d3: config carried forward, not a change
    assert {"direct_arx", "naive", "gate_version=3"} <= r[d1]
    # Name-only rosters would have missed it.
    assert lp.last_roster_change({x["day"]: frozenset(x["models"]) for x in models}) is None
