"""Learning progress (choice-as-EFE design Step 1, acceptance S1.1 and S1.2)."""

from __future__ import annotations

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
    assert p.direction == "better" and p.t >= lp.REST_T


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
    assert p.t <= -lp.REST_T and p.direction == "flat"


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
    assert seen["pairs"][1] == 15 and seen["pairs"][2] == datetime(2026, 11, 5, tzinfo=timezone.utc)
    assert c.get("/learning-progress", params={"series_id": "office.temp_c"}).json()["series"][0]["series_id"] \
        == "office.temp_c"
    assert c.get("/learning-progress", params={"series_id": "node:prediction.chat_turn"}).status_code == 404
    monkeypatch.setattr(main, "engine", None)
    assert c.get("/learning-progress").status_code == 503
