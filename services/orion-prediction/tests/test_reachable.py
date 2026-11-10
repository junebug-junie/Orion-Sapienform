from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from app import reachable as R

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _bursty(days: int, seed: int = 0, local_hours=(19, 20, 21)) -> list[datetime]:
    """Juniper-like: on ~1 day in 3, a burst of 2-5 messages a few minutes apart, at an evening hour."""
    rng = np.random.default_rng(seed)
    out = []
    for d in range(days):
        if rng.random() < 0.33:
            day = (T0 + timedelta(days=d)).astimezone(R.LOCAL_TZ).replace(hour=int(rng.choice(local_hours)),
                                                                          minute=int(rng.integers(0, 40)))
            t = day.astimezone(timezone.utc)
            for _ in range(int(rng.integers(2, 6))):
                out.append(t)
                t += timedelta(minutes=int(rng.integers(5, 25)))
    return out


def test_no_history_returns_prior_for_all_three():
    f = R.forecast([], T0)
    assert f["p"] == f["p_state"] == f["p_baseline"] == R.PRIOR_POS / R.PRIOR_HOURS
    assert f["state_bucket"] == len(R.STATE_EDGES_H) and f["hours_since"] is None


def test_forecast_ignores_messages_at_or_after_origin():
    msgs = _bursty(30)
    origin = R.hour_floor(msgs[-1])
    a = R.forecast(msgs, origin)
    b = R.forecast([m for m in msgs if m < origin] + [origin + timedelta(minutes=m) for m in (0, 5, 70)], origin)
    assert a == b


def test_state_is_hours_since_last_message_and_local_hour_is_denver():
    m = datetime(2026, 11, 10, 2, 30, tzinfo=timezone.utc)   # 19:30 MST on 11-09
    f = R.forecast([m], datetime(2026, 11, 10, 4, 0, tzinfo=timezone.utc))
    assert f["hours_since"] == 1.5 and f["state_bucket"] == 1
    assert f["local_hour"] == 21


def test_mid_conversation_reads_far_higher_than_after_two_quiet_days():
    msgs = _bursty(60)
    last = msgs[-1]
    hot = R.forecast(msgs, R.hour_floor(last) + timedelta(hours=1))
    cold = R.forecast(msgs, R.hour_floor(last) + timedelta(days=3))
    assert hot["state_bucket"] == 0 and cold["state_bucket"] == 4
    assert hot["p"] > 5 * cold["p"]


def test_outcome_window_is_half_open():
    o = T0
    assert R.outcome([o], o) == (True, 1)
    assert R.outcome([o + R.WINDOW], o) == (False, 0)
    assert R.outcome([o - timedelta(seconds=1)], o) == (False, 0)


def _replay(msgs, days):
    rows = []
    for h in range(24 * 7, 24 * days):
        o = T0 + timedelta(hours=h)
        f = R.forecast(msgs, o)
        rows.append({**f, "happened": R.outcome(msgs, o)[0]})
    return rows


def test_verdict_pending_below_min_days():
    msgs = _bursty(20)
    v = R.verdict(_replay(msgs, 20))
    assert v["all"]["verdict"] == "pending" and v["all"]["n_days"] < R.MIN_DAYS
    assert v["served"] == R.MODEL


def test_bursty_history_beats_base_rate_overall_on_replay():
    msgs = _bursty(70, seed=1)
    v = R.verdict(_replay(msgs, 70))
    assert v["all"]["verdict"] == "better", v["all"]
    assert v["all"]["skill_vs_baseline"] > 0
    # cold hours exist and are graded separately
    assert v["cold"]["n_hours"] > 0 and v["cold"]["n_hours"] < v["all"]["n_hours"]


def test_random_messages_are_not_better_than_base_rate():
    rng = np.random.default_rng(3)
    msgs = sorted(T0 + timedelta(minutes=int(x)) for x in rng.integers(0, 70 * 1440, 40))
    v = R.verdict(_replay(msgs, 70))
    assert v["all"]["verdict"] in ("unclear", "worse"), v["all"]


def test_served_falls_back_to_baseline_only_when_worse(monkeypatch):
    msgs = _bursty(40, seed=2)
    rows = _replay(msgs, 40)
    bad = [{**r, "p": 0.9 if not r["happened"] else 0.01} for r in rows]   # confidently wrong
    v = R.verdict(bad)
    assert v["all"]["verdict"] == "worse" and v["served"] == R.BASELINE
    assert R.served_p(bad[0], v["served"]) == bad[0]["p_baseline"]


def test_calibration_bins_cover_every_row():
    rows = _replay(_bursty(30), 30)
    assert sum(b["n"] for b in R.calibration(rows)) == len(rows)


class _Store:
    def __init__(self, msgs):
        self.msgs, self.rows = msgs, {}

    def juniper_messages(self, start, end):
        return [m for m in self.msgs if start <= m < end]

    def reachable_exists(self, origin):
        return any(r["origin"] == origin for r in self.rows.values())

    def insert_reachable(self, row):
        if row["forecast_id"] in self.rows:
            return None
        self.rows[row["forecast_id"]] = dict(row)
        return row["forecast_id"]

    def pending_reachable(self, before):
        return [r for r in self.rows.values() if r["settled_at"] is None and r["window_end"] <= before]

    def settle_reachable(self, fid, happened, n, now):
        r = self.rows[fid]
        if r["settled_at"] is not None:
            return 0
        r.update(happened=happened, n_in_window=n, settled_at=now)
        return 1

    def reachable_rows(self, since, settled_only):
        return [r for r in self.rows.values() if r["origin"] >= since and (r["settled_at"] or not settled_only)]


def test_tick_issues_once_per_hour_only_early_and_settles_after_grace():
    o = datetime(2026, 11, 10, 3, 0, tzinfo=timezone.utc)
    store = _Store([o - timedelta(minutes=30), o + timedelta(minutes=20)])
    rc = R.Reachable(store)
    assert rc.tick(o + timedelta(minutes=2))["issued"] == "reachable:2026-11-10T03"
    assert rc.tick(o + timedelta(minutes=5))["issued"] is None              # already issued
    assert rc.tick(o + timedelta(minutes=30))["issued"] is None             # too late in the hour
    assert rc.tick(o + timedelta(minutes=65))["settled"] == 0               # inside the grace
    p = rc.tick(o + timedelta(hours=1, minutes=11))
    assert p["settled"] == 1
    row = store.rows["reachable:2026-11-10T03"]
    assert row["happened"] is True and row["n_in_window"] == 1
    # the 03:00 forecast saw only the 02:30 message
    assert row["n_messages"] == 1 and row["state_bucket"] == 0
    live = rc.live(o + timedelta(hours=1, minutes=12))
    assert live["current"]["served_p"] > 0 and live["verdict"]["all"]["verdict"] == "pending"
